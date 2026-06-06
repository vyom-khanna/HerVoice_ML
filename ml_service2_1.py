"""
HerVoice - Spatio-Temporal Predictive Safety Engine
app/services/ml_service.py

SafetyPredictor encapsulates the full ML lifecycle:
  - Feature engineering (spatial, tags, grid cell, temporal, zone, night flag)
  - XGBoost Regressor training with optional GridSearchCV
  - joblib serialization / deserialization
  - Single-point inference with safe fallback

Schema update (hervoice_ml_training_data2.csv):
  - lat/lng           → renamed to latitude/longitude internally
  - zone_label        → label-encoded as zone_encoded (new)
  - is_night          → passed through as binary feature (new)
  - time_context      → stronger night signal; kept as direct feature
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RESOURCES_DIR      = Path("resources")
MODEL_PATH         = RESOURCES_DIR / "trained_model.pkl"
LABEL_ENC_PATH     = RESOURCES_DIR / "label_encoder.pkl"
ZONE_ENC_PATH      = RESOURCES_DIR / "zone_encoder.pkl"      # NEW: zone_label encoder
TAG_BIN_PATH       = RESOURCES_DIR / "tag_binarizer.pkl"
FEAT_COLS_PATH     = RESOURCES_DIR / "feature_columns.pkl"

RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH        = "hervoice_ml_training_data2.csv"           # updated default
TARGET_COL      = "safety_rating"
USE_GRID_SEARCH = False
FALLBACK_SCORE  = 3.0
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# Night hours definition (must match generate_data2.py)
NIGHT_HOURS = set(range(0, 5)) | {22, 23}   # 22:00–04:00


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _encode_tags(
    series: pd.Series,
    mlb: Optional[MultiLabelBinarizer] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    """
    Convert comma-separated tag strings into binary (0/1) columns.

    Parameters
    ----------
    series : pd.Series
        Raw tag strings, e.g. "dimly_lit,suspicious_activity".
    mlb    : MultiLabelBinarizer, optional
        Pre-fitted binarizer for inference-time use.
    fit    : bool
        If True, fit a fresh binarizer on *series*.

    Returns
    -------
    tag_df : pd.DataFrame  — one binary column per unique tag
    mlb    : MultiLabelBinarizer — (possibly freshly fitted)
    """
    cleaned = (
        series.fillna("")
              .str.lower()
              .str.strip()
    )
    tag_lists = cleaned.apply(
        lambda s: [t.strip() for t in s.split(",") if t.strip()]
    )

    if fit or mlb is None:
        mlb = MultiLabelBinarizer()
        tag_matrix = mlb.fit_transform(tag_lists)
    else:
        tag_matrix = mlb.transform(tag_lists)

    tag_df = pd.DataFrame(
        tag_matrix,
        columns=[f"tag_{c}" for c in mlb.classes_],
        index=series.index,
    )
    return tag_df, mlb


def _encode_grid(
    series: pd.Series,
    le: Optional[LabelEncoder] = None,
    fit: bool = True,
) -> tuple[pd.Series, LabelEncoder]:
    """
    Label-encode grid_cell_id. Unseen labels at inference time are mapped
    to the first known class instead of raising an error.
    """
    if fit or le is None:
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna("unknown"))
    else:
        known = set(le.classes_)
        safe  = series.astype(str).fillna("unknown").apply(
            lambda x: x if x in known else le.classes_[0]
        )
        encoded = le.transform(safe)

    return pd.Series(encoded, index=series.index, name="grid_encoded"), le


def _encode_zone(
    series: pd.Series,
    le: Optional[LabelEncoder] = None,
    fit: bool = True,
) -> tuple[pd.Series, LabelEncoder]:
    """
    Label-encode zone_label (e.g. "Saket", "Rohini").
    Unseen zones at inference time fall back to the first known class.
    """
    if fit or le is None:
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna("unknown"))
    else:
        known = set(le.classes_)
        safe  = series.astype(str).fillna("unknown").apply(
            lambda x: x if x in known else le.classes_[0]
        )
        encoded = le.transform(safe)

    return pd.Series(encoded, index=series.index, name="zone_encoded"), le


def _extract_temporal(series: pd.Series) -> pd.DataFrame:
    """
    Parse a datetime-like column and extract:
        day_of_week  (0=Monday … 6=Sunday)
        month        (1-12)
        hour         (0-23)
    Handles both "YYYY-MM-DD HH:MM:SS" (new CSV) and
    "DD-MM-YYYY HH:MM:SS" (legacy) formats automatically.
    """
    # Try ISO format first (new CSV: "2025-04-20 14:32:00")
    dt = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    # Fallback to legacy format ("09-05-2026 20:54:47")
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(series, format="%d-%m-%Y %H:%M:%S", errors="coerce")
    # Last resort: pandas inference
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")

    return pd.DataFrame(
        {
            "day_of_week": dt.dt.dayofweek.fillna(0).astype(int),
            "month":       dt.dt.month.fillna(1).astype(int),
            "hour":        dt.dt.hour.fillna(0).astype(int),
        },
        index=series.index,
    )


def _compute_is_night(hour_series: pd.Series) -> pd.Series:
    """
    Derive a binary is_night flag from an hour (0-23) series.
    Used at inference time when is_night is not directly provided.
    """
    return hour_series.apply(lambda h: int(h in NIGHT_HOURS)).rename("is_night")


def build_features(
    df: pd.DataFrame,
    mlb:     Optional[MultiLabelBinarizer] = None,
    le:      Optional[LabelEncoder]        = None,
    zone_le: Optional[LabelEncoder]        = None,
    fit:     bool = True,
) -> tuple[pd.DataFrame, MultiLabelBinarizer, LabelEncoder, LabelEncoder]:
    """
    Full feature engineering pipeline.

    New vs original:
      • zone_label  → zone_encoded  (label-encoded Delhi area name)
      • is_night    → binary 0/1, derived from time_context if not present
      • hour        → overridden by time_context when available (unchanged)

    Returns
    -------
    X        : pd.DataFrame with all engineered features
    mlb      : (fitted) MultiLabelBinarizer
    le       : (fitted) LabelEncoder  for grid_cell_id
    zone_le  : (fitted) LabelEncoder  for zone_label
    """
    df = df.copy()

    # ── 1. Spatial ─────────────────────────────────────────────────────────────
    spatial_df = df[["latitude", "longitude"]].copy()
    spatial_df["latitude"]  = pd.to_numeric(spatial_df["latitude"],  errors="coerce").fillna(0.0)
    spatial_df["longitude"] = pd.to_numeric(spatial_df["longitude"], errors="coerce").fillna(0.0)

    # Distance to nearest Delhi zone center
    ZONE_CENTERS = [
        (28.6315, 77.2167), (28.5677, 77.2437), (28.7041, 77.1025),
        (28.5921, 77.0460), (28.5245, 77.2066), (28.6519, 77.1909),
        (28.6562, 77.2310), (28.5450, 77.1577),
    ]
    def _min_zone_dist(lat: float, lng: float) -> float:
        return min(((lat - zl) ** 2 + (lng - zo) ** 2) ** 0.5 for zl, zo in ZONE_CENTERS)

    spatial_df["zone_dist"] = [
        _min_zone_dist(row.latitude, row.longitude)
        for row in spatial_df.itertuples(index=False)
    ]

    # ── 2. Tags ────────────────────────────────────────────────────────────────
    tag_df, mlb = _encode_tags(
        df.get("tags", pd.Series([""] * len(df), index=df.index)),
        mlb=mlb, fit=fit,
    )

    # ── 3. Grid Cell ───────────────────────────────────────────────────────────
    grid_col = df.get("grid_cell_id", pd.Series(["unknown"] * len(df), index=df.index))
    grid_encoded, le = _encode_grid(grid_col, le=le, fit=fit)

    # ── 4. Zone Label (NEW) ────────────────────────────────────────────────────
    zone_col = df.get("zone_label", pd.Series(["unknown"] * len(df), index=df.index))
    zone_encoded, zone_le = _encode_zone(zone_col, le=zone_le, fit=fit)

    # ── 5. Temporal from created_at ────────────────────────────────────────────
    time_col = df.get("created_at", pd.Series([pd.NaT] * len(df), index=df.index))
    temporal_df = _extract_temporal(time_col)

    # ── 6. time_context — override parsed hour when available ─────────────────
    if "time_context" in df.columns:
        tc = pd.to_numeric(df["time_context"], errors="coerce").fillna(0).astype(int)
        temporal_df["hour"]         = tc.values
        temporal_df["time_context"] = tc.values

    # ── 6b. Cyclic hour encoding — makes hour 23 and hour 0 numerically adjacent
    temporal_df["hour_sin"] = np.sin(2 * np.pi * temporal_df["hour"] / 24)
    temporal_df["hour_cos"] = np.cos(2 * np.pi * temporal_df["hour"] / 24)

    # ── 7. is_night flag (NEW) ─────────────────────────────────────────────────
    # Use the column directly if present (training CSV); derive it otherwise.
    if "is_night" in df.columns:
        is_night = pd.to_numeric(df["is_night"], errors="coerce").fillna(0).astype(int)
        is_night = is_night.rename("is_night")
    else:
        is_night = _compute_is_night(temporal_df["hour"])

    # ── Combine ────────────────────────────────────────────────────────────────
    X = pd.concat(
        [spatial_df, tag_df, grid_encoded, zone_encoded, temporal_df, is_night],
        axis=1,
    )
    return X, mlb, le, zone_le


# ══════════════════════════════════════════════════════════════════════════════
# Model Building
# ══════════════════════════════════════════════════════════════════════════════

def _build_base_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,            # was 6 — slightly shallower to reduce overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,   # was 0.8 — sample fewer columns per tree
        min_child_weight=5,     # was default 1 — forces broader, less noisy splits
        reg_alpha=0.1,          # L1 regularization (sparsity)
        reg_lambda=1.5,         # L2 regularization (was default 1.0)
        random_state=RANDOM_STATE,
        tree_method="hist",
        device="cuda",
        verbosity=0,
        n_jobs=-1,
    )


def _grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Run 5-fold GridSearchCV and return the best estimator."""
    logger.info("GridSearchCV: starting hyperparameter search (5-fold CV)…")
    param_grid = {
        "max_depth":        [4, 6, 8],
        "n_estimators":     [300, 500],
        "min_child_weight": [1, 3, 5],
        "learning_rate":    [0.01, 0.05, 0.1],
    }
    base = XGBRegressor(
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    logger.info("GridSearchCV best params: %s", gs.best_params_)
    logger.info("GridSearchCV best CV R²: %.4f", gs.best_score_)
    return gs.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(name: str, model: XGBRegressor, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Compute and log R², MAE, RMSE for a given split."""
    preds = model.predict(X)
    r2   = r2_score(y, preds)
    mae  = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    print(f"{name} R²   : {r2:.4f}")
    print(f"{name} MAE  : {mae:.4f}")
    print(f"{name} RMSE : {rmse:.4f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}


# ══════════════════════════════════════════════════════════════════════════════
# Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_importance(model: XGBRegressor, feature_names: list[str], top_n: int = 20) -> None:
    """Print top-N most important features ranked by gain."""
    importances = model.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n{'─'*40}")
    print(f"Top {top_n} Feature Importances")
    print(f"{'─'*40}")
    for i, row in feat_imp.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {i+1:>2}. {row['feature']:<35}  {row['importance']:.5f}  {bar}")
    print(f"{'─'*40}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(
    model:           XGBRegressor,
    le:              LabelEncoder,
    zone_le:         LabelEncoder,
    mlb:             MultiLabelBinarizer,
    feature_columns: list[str],
) -> None:
    """Persist all model artifacts to disk using joblib."""
    joblib.dump(model,           MODEL_PATH,     compress=3)
    joblib.dump(le,              LABEL_ENC_PATH, compress=3)
    joblib.dump(zone_le,         ZONE_ENC_PATH,  compress=3)
    joblib.dump(mlb,             TAG_BIN_PATH,   compress=3)
    joblib.dump(feature_columns, FEAT_COLS_PATH, compress=3)
    logger.info("Artifacts saved → %s", RESOURCES_DIR)


def load_artifacts() -> tuple[XGBRegressor, LabelEncoder, LabelEncoder, MultiLabelBinarizer, list[str]]:
    """Load all saved artifacts. Raises FileNotFoundError if any are missing."""
    for p in (MODEL_PATH, LABEL_ENC_PATH, ZONE_ENC_PATH, TAG_BIN_PATH, FEAT_COLS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}. Run training first.")

    model           = joblib.load(MODEL_PATH)
    le              = joblib.load(LABEL_ENC_PATH)
    zone_le         = joblib.load(ZONE_ENC_PATH)
    mlb             = joblib.load(TAG_BIN_PATH)
    feature_columns = joblib.load(FEAT_COLS_PATH)
    logger.info("Artifacts loaded from %s ✓", RESOURCES_DIR)
    return model, le, zone_le, mlb, feature_columns


# ══════════════════════════════════════════════════════════════════════════════
# Training Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def train(csv_path: str = CSV_PATH) -> dict:
    """
    Full training pipeline:
        1. Load CSV
        2. Validate & clean
        3. Engineer features (incl. zone_label + is_night)
        4. Train / (optionally) tune model
        5. Evaluate on train & test splits
        6. Print feature importance
        7. Save artifacts

    Returns a metrics dict.
    """
    logger.info("Loading dataset: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Columns found: %s", df.columns.tolist())

    # ── Normalise column names to internal schema ──────────────────────────────
    df = df.rename(columns={
        "lat": "latitude",
        "lng": "longitude",
    })

    # ── Validate target ────────────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    before = len(df)
    df.dropna(subset=[TARGET_COL], inplace=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with null target.", before - len(df))

    # ── Feature engineering ────────────────────────────────────────────────────
    logger.info("Engineering features…")
    X, mlb, le, zone_le = build_features(df, fit=True)
    y = df[TARGET_COL].values

    feature_columns = list(X.columns)
    logger.info("Feature matrix shape: %s  (columns: %d)", X.shape, len(feature_columns))

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Split: %d train / %d test", len(X_train), len(X_test))

    # ── Model selection ────────────────────────────────────────────────────────
    if USE_GRID_SEARCH:
        model = _grid_search(X_train, y_train)
    else:
        logger.info("Training XGBRegressor (USE_GRID_SEARCH=False)…")
        model = _build_base_model()
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n{'═'*45}")
    print("  Model Evaluation")
    print(f"{'═'*45}")
    train_metrics = _evaluate("TRAIN", model, X_train, y_train)
    print()
    test_metrics  = _evaluate("TEST",  model, X_test,  y_test)
    print(f"{'═'*45}\n")

    # ── Feature importance ────────────────────────────────────────────────────
    print_feature_importance(model, feature_columns, top_n=20)

    # ── Persist ───────────────────────────────────────────────────────────────
    save_artifacts(model, le, zone_le, mlb, feature_columns)

    return {
        "train":      train_metrics,
        "test":       test_metrics,
        "n_samples":  int(len(df)),
        "n_features": len(feature_columns),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Prediction Function
# ══════════════════════════════════════════════════════════════════════════════

def predict_safety(
    latitude:     float,
    longitude:    float,
    tags:         str,
    grid_cell_id: str,
    created_at:   str,
    time_context: int  = 0,
    zone_label:   str  = "unknown",
    is_night:     Optional[int] = None,   # NEW: pass explicitly or auto-derived
) -> float:
    """
    Predict safety score for a single location/time observation.

    Parameters
    ----------
    latitude     : float  — WGS-84 latitude
    longitude    : float  — WGS-84 longitude
    tags         : str    — comma-separated tag string, e.g. "dimly_lit,sparse_crowd"
    grid_cell_id : str    — spatial grid identifier
    created_at   : str    — datetime string, e.g. "2025-06-15 22:30:00"
    time_context : int    — hour of day (0-23) as recorded directly in the CSV
    zone_label   : str    — Delhi area name, e.g. "Saket" (NEW)
    is_night     : int    — 1 if night (22–04h), 0 otherwise; auto-derived if None (NEW)

    Returns
    -------
    float : predicted safety score clipped to [1.0, 5.0].
            Returns FALLBACK_SCORE if artifacts cannot be loaded.
    """
    try:
        model, le, zone_le, mlb, feature_columns = load_artifacts()
    except FileNotFoundError as exc:
        logger.warning("predict_safety fallback (%s); returning %.1f", exc, FALLBACK_SCORE)
        return FALLBACK_SCORE

    # Auto-derive is_night from time_context if not supplied
    if is_night is None:
        is_night = int(time_context in NIGHT_HOURS)

    row = pd.DataFrame([{
        "latitude":     latitude,
        "longitude":    longitude,
        "tags":         tags,
        "grid_cell_id": grid_cell_id,
        "created_at":   created_at,
        "time_context": time_context,
        "zone_label":   zone_label,
        "is_night":     is_night,
    }])

    X, _, _, _ = build_features(row, mlb=mlb, le=le, zone_le=zone_le, fit=False)
    X = X.reindex(columns=feature_columns, fill_value=0)

    raw   = float(model.predict(X)[0])
    score = round(float(np.clip(raw, 1.0, 5.0)), 3)
    logger.debug("predict_safety(%.4f, %.4f, %s) → %.3f", latitude, longitude, created_at, score)
    return score


# ══════════════════════════════════════════════════════════════════════════════
# SafetyPredictor class  (backward-compatible wrapper for main.py / routes.py)
# ══════════════════════════════════════════════════════════════════════════════

class SafetyPredictor:
    """
    Thin object-oriented wrapper around the module-level functions.
    Maintains the same public interface as the original service so that
    existing routes/controllers require no changes.

    New optional parameters (zone_label, is_night) are added with safe
    defaults so callers that don't supply them continue to work unchanged.
    """

    def __init__(self) -> None:
        self._model:    Optional[XGBRegressor]       = None
        self._le:       Optional[LabelEncoder]        = None
        self._zone_le:  Optional[LabelEncoder]        = None   # NEW
        self._mlb:      Optional[MultiLabelBinarizer] = None
        self._feat_cols: Optional[list[str]]          = None
        self._is_ready: bool = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load(self) -> bool:
        """Attempt to load pre-trained artifacts from disk."""
        try:
            self._model, self._le, self._zone_le, self._mlb, self._feat_cols = load_artifacts()
            self._is_ready = True
            return True
        except FileNotFoundError:
            logger.info("No pre-trained artifacts found; call .train() first.")
            self._is_ready = False
            return False

    def train(self, csv_path: str = CSV_PATH) -> dict:
        """Train the model and cache artifacts in memory."""
        metrics = train(csv_path)
        self._model, self._le, self._zone_le, self._mlb, self._feat_cols = load_artifacts()
        self._is_ready = True
        return metrics

    def predict(
        self,
        latitude:     float,
        longitude:    float,
        tags:         str  = "",
        grid_cell_id: str  = "unknown",
        created_at:   str  = "2024-01-01 00:00:00",
        time_context: int  = 0,
        zone_label:   str  = "unknown",          # NEW
        is_night:     Optional[int] = None,      # NEW
    ) -> float:
        """Single-point prediction with fallback."""
        if not self._is_ready:
            return FALLBACK_SCORE

        if is_night is None:
            is_night = int(time_context in NIGHT_HOURS)

        row = pd.DataFrame([{
            "latitude":     latitude,
            "longitude":    longitude,
            "tags":         tags,
            "grid_cell_id": grid_cell_id,
            "created_at":   created_at,
            "time_context": time_context,
            "zone_label":   zone_label,
            "is_night":     is_night,
        }])
        X, _, _, _ = build_features(row, mlb=self._mlb, le=self._le, zone_le=self._zone_le, fit=False)
        X = X.reindex(columns=self._feat_cols, fill_value=0)
        raw = float(self._model.predict(X)[0])
        return round(float(np.clip(raw, 1.0, 5.0)), 3)

    def predict_batch(self, records: list[dict]) -> list[float]:
        """
        Batch inference.

        Each record must be a dict with keys:
            latitude, longitude, tags, grid_cell_id, created_at
        Optional keys (new): zone_label, is_night, time_context
        """
        if not self._is_ready:
            return [FALLBACK_SCORE] * len(records)

        df_batch = pd.DataFrame(records)

        # Derive is_night for any rows missing it
        if "is_night" not in df_batch.columns:
            tc = pd.to_numeric(df_batch.get("time_context", 0), errors="coerce").fillna(0).astype(int)
            df_batch["is_night"] = tc.apply(lambda h: int(h in NIGHT_HOURS))

        X, _, _, _ = build_features(df_batch, mlb=self._mlb, le=self._le, zone_le=self._zone_le, fit=False)
        X          = X.reindex(columns=self._feat_cols, fill_value=0)
        raws       = self._model.predict(X)
        return [round(float(np.clip(v, 1.0, 5.0)), 3) for v in raws]


# ── Module-level singleton (imported by main.py and routes.py) ─────────────────
safety_predictor = SafetyPredictor()


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    logger.info("Starting HerVoice SafetyPredictor training pipeline…")
    logger.info("Dataset    : %s", csv_file)
    logger.info("Grid search: %s", USE_GRID_SEARCH)

    results = train(csv_file)

    print("\n── Final Metrics Summary ──────────────────────────────")
    print(f"  Samples   : {results['n_samples']}")
    print(f"  Features  : {results['n_features']}")
    print(f"  Train R²  : {results['train']['r2']:.4f}")
    print(f"  Test  R²  : {results['test']['r2']:.4f}")
    print(f"  Train MAE : {results['train']['mae']:.4f}")
    print(f"  Test  MAE : {results['test']['mae']:.4f}")
    print(f"  Train RMSE: {results['train']['rmse']:.4f}")
    print(f"  Test  RMSE: {results['test']['rmse']:.4f}")
    print("───────────────────────────────────────────────────────\n")

    # ── Smoke-test the predict_safety function ─────────────────────────────────
    logger.info("Running prediction smoke-test…")
    sample_score = predict_safety(
        latitude=28.6315,
        longitude=77.2167,
        tags="dimly_lit,sparse_crowd",
        grid_cell_id="GRID_001",
        created_at="2025-06-15 22:30:00",
        time_context=22,
        zone_label="Connaught Place",
        is_night=1,
    )
    print(f"  Sample prediction (night) → safety_score: {sample_score}")

    day_score = predict_safety(
        latitude=28.5245,
        longitude=77.2066,
        tags="well_lit,crowded,cctv_present",
        grid_cell_id="GRID_002",
        created_at="2025-06-15 14:00:00",
        time_context=14,
        zone_label="Saket",
        is_night=0,
    )
    print(f"  Sample prediction (day)   → safety_score: {day_score}")
