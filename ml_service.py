"""
HerVoice - Spatio-Temporal Predictive Safety Engine
app/services/ml_service.py

SafetyPredictor encapsulates the full ML lifecycle:
  - Feature engineering (spatial, tags, grid cell, temporal)
  - XGBoost Regressor training with optional GridSearchCV
  - joblib serialization / deserialization
  - Single-point inference with safe fallback
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
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RESOURCES_DIR      = Path(__file__).resolve().parent / "resources"
MODEL_PATH         = RESOURCES_DIR / "trained_model.pkl"
LABEL_ENC_PATH     = RESOURCES_DIR / "label_encoder.pkl"
TAG_BIN_PATH       = RESOURCES_DIR / "tag_binarizer.pkl"
FEAT_COLS_PATH     = RESOURCES_DIR / "feature_columns.pkl"
SPATIAL_REF_PATH   = RESOURCES_DIR / "spatial_reference.pkl"

RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH        = "hervoice_ml_training_data_12.csv"
TARGET_COL      = "safety_rating"
USE_GRID_SEARCH = False          # Set True to enable hyperparameter tuning
FALLBACK_SCORE  = 3.0            # Returned when model is not ready
TEST_SIZE       = 0.20
RANDOM_STATE    = 42


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
    # Normalise: fill nulls, strip whitespace, lowercase
    cleaned = (
        series.fillna("")
              .str.lower()
              .str.strip()
    )
    # Split each row into a list; empty string → empty list
    tag_lists = cleaned.apply(
        lambda s: [t.strip() for t in s.split(",") if t.strip()]
    )

    if fit or mlb is None:
        mlb = MultiLabelBinarizer()
        tag_matrix = mlb.fit_transform(tag_lists)
    else:
        # Unseen tags at inference time are silently ignored
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
    Label-encode grid_cell_id.  Unseen labels at inference time are mapped
    to -1 (unknown class) instead of raising an error.
    """
    if fit or le is None:
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna("unknown"))
    else:
        known = set(le.classes_)
        safe  = series.astype(str).fillna("unknown").apply(
            lambda x: x if x in known else le.classes_[0]   # fallback to first class
        )
        encoded = le.transform(safe)

    return pd.Series(encoded, index=series.index, name="grid_encoded"), le


def _extract_temporal(series: pd.Series) -> pd.DataFrame:
    """
    Parse a datetime-like column and extract:
        day_of_week  (0=Monday … 6=Sunday)
        month        (1-12)
        hour         (0-23)
    """
    # Format matches HerVoice CSV: DD-MM-YYYY HH:MM:SS
    # e.g. "09-05-2026 20:54:47"
    dt = pd.to_datetime(series, format="%d-%m-%Y %H:%M:%S", errors="coerce")
    # Fallback: if most values are NaT (wrong format), try default inference
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(series, errors="coerce")
    return pd.DataFrame(
        {
            "day_of_week": dt.dt.dayofweek.fillna(0).astype(int),
            "month":       dt.dt.month.fillna(1).astype(int),
            "hour":        dt.dt.hour.fillna(0).astype(int),
        },
        index=series.index,
    )


def _compute_knn_features(
    coords: np.ndarray,
    fit: bool,
    spatial_ref: Optional[dict] = None,
    current_ratings: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute KNN-style spatial features:
      - mean_neighbor_safety (average safety rating of the 5 nearest neighbors)
      - dist_to_nearest_incident (distance to the closest rating <= 2.0)
    """
    if fit:
        ref_coords = coords
        ref_ratings = current_ratings if current_ratings is not None else np.full(len(coords), 3.0)
        incident_mask = ref_ratings <= 2.0
        incident_coords = ref_coords[incident_mask]
    else:
        if spatial_ref is None:
            return np.full(len(coords), 3.0), np.full(len(coords), 10.0)
        ref_coords = spatial_ref["coords"]
        ref_ratings = spatial_ref["ratings"]
        incident_coords = spatial_ref["incident_coords"]

    # ── 1. Mean Neighbor Safety (k=5 neighbors) ──
    tree = KDTree(ref_coords)
    if fit:
        k_query = min(6, len(ref_coords))
        dists, indices = tree.query(coords, k=k_query)
        if indices.ndim == 1:
            indices = indices[:, np.newaxis]
        # Skip the first column (self representation)
        neighbor_indices = indices[:, 1:] if indices.shape[1] > 1 else indices
        mean_safety = np.mean(ref_ratings[neighbor_indices], axis=1)
    else:
        k_query = min(5, len(ref_coords))
        dists, indices = tree.query(coords, k=k_query)
        if indices.ndim == 1:
            indices = indices[:, np.newaxis]
        mean_safety = np.mean(ref_ratings[indices], axis=1)

    # ── 2. Distance to Nearest Incident ──
    if len(incident_coords) == 0:
        dist_to_incident = np.full(len(coords), 10.0)
    else:
        inc_tree = KDTree(incident_coords)
        if fit:
            dists, indices = inc_tree.query(coords, k=min(2, len(incident_coords)))
            if dists.ndim == 1:
                dists = dists[:, np.newaxis]
            dist_to_incident = []
            for i in range(len(coords)):
                # If the point itself is an incident, distance is ~0, so take the 2nd closest
                if dists[i, 0] < 1e-7 and dists.shape[1] > 1:
                    dist_to_incident.append(dists[i, 1])
                else:
                    dist_to_incident.append(dists[i, 0])
            dist_to_incident = np.array(dist_to_incident)
        else:
            dists, indices = inc_tree.query(coords, k=1)
            dist_to_incident = dists

    # Clean up NaNs / infs
    mean_safety = np.nan_to_num(mean_safety, nan=3.0)
    dist_to_incident = np.nan_to_num(dist_to_incident, nan=10.0)

    return mean_safety, dist_to_incident


def build_features(
    df: pd.DataFrame,
    mlb: Optional[MultiLabelBinarizer] = None,
    le:  Optional[LabelEncoder]        = None,
    fit: bool = True,
    spatial_ref: Optional[dict] = None,
) -> tuple[pd.DataFrame, MultiLabelBinarizer, LabelEncoder]:
    """
    Full feature engineering pipeline with KNN-style spatial features.

    Returns
    -------
    X   : pd.DataFrame with all engineered features
    mlb : (fitted) MultiLabelBinarizer
    le  : (fitted) LabelEncoder
    """
    df = df.copy()

    # ── 1. Spatial ─────────────────────────────────────────────────────────────
    spatial_df = df[["latitude", "longitude"]].copy()
    spatial_df["latitude"]  = pd.to_numeric(spatial_df["latitude"],  errors="coerce").fillna(0.0)
    spatial_df["longitude"] = pd.to_numeric(spatial_df["longitude"], errors="coerce").fillna(0.0)

    # ── KNN-style Spatial Neighborhood Features ──
    coords = spatial_df[["latitude", "longitude"]].values
    current_ratings = df[TARGET_COL].values if (fit and TARGET_COL in df.columns) else None
    
    mean_safety, dist_to_incident = _compute_knn_features(
        coords, fit=fit, spatial_ref=spatial_ref, current_ratings=current_ratings
    )
    spatial_df["mean_neighbor_safety"] = mean_safety
    spatial_df["dist_to_nearest_incident"] = dist_to_incident

    # ── 2. Tags ────────────────────────────────────────────────────────────────
    tag_df, mlb = _encode_tags(df.get("tags", pd.Series([""] * len(df), index=df.index)), mlb=mlb, fit=fit)

    # ── 3. Grid Cell ───────────────────────────────────────────────────────────
    grid_col = df.get("grid_cell_id", pd.Series(["unknown"] * len(df), index=df.index))
    grid_encoded, le = _encode_grid(grid_col, le=le, fit=fit)

    # ── 4. Temporal from created_at ─────────────────────────────────────────────────────────────
    time_col = df.get("created_at", pd.Series([pd.NaT] * len(df), index=df.index))
    temporal_df = _extract_temporal(time_col)

    # ── 5. time_context (raw hour 0-23, present in HerVoice CSV) ───────────────────
    if "time_context" in df.columns:
        tc = pd.to_numeric(df["time_context"], errors="coerce").fillna(0).astype(int)
        temporal_df["hour"] = tc.values
        temporal_df["time_context"] = tc.values

    # ── Combine ────────────────────────────────────────────────────────────────────────────
    X = pd.concat([spatial_df, tag_df, grid_encoded, temporal_df], axis=1)
    return X, mlb, le


# ══════════════════════════════════════════════════════════════════════════════
# Model Building
# ══════════════════════════════════════════════════════════════════════════════

def _build_base_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=10,
        reg_alpha=1.0,
        reg_lambda=3.0,
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
    model: XGBRegressor,
    le: LabelEncoder,
    mlb: MultiLabelBinarizer,
    feature_columns: list[str],
    spatial_ref: dict,
) -> None:
    """Persist all model artifacts to disk using joblib."""
    joblib.dump(model,          MODEL_PATH,       compress=3)
    joblib.dump(le,             LABEL_ENC_PATH,   compress=3)
    joblib.dump(mlb,            TAG_BIN_PATH,     compress=3)
    joblib.dump(feature_columns, FEAT_COLS_PATH,   compress=3)
    joblib.dump(spatial_ref,     SPATIAL_REF_PATH, compress=3)
    logger.info("Artifacts saved → %s", RESOURCES_DIR)


def load_artifacts() -> tuple[XGBRegressor, LabelEncoder, MultiLabelBinarizer, list[str], dict]:
    """Load all saved artifacts. Raises FileNotFoundError if any are missing."""
    for p in (MODEL_PATH, LABEL_ENC_PATH, TAG_BIN_PATH, FEAT_COLS_PATH, SPATIAL_REF_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}. Run training first.")

    model           = joblib.load(MODEL_PATH)
    le              = joblib.load(LABEL_ENC_PATH)
    mlb             = joblib.load(TAG_BIN_PATH)
    feature_columns = joblib.load(FEAT_COLS_PATH)
    spatial_ref     = joblib.load(SPATIAL_REF_PATH)
    logger.info("Artifacts loaded from %s ✓", RESOURCES_DIR)
    return model, le, mlb, feature_columns, spatial_ref


# ══════════════════════════════════════════════════════════════════════════════
# Training Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def train(csv_path: str = CSV_PATH) -> dict:
    """
    Full training pipeline:
        1. Load CSV
        2. Validate & clean
        3. Engineer features
        4. Train / (optionally) tune model
        5. Evaluate on train & test splits
        6. Print feature importance
        7. Save artifacts

    Returns a metrics dict.
    """
    if isinstance(csv_path, pd.DataFrame):
        logger.info("Training on passed DataFrame: %d rows", len(csv_path))
        df = csv_path.copy()
    else:
        logger.info("Loading dataset: %s", csv_path)
        df = pd.read_csv(csv_path)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Columns found: %s", df.columns.tolist())

    # ── Normalise column names to internal schema ──────────────────────────────
    # Handles CSVs that use 'lat'/'lng' instead of 'latitude'/'longitude'
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

    # Construct spatial reference dictionary for inference
    spatial_ref = {
        "coords": df[["latitude", "longitude"]].values,
        "ratings": df[TARGET_COL].values,
        "incident_coords": df.loc[df[TARGET_COL] <= 2.0, ["latitude", "longitude"]].values
    }

    # ── Feature engineering ────────────────────────────────────────────────────
    logger.info("Engineering features…")
    X, mlb, le = build_features(df, fit=True)
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
    save_artifacts(model, le, mlb, feature_columns, spatial_ref)

    return {
        "train": train_metrics,
        "test":  test_metrics,
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
    time_context: int = 0,
) -> float:
    """
    Predict safety score for a single location/time observation.

    Parameters
    ----------
    latitude     : float  — WGS-84 latitude
    longitude    : float  — WGS-84 longitude
    tags         : str    — comma-separated tag string, e.g. "dimly_lit,sparse_crowd"
    grid_cell_id : str    — spatial grid identifier
    created_at   : str    — ISO datetime string, e.g. "2024-06-15 22:30:00"
    time_context : int    — hour of day (0-23) as recorded directly in the CSV

    Returns
    -------
    float : predicted safety score, clipped to [1.0, 5.0].
            Returns FALLBACK_SCORE if artifacts cannot be loaded.
    """
    try:
        model, le, mlb, feature_columns, spatial_ref = load_artifacts()
    except FileNotFoundError as exc:
        logger.warning("predict_safety fallback (%s); returning %.1f", exc, FALLBACK_SCORE)
        return FALLBACK_SCORE

    # Build a single-row DataFrame mirroring the training schema
    row = pd.DataFrame([{
        "latitude":     latitude,
        "longitude":    longitude,
        "tags":         tags,
        "grid_cell_id": grid_cell_id,
        "created_at":   created_at,
        "time_context": time_context,
    }])

    # Apply identical feature engineering (fit=False → use saved encoders)
    X, _, _ = build_features(row, mlb=mlb, le=le, fit=False, spatial_ref=spatial_ref)

    # Align columns to the training schema (fill unseen cols with 0)
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
    """

    def __init__(self) -> None:
        self._model:   Optional[XGBRegressor]        = None
        self._le:      Optional[LabelEncoder]         = None
        self._mlb:     Optional[MultiLabelBinarizer]  = None
        self._feat_cols: Optional[list[str]]          = None
        self._spatial_ref: Optional[dict]             = None
        self._is_ready: bool = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load(self) -> bool:
        """Attempt to load pre-trained artifacts from disk."""
        try:
            self._model, self._le, self._mlb, self._feat_cols, self._spatial_ref = load_artifacts()
            self._is_ready = True
            return True
        except FileNotFoundError:
            logger.info("No pre-trained artifacts found; call .train() first.")
            self._is_ready = False
            return False

    def train(self, csv_path: str = CSV_PATH) -> dict:
        """Train the model and cache artifacts in memory."""
        metrics = train(csv_path)
        # Reload freshly saved artifacts into instance state
        self._model, self._le, self._mlb, self._feat_cols, self._spatial_ref = load_artifacts()
        self._is_ready = True
        return metrics

    def predict(
        self,
        latitude:     float,
        longitude:    float,
        tags:         str = "",
        grid_cell_id: str = "unknown",
        created_at:   str = "2024-01-01 00:00:00",
        time_context: int = 0,
    ) -> float:
        """Single-point prediction with fallback."""
        if not self._is_ready:
            return FALLBACK_SCORE
        row = pd.DataFrame([{
            "latitude":     latitude,
            "longitude":    longitude,
            "tags":         tags,
            "grid_cell_id": grid_cell_id,
            "created_at":   created_at,
            "time_context": time_context,
        }])
        X, _, _ = build_features(row, mlb=self._mlb, le=self._le, fit=False, spatial_ref=self._spatial_ref)
        X = X.reindex(columns=self._feat_cols, fill_value=0)
        raw = float(self._model.predict(X)[0])
        return round(float(np.clip(raw, 1.0, 5.0)), 3)

    def predict_batch(self, records: list[dict]) -> list[float]:
        """
        Batch inference.

        Each record must be a dict with keys:
            latitude, longitude, tags, grid_cell_id, created_at
        """
        if not self._is_ready:
            return [FALLBACK_SCORE] * len(records)

        df_batch = pd.DataFrame(records)
        X, _, _  = build_features(df_batch, mlb=self._mlb, le=self._le, fit=False, spatial_ref=self._spatial_ref)
        X        = X.reindex(columns=self._feat_cols, fill_value=0)
        raws     = self._model.predict(X)
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
    logger.info("Dataset  : %s", csv_file)
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
        latitude=28.6139,
        longitude=77.2090,
        tags="dimly_lit,sparse_crowd",
        grid_cell_id="GRID_001",
        created_at="2024-06-15 22:30:00",
        time_context=22,
    )
    print(f"  Sample prediction → safety_score: {sample_score}")
