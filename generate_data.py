"""
HerVoice - Synthetic Training Data Generator
Generates realistic crowd-sourced safety rating data mirroring the production schema.
Usage: python generate_data.py
Output: hervoice_ml_training_data.csv
"""

import uuid
import random
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.utils.geohash import encode_hash

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_RATINGS      = 20_000
NUM_DEVICES      = 500
OUTPUT_FILE      = "hervoice_ml_training_data.csv"
DAYS_HISTORY     = 90       # ratings spread over last 90 days

# ── Simulated safety zones (lat, lng, zone_label, base_safety) ─────────────────
ZONES = [
    {"lat": 28.6139, "lng": 77.2090, "label": "Central Delhi",    "base_safety": 3.2, "radius_deg": 0.04},
    {"lat": 28.5355, "lng": 77.3910, "label": "Noida Sector 18",  "base_safety": 3.8, "radius_deg": 0.03},
    {"lat": 28.4595, "lng": 77.0266, "label": "Gurgaon Cyber Hub","base_safety": 4.1, "radius_deg": 0.03},
]

# ── Common safety tags per rating level ────────────────────────────────────────
TAGS_BY_LEVEL = {
    1: ["dark_street", "no_people", "felt_followed", "isolated", "poor_lighting"],
    2: ["sparse_crowd", "dimly_lit", "suspicious_activity", "no_cctv"],
    3: ["moderate_crowd", "some_lighting", "mixed_vibes", "ok_transport"],
    4: ["well_lit", "crowded", "cctv_present", "police_nearby", "busy_market"],
    5: ["very_safe", "well_lit", "heavy_crowd", "security_present", "good_transport"],
}

# ── Temporal modifier: suppress safety at night ────────────────────────────────
def temporal_safety_modifier(hour: int) -> float:
    """
    Returns a float additive modifier for safety_rating based on hour.
    Night (22:00 - 04:00): -1.2 to -0.6
    Early morning (05:00 - 07:00): -0.3 to 0.0
    Day (08:00 - 18:00): +0.2 to +0.5
    Evening (19:00 - 21:00): -0.1 to +0.1
    """
    if 22 <= hour <= 23 or 0 <= hour <= 4:
        return random.uniform(-1.2, -0.6)   # Night penalty
    elif 5 <= hour <= 7:
        return random.uniform(-0.3, 0.0)    # Pre-dawn slight penalty
    elif 8 <= hour <= 18:
        return random.uniform(0.2, 0.5)     # Daytime bonus
    else:                                    # 19-21 evening
        return random.uniform(-0.1, 0.1)

# ── Spatial noise around a zone center ────────────────────────────────────────
def sample_coords(zone: dict) -> tuple[float, float]:
    """Gaussian scatter around zone center, bounded by radius_deg."""
    sigma = zone["radius_deg"] / 2.5
    lat = np.random.normal(zone["lat"], sigma)
    lng = np.random.normal(zone["lng"], sigma)
    return round(lat, 6), round(lng, 6)

# ── Weighted zone selection (safer zones get more ratings) ────────────────────
zone_weights = [z["base_safety"] for z in ZONES]
total_w      = sum(zone_weights)
zone_probs   = [w / total_w for w in zone_weights]

# ── Device pool ───────────────────────────────────────────────────────────────
device_pool = [str(uuid.uuid4()) for _ in range(NUM_DEVICES)]

# ── Generate rows ─────────────────────────────────────────────────────────────
rows = []
now  = datetime.utcnow()

for i in range(NUM_RATINGS):
    # Pick zone
    zone = random.choices(ZONES, weights=zone_probs, k=1)[0]

    # Coordinates
    lat, lng = sample_coords(zone)

    # Timestamp: random point in last DAYS_HISTORY days
    days_ago   = random.uniform(0, DAYS_HISTORY)
    created_at = now - timedelta(days=days_ago)
    hour       = created_at.hour

    # Safety rating: zone base + temporal modifier, clipped to [1, 5]
    raw_score      = zone["base_safety"] + temporal_safety_modifier(hour) + random.uniform(-0.5, 0.5)
    safety_rating  = int(max(1, min(5, round(raw_score))))

    # Tags: 1–3 tags drawn from appropriate level, ±1 level for realism
    tag_level = max(1, min(5, safety_rating + random.choice([-1, 0, 0, 1])))
    num_tags  = random.randint(1, 3)
    tags      = random.sample(TAGS_BY_LEVEL[tag_level], min(num_tags, len(TAGS_BY_LEVEL[tag_level])))

    # Geohash cell id
    grid_cell_id = encode_hash(lat, lng, 7)

    rows.append({
        "device_uuid":   random.choice(device_pool),
        "lat":           lat,
        "lng":           lng,
        "grid_cell_id":  grid_cell_id,
        "safety_rating": safety_rating,
        "time_context":  str(hour),          # mirrors production column
        "created_at":    created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "tags":          ",".join(tags),
    })

# ── Persist ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Generated {len(df):,} ratings across {df['grid_cell_id'].nunique()} unique grid cells")
print(f"   Devices      : {df['device_uuid'].nunique()}")
print(f"   Rating dist  : {df['safety_rating'].value_counts().sort_index().to_dict()}")
print(f"   Saved to     : {OUTPUT_FILE}")
