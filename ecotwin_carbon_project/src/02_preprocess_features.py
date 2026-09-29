"""
STEP 2 - CLEANING + FEATURE ENGINEERING
---------------------------------------
Purpose: Turn the raw CSV into a modelling-ready dataset.

What we do and WHY:
  1. Parse timestamp (day-first format) -> hour, day, month, weekday, is_weekend.
     Pollution follows a daily rhythm (rush hours), so time features matter.
  2. Cyclic encoding (sin/cos) for hour and wind direction, because 23h is next
     to 0h and 359 degrees is next to 0 degrees - plain numbers would mislead the model.
  3. Domain features:
       total_emission_kg_h  = traffic + industrial
       emission_per_wind    = total emission / wind speed  (poor wind = trapped air)
       ventilation_index    = wind_speed * ventilation_control (how much air is moved)
       heat_humidity_index  = temperature * humidity / 100
  4. Lag + rolling features per zone (previous-hour CO2, 3h rolling mean) because
     air quality is strongly auto-correlated.
  5. One-hot encode zone_id.

Outputs:
  data/processed/clean_dataset.csv        (cleaned + time parsed, human readable)
  data/processed/model_ready_dataset.csv  (fully engineered, used for training)
  outputs/reports/02_feature_dictionary.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ecotwin_urban_carbon_dispersal_dataset.csv"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
for p in (PROC, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

# ---------- 1. cleaning ----------
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
df = df.drop_duplicates().sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

# physical sanity clipping (no negative rates / impossible humidity)
df["wind_speed_ms"] = df["wind_speed_ms"].clip(lower=0)
df["humidity_pct"] = df["humidity_pct"].clip(0, 100)
for c in ["traffic_emission_rate_kg_h", "industrial_emission_rate_kg_h", "pm25_ug_m3"]:
    df[c] = df[c].clip(lower=0)

df.to_csv(PROC / "clean_dataset.csv", index=False)

# ---------- 2. time features ----------
ts = df["timestamp"]
df["hour"] = ts.dt.hour
df["day"] = ts.dt.day
df["month"] = ts.dt.month
df["weekday"] = ts.dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction_deg"]))
df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction_deg"]))

# ---------- 3. domain features ----------
df["total_emission_kg_h"] = df["traffic_emission_rate_kg_h"] + df["industrial_emission_rate_kg_h"]
df["emission_per_wind"] = df["total_emission_kg_h"] / (df["wind_speed_ms"] + 0.5)
df["ventilation_index"] = df["wind_speed_ms"] * df["rl_ventilation_control_mw"]
df["heat_humidity_index"] = df["temperature_c"] * df["humidity_pct"] / 100
df["traffic_share"] = df["traffic_emission_rate_kg_h"] / df["total_emission_kg_h"]

# ---------- 4. lag / rolling per zone ----------
g = df.groupby("zone_id")
df["co2_lag1"] = g["co2_concentration_ppm"].shift(1)
df["pm25_lag1"] = g["pm25_ug_m3"].shift(1)
df["co2_roll3"] = g["co2_concentration_ppm"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df["emission_roll3"] = g["total_emission_kg_h"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
# first record of each zone has no history -> fill with that zone's mean
for c in ["co2_lag1", "pm25_lag1", "co2_roll3", "emission_roll3"]:
    df[c] = df[c].fillna(df.groupby("zone_id")[c].transform("mean"))

# ---------- 5. encode zone ----------
model_df = pd.get_dummies(df, columns=["zone_id"], prefix="zone", drop_first=False)
model_df = model_df.drop(columns=["latitude", "longitude"])  # zone dummies already capture location
model_df.to_csv(PROC / "model_ready_dataset.csv", index=False)

# ---------- feature dictionary ----------
doc = {
    "timestamp": "Hourly observation time (parsed, day-first)",
    "hour/day/month/weekday/is_weekend": "Calendar parts extracted from timestamp",
    "hour_sin/hour_cos": "Cyclic encoding of hour (23h and 0h are neighbours)",
    "wind_dir_sin/wind_dir_cos": "Cyclic encoding of wind direction in degrees",
    "wind_speed_ms": "Wind speed - main natural dispersal driver",
    "temperature_c/humidity_pct": "Weather conditions affecting dispersal",
    "traffic_emission_rate_kg_h": "CO2 emitted by traffic in the zone",
    "industrial_emission_rate_kg_h": "CO2 emitted by industry in the zone",
    "total_emission_kg_h": "traffic + industrial emission",
    "emission_per_wind": "Emission load divided by wind speed (trapping indicator)",
    "ventilation_index": "wind_speed * RL ventilation power (air actually moved)",
    "heat_humidity_index": "temperature * humidity / 100",
    "traffic_share": "Fraction of emission coming from traffic",
    "rl_traffic_signal_offset_s": "RL action: signal timing offset in seconds",
    "rl_green_corridor_active": "RL action: green corridor on (1) / off (0)",
    "rl_ventilation_control_mw": "RL action: ventilation power in MW",
    "co2_lag1/pm25_lag1": "Previous hour value in the same zone",
    "co2_roll3/emission_roll3": "3-hour rolling mean of past values (same zone)",
    "zone_XXX": "One-hot flag for each of the 10 city zones",
    "co2_concentration_ppm": "TARGET 1 - CO2 concentration",
    "pm25_ug_m3": "TARGET 2 - PM2.5 concentration",
    "reward_value": "TARGET 3 - RL controller reward score",
}
with open(REPORTS / "02_feature_dictionary.txt", "w") as f:
    f.write("FEATURE DICTIONARY\n" + "=" * 60 + "\n")
    for k, v in doc.items():
        f.write(f"{k:35s} : {v}\n")
    f.write(f"\nFinal model-ready shape: {model_df.shape[0]} rows x {model_df.shape[1]} columns\n")

print("Step 2 done.", model_df.shape)
