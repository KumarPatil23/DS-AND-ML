"""
STEP 1 - DATA UNDERSTANDING
---------------------------
Purpose: Load the raw EcoTwin urban carbon dispersal dataset and produce a
complete data-quality / profiling report so the team knows exactly what we have
before any modelling.

Outputs:
  outputs/reports/01_data_profile.txt
  outputs/tables/01_summary_statistics.csv
  outputs/tables/01_zone_summary.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ecotwin_urban_carbon_dispersal_dataset.csv"
REPORTS = ROOT / "outputs" / "reports"
TABLES = ROOT / "outputs" / "tables"
for p in (REPORTS, TABLES):
    p.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

lines = []
add = lines.append
add("ECOTWIN URBAN CARBON DISPERSAL - DATA PROFILE")
add("=" * 60)
add(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
add("")
add("COLUMN TYPES")
add(df.dtypes.to_string())
add("")
add("MISSING VALUES PER COLUMN")
add(df.isna().sum().to_string())
add("")
add("DUPLICATE ROWS: " + str(int(df.duplicated().sum())))
add("")
add("UNIQUE ZONES: " + ", ".join(sorted(df["zone_id"].unique())))
add("")
add("TIME RANGE (raw strings): " + f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
add("")
add("SUMMARY STATISTICS (numeric)")
add(df.describe().T.round(3).to_string())
add("")
add("TARGET CANDIDATES")
add("  co2_concentration_ppm : main air-quality target (regression)")
add("  pm25_ug_m3            : secondary pollution target (regression)")
add("  reward_value          : RL controller performance score (regression)")
add("")
add("ACTION / CONTROL COLUMNS (decided by the RL agent)")
add("  rl_traffic_signal_offset_s, rl_green_corridor_active, rl_ventilation_control_mw")

(REPORTS / "01_data_profile.txt").write_text("\n".join(lines))
df.describe().T.round(4).to_csv(TABLES / "01_summary_statistics.csv")

zone = df.groupby("zone_id").agg(
    records=("zone_id", "size"),
    mean_co2_ppm=("co2_concentration_ppm", "mean"),
    mean_pm25=("pm25_ug_m3", "mean"),
    mean_traffic_emission=("traffic_emission_rate_kg_h", "mean"),
    mean_industrial_emission=("industrial_emission_rate_kg_h", "mean"),
    mean_reward=("reward_value", "mean"),
).round(3).sort_values("mean_co2_ppm", ascending=False)
zone.to_csv(TABLES / "01_zone_summary.csv")

print("Step 1 done. Profile + summary tables written.")
print(zone)
