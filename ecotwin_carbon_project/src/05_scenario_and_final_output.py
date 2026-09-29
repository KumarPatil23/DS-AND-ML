"""
STEP 5 - PREDICTIONS, RL CONTROL SCENARIO TEST, FINAL DELIVERABLE DATASET
-------------------------------------------------------------------------
1. Score the trained models on the whole dataset -> predictions + residuals.
2. Business/what-if scenario: keep everything the same, but switch the RL
   controls to a "max mitigation" setting (green corridor ON, ventilation at
   max, signal offset +30s) and see the predicted CO2 reduction.
3. Risk labelling: classify each hour into Low/Moderate/High/Severe CO2 risk
   so operations teams can act.

Outputs:
  data/final/final_dataset_with_predictions.csv   <- MAIN DELIVERABLE
  data/final/zone_risk_summary.csv
  outputs/tables/05_scenario_impact.csv
  outputs/figures/10_scenario_impact.png
  outputs/reports/05_business_insights.txt
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "model_ready_dataset.csv", parse_dates=["timestamp"])
meta = json.load(open(ROOT / "models" / "feature_columns.json"))
FEATURES, TARGETS = meta["features"], meta["targets"]
FINAL = ROOT / "data" / "final"; FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"; REP = ROOT / "outputs" / "reports"
for p in (FINAL, FIG, TAB, REP):
    p.mkdir(parents=True, exist_ok=True)

models = {t: joblib.load(ROOT / "models" / f"{t}_best_model.joblib") for t in TARGETS}
X = df[FEATURES].astype(float)

out = df.copy()
for t in TARGETS:
    out[f"pred_{t}"] = models[t].predict(X)
    out[f"residual_{t}"] = out[t] - out[f"pred_{t}"]

# ---- scenario: maximum RL mitigation ----
Xs = X.copy()
Xs["rl_green_corridor_active"] = 1
Xs["rl_ventilation_control_mw"] = df["rl_ventilation_control_mw"].max()
Xs["rl_traffic_signal_offset_s"] = 30
Xs["ventilation_index"] = Xs["wind_speed_ms"] * Xs["rl_ventilation_control_mw"]
out["scenario_pred_co2"] = models["co2_concentration_ppm"].predict(Xs)
out["scenario_co2_saving_ppm"] = out["pred_co2_concentration_ppm"] - out["scenario_pred_co2"]

# ---- risk label ----
q = out["co2_concentration_ppm"].quantile([0.25, 0.5, 0.75]).values
out["co2_risk_level"] = pd.cut(out["co2_concentration_ppm"],
                               bins=[-np.inf, q[0], q[1], q[2], np.inf],
                               labels=["Low", "Moderate", "High", "Severe"])

zone_cols = [c for c in out.columns if c.startswith("zone_")]
out["zone_id"] = out[zone_cols].idxmax(axis=1).str.replace("zone_", "", regex=False)
out.to_csv(FINAL / "final_dataset_with_predictions.csv", index=False)

risk = (out.groupby("zone_id")
          .agg(mean_co2=("co2_concentration_ppm", "mean"),
               mean_pred_co2=("pred_co2_concentration_ppm", "mean"),
               mean_scenario_co2=("scenario_pred_co2", "mean"),
               mean_saving_ppm=("scenario_co2_saving_ppm", "mean"),
               severe_hours=("co2_risk_level", lambda s: (s == "Severe").sum()),
               mean_reward=("reward_value", "mean"))
          .round(3).sort_values("mean_co2", ascending=False))
risk.to_csv(FINAL / "zone_risk_summary.csv")

scen = pd.DataFrame({
    "metric": ["mean actual CO2 ppm", "mean predicted CO2 ppm",
               "mean CO2 under max-mitigation scenario", "mean saving ppm", "mean saving %"],
    "value": [round(out["co2_concentration_ppm"].mean(), 3),
              round(out["pred_co2_concentration_ppm"].mean(), 3),
              round(out["scenario_pred_co2"].mean(), 3),
              round(out["scenario_co2_saving_ppm"].mean(), 3),
              round(100 * out["scenario_co2_saving_ppm"].mean() / out["pred_co2_concentration_ppm"].mean(), 3)],
})
scen.to_csv(TAB / "05_scenario_impact.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4.5))
idx = np.arange(len(risk))
ax.bar(idx - 0.2, risk.mean_pred_co2, 0.4, label="current control")
ax.bar(idx + 0.2, risk.mean_scenario_co2, 0.4, label="max mitigation scenario")
ax.set_xticks(idx, risk.index, rotation=45)
ax.set_ylabel("CO2 ppm"); ax.legend(); ax.set_title("Predicted CO2 per zone: current vs max mitigation")
fig.tight_layout(); fig.savefig(FIG / "10_scenario_impact.png"); plt.close(fig)

corr = df.corr(numeric_only=True)["co2_concentration_ppm"].drop("co2_concentration_ppm")
with open(REP / "05_business_insights.txt", "w") as f:
    f.write("BUSINESS INSIGHTS - ECOTWIN URBAN CARBON DISPERSAL\n" + "=" * 70 + "\n\n")
    f.write("1) SCENARIO RESULT (max RL mitigation on every hour)\n")
    f.write(scen.to_string(index=False) + "\n\n")
    f.write("2) ZONE PRIORITY (highest mean CO2 first)\n")
    f.write(risk.to_string() + "\n\n")
    f.write("3) STRONGEST LINEAR RELATIONSHIPS WITH CO2\n")
    f.write(corr.abs().sort_values(ascending=False).head(10).round(4).to_string() + "\n\n")
    f.write("4) RECOMMENDED ACTIONS\n")
    f.write("   - Focus mitigation budget on the top-3 zones in the table above.\n")
    f.write("   - Pre-activate green corridors before the peak hours seen in 03_hourly_pattern.png.\n")
    f.write("   - On low-wind hours (emission_per_wind high), raise ventilation power early.\n")
    f.write("   - Use the saved models to forecast next-hour CO2 and trigger alerts on 'Severe' risk.\n")

print(scen.to_string(index=False))
print(risk)
print("Step 5 done. Final dataset written.")
