"""
STEP 3 - EXPLORATORY DATA ANALYSIS (EDA) + CHARTS
-------------------------------------------------
Purpose: Understand patterns and relationships, and save every chart as PNG so
the team can use them directly in slides.

Outputs (outputs/figures/):
  01_target_distributions.png
  02_correlation_heatmap.png
  03_hourly_pattern.png
  04_zone_comparison.png
  05_wind_vs_co2.png
  06_green_corridor_effect.png
Outputs (outputs/tables/):
  03_correlation_with_targets.csv
  03_hourly_profile.csv
  03_green_corridor_effect.csv
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "model_ready_dataset.csv", parse_dates=["timestamp"])
clean = pd.read_csv(ROOT / "data" / "processed" / "clean_dataset.csv", parse_dates=["timestamp"])
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
TARGETS = ["co2_concentration_ppm", "pm25_ug_m3", "reward_value"]

# 1. target distributions
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, t in zip(ax, TARGETS):
    a.hist(df[t], bins=30, color="#2e7d32", edgecolor="white")
    a.set_title(t)
fig.suptitle("Distribution of the three targets")
fig.tight_layout()
fig.savefig(FIG / "01_target_distributions.png")
plt.close(fig)

# 2. correlation heatmap
num = df.select_dtypes("number")
drivers = [c for c in num.columns if not c.startswith("zone_")]
corr = num[drivers].corr()
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(drivers)), drivers, rotation=90, fontsize=7)
ax.set_yticks(range(len(drivers)), drivers, fontsize=7)
fig.colorbar(im, shrink=0.8)
ax.set_title("Correlation heatmap (numeric drivers)")
fig.tight_layout()
fig.savefig(FIG / "02_correlation_heatmap.png")
plt.close(fig)
corr[TARGETS].round(4).to_csv(TAB / "03_correlation_with_targets.csv")

# 3. hourly pattern
hourly = clean.assign(hour=clean.timestamp.dt.hour).groupby("hour")[TARGETS + ["traffic_emission_rate_kg_h"]].mean()
hourly.round(3).to_csv(TAB / "03_hourly_profile.csv")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(hourly.index, hourly.co2_concentration_ppm, marker="o", label="CO2 ppm")
ax.plot(hourly.index, hourly.pm25_ug_m3, marker="s", label="PM2.5 ug/m3")
ax.plot(hourly.index, hourly.reward_value, marker="^", label="RL reward")
ax.set_xlabel("Hour of day"); ax.legend(); ax.set_title("Average value by hour of day")
fig.tight_layout(); fig.savefig(FIG / "03_hourly_pattern.png"); plt.close(fig)

# 4. zone comparison
zone = clean.groupby("zone_id")[TARGETS].mean().sort_values("co2_concentration_ppm")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(zone.index, zone.co2_concentration_ppm, color="#00695c")
ax.set_ylabel("Mean CO2 ppm"); ax.set_title("Mean CO2 concentration by zone")
plt.xticks(rotation=45)
fig.tight_layout(); fig.savefig(FIG / "04_zone_comparison.png"); plt.close(fig)

# 5. wind vs co2
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(clean.wind_speed_ms, clean.co2_concentration_ppm,
                c=clean.total_emission_kg_h if "total_emission_kg_h" in clean else
                  clean.traffic_emission_rate_kg_h + clean.industrial_emission_rate_kg_h,
                cmap="viridis", s=14, alpha=0.75)
fig.colorbar(sc, label="total emission kg/h")
ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("CO2 ppm")
ax.set_title("Higher wind disperses CO2")
fig.tight_layout(); fig.savefig(FIG / "05_wind_vs_co2.png"); plt.close(fig)

# 6. RL green corridor effect
eff = clean.groupby("rl_green_corridor_active")[TARGETS].mean().round(3)
eff.to_csv(TAB / "03_green_corridor_effect.csv")
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, t in zip(ax, TARGETS):
    a.bar(["OFF", "ON"], eff[t].values, color=["#b71c1c", "#1b5e20"])
    a.set_title(t)
fig.suptitle("Green corridor OFF vs ON (mean values)")
fig.tight_layout(); fig.savefig(FIG / "06_green_corridor_effect.png"); plt.close(fig)

print("Step 3 done. Figures + EDA tables saved.")
print(eff)
