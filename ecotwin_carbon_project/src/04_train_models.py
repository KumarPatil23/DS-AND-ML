"""
STEP 4 - MODEL TRAINING + EVALUATION
------------------------------------
Problem framing: supervised regression. For each target
(co2_concentration_ppm, pm25_ug_m3, reward_value) we predict the value from
weather, emissions, RL control actions, zone and time features.

Models compared (simple -> strong):
  Baseline_Mean         - always predicts the training mean (the bar to beat)
  LinearRegression      - transparent baseline
  Ridge                 - regularised linear
  RandomForestRegressor - non-linear, robust
  GradientBoostingRegressor - usually the strongest on tabular data

Validation: chronological 80/20 split (we must NOT train on the future) plus
5-fold KFold cross-validation on the training part for stability.
Metrics: MAE, RMSE, R2.

Outputs:
  models/<target>_best_model.joblib , models/feature_columns.json
  outputs/tables/04_model_comparison.csv
  outputs/tables/04_feature_importance_<target>.csv
  outputs/figures/07_model_comparison_r2.png
  outputs/figures/08_actual_vs_predicted.png
  outputs/figures/09_feature_importance_co2.png
  outputs/reports/04_model_report.txt
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "model_ready_dataset.csv", parse_dates=["timestamp"])
MODELS = ROOT / "models"; FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"; REP = ROOT / "outputs" / "reports"
for p in (MODELS, FIG, TAB, REP):
    p.mkdir(parents=True, exist_ok=True)

TARGETS = ["co2_concentration_ppm", "pm25_ug_m3", "reward_value"]
df = df.sort_values("timestamp").reset_index(drop=True)
FEATURES = [c for c in df.columns if c not in TARGETS + ["timestamp"]]
X = df[FEATURES].astype(float)
split = int(len(df) * 0.8)

def build_models():
    return {
        "Baseline_Mean": DummyRegressor(strategy="mean"),
        "LinearRegression": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                              random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                      max_depth=3, random_state=42),
    }

rows, best_per_target, preds = [], {}, {}
for target in TARGETS:
    y = df[target]
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    best = (None, None, -np.inf)
    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        cv = cross_val_score(model, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
        rows.append({
            "target": target, "model": name,
            "test_MAE": round(mean_absolute_error(yte, p), 4),
            "test_RMSE": round(float(np.sqrt(mean_squared_error(yte, p))), 4),
            "test_R2": round(r2_score(yte, p), 4),
            "cv_R2_mean": round(cv.mean(), 4), "cv_R2_std": round(cv.std(), 4),
        })
        if r2_score(yte, p) > best[2]:
            best = (name, model, r2_score(yte, p)); preds[target] = (yte.values, p)
    best_per_target[target] = best[0]
    # Always also save a RandomForest importance table for interpretability,
    # even if the mean baseline happened to score best.
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).round(6).to_csv(
        TAB / f"04_rf_feature_importance_{target}.csv", header=["importance"])
    joblib.dump(best[1], MODELS / f"{target}_best_model.joblib")

    # feature importance / coefficients of the winning model
    m = best[1]
    inner = m.named_steps["m"] if hasattr(m, "named_steps") else m
    if hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=FEATURES)
    elif hasattr(inner, "coef_"):
        imp = pd.Series(np.abs(np.ravel(inner.coef_)), index=FEATURES)
    else:  # baseline model has no notion of feature importance
        imp = pd.Series(0.0, index=FEATURES)
    imp.sort_values(ascending=False).round(6).to_csv(TAB / f"04_feature_importance_{target}.csv",
                                                    header=["importance"])

comp = pd.DataFrame(rows)
comp.to_csv(TAB / "04_model_comparison.csv", index=False)
json.dump({"features": FEATURES, "targets": TARGETS, "best_models": best_per_target},
          open(MODELS / "feature_columns.json", "w"), indent=2)

# chart: R2 per model per target
fig, ax = plt.subplots(figsize=(10, 4.5))
pivot = comp.pivot(index="model", columns="target", values="test_R2")
pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("Test R2"); ax.set_title("Model comparison (higher is better)")
plt.xticks(rotation=20); fig.tight_layout()
fig.savefig(FIG / "07_model_comparison_r2.png"); plt.close(fig)

# chart: actual vs predicted for each target (best model)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
for a, t in zip(ax, TARGETS):
    yt, yp = preds[t]
    a.scatter(yt, yp, s=14, alpha=0.7, color="#1565c0")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    a.plot([lo, hi], [lo, hi], "r--")
    a.set_xlabel("actual"); a.set_ylabel("predicted")
    a.set_title(f"{t}\nbest: {best_per_target[t]}")
fig.tight_layout(); fig.savefig(FIG / "08_actual_vs_predicted.png"); plt.close(fig)

# chart: top drivers of CO2
imp = pd.read_csv(TAB / "04_rf_feature_importance_co2_concentration_ppm.csv", index_col=0).head(15)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp.index[::-1], imp["importance"][::-1], color="#00838f")
ax.set_title("Top 15 drivers of CO2 concentration")
fig.tight_layout(); fig.savefig(FIG / "09_feature_importance_co2.png"); plt.close(fig)

with open(REP / "04_model_report.txt", "w") as f:
    f.write("MODEL TRAINING REPORT\n" + "=" * 70 + "\n")
    f.write(f"Rows: {len(df)}  Features used: {len(FEATURES)}\n")
    f.write(f"Chronological split -> train {split} rows / test {len(df)-split} rows\n\n")
    f.write(comp.to_string(index=False) + "\n\n")
    f.write("BEST MODEL PER TARGET\n")
    for t, m in best_per_target.items():
        r = comp[(comp.target == t) & (comp.model == m)].iloc[0]
        f.write(f"  {t:25s} -> {m:18s} R2={r.test_R2}  MAE={r.test_MAE}  RMSE={r.test_RMSE}\n")

print(comp.to_string(index=False))
print("Best:", best_per_target)
