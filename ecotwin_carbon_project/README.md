# EcoTwin Urban Carbon Dispersal - End-to-End Data Science Project

A complete, reproducible data science project built on
`ecotwin_urban_carbon_dispersal_dataset.csv` (1,500 hourly records, 10 city zones).

Goal: predict urban air-quality outcomes (CO2, PM2.5) and the RL controller's
reward from weather, emission and control-action data - and evaluate honestly
whether it is possible with this data.

---

## 1. Folder structure

```
ecotwin_carbon_project/
├── README.md                     <- start here (this file)
├── PROJECT_EXPLANATION.md        <- step-by-step walkthrough for the team
├── FINDINGS_AND_LIMITATIONS.md   <- the honest result + what to do next
├── requirements.txt              <- python packages
├── run_all.py                    <- runs the whole pipeline in order
├── data/
│   ├── raw/        ecotwin_urban_carbon_dispersal_dataset.csv   (untouched input)
│   ├── processed/  clean_dataset.csv, model_ready_dataset.csv
│   └── final/      final_dataset_with_predictions.csv  <- MAIN DELIVERABLE
│                   zone_risk_summary.csv
├── src/
│   ├── 01_data_understanding.py       profiling + data quality
│   ├── 02_preprocess_features.py      cleaning + feature engineering
│   ├── 03_eda_visuals.py              EDA charts and tables
│   ├── 04_train_models.py             5 models x 3 targets + evaluation
│   └── 05_scenario_and_final_output.py predictions, what-if scenario, risk levels
├── models/          saved .joblib models + feature_columns.json
└── outputs/
    ├── figures/     10 PNG charts (slide-ready)
    ├── tables/      all numeric results as CSV
    └── reports/     text reports for each stage
```

## 2. How to run

```bash
pip install -r requirements.txt
python run_all.py
```

Everything regenerates from the raw CSV - no manual steps, no hidden state.

## 3. Pipeline at a glance

| Step | Script | What it produces |
|---|---|---|
| 1 | `01_data_understanding.py` | data profile, summary stats, zone summary |
| 2 | `02_preprocess_features.py` | cleaned data + 25 engineered features, feature dictionary |
| 3 | `03_eda_visuals.py` | 6 EDA charts, correlation table, hourly profile |
| 4 | `04_train_models.py` | Baseline / Linear / Ridge / RandomForest / GradientBoosting compared on 3 targets |
| 5 | `05_scenario_and_final_output.py` | per-row predictions, "max mitigation" what-if, risk labels, business insights |

## 4. Headline result (read the details in FINDINGS_AND_LIMITATIONS.md)

On a strict chronological 80/20 split, **no model beats the mean baseline**:

| Target | Best model | Test R² | Test MAE |
|---|---|---|---|
| co2_concentration_ppm | Baseline (mean) | -0.002 | 88.2 ppm |
| pm25_ug_m3 | Baseline (mean) | -0.011 | 26.97 µg/m³ |
| reward_value | Baseline (mean) | -0.000 | 38.97 |

All feature-to-target correlations sit between -0.06 and +0.06, i.e. the targets
in this file are statistically independent of the inputs. The dataset looks
synthetically generated with independent random columns. The **pipeline is
correct and production-shaped**; the *data* carries no learnable signal yet.
This is a valid and important finding to present, not a failure of the code.

## 5. Main deliverable file

`data/final/final_dataset_with_predictions.csv` - every original row plus:
engineered features, `pred_*` predictions, `residual_*` errors,
`scenario_pred_co2`, `scenario_co2_saving_ppm` and a `co2_risk_level`
label (Low / Moderate / High / Severe).
