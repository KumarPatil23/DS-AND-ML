"""Run the whole pipeline end-to-end: python run_all.py"""
import subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
steps = [
    "01_data_understanding.py",
    "02_preprocess_features.py",
    "03_eda_visuals.py",
    "04_train_models.py",
    "05_scenario_and_final_output.py",
]
for s in steps:
    print("\n" + "=" * 70 + f"\nRUNNING {s}\n" + "=" * 70)
    subprocess.run([sys.executable, str(SRC / s)], check=True)
print("\nPipeline finished. Check outputs/ and data/final/.")
