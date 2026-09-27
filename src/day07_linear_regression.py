from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sklearn.linear_model import LinearRegression
from model_common import evaluate
from config import MODELS, REPORTS
import joblib

df,pipe,pred,yte,m=evaluate('Linear Regression',LinearRegression())
joblib.dump(pipe,MODELS/'linear_regression.joblib')
REPORTS.joinpath('day07_metrics.txt').write_text(str(m),encoding='utf-8')
print(m)
