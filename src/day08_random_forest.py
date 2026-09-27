from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sklearn.ensemble import RandomForestRegressor
from model_common import evaluate
from config import MODELS, REPORTS
import joblib

model=RandomForestRegressor(n_estimators=250,random_state=42,n_jobs=-1)
df,pipe,pred,yte,m=evaluate('Random Forest',model)
joblib.dump(pipe,MODELS/'random_forest.joblib')
REPORTS.joinpath('day08_metrics.txt').write_text(str(m),encoding='utf-8')
print(m)
