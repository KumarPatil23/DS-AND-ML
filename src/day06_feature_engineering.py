import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA, FEATURE_DATA, REPORTS
from utils import add_time_features
import pandas as pd

df=pd.read_csv(PROCESSED_DATA,parse_dates=['timestamp'])
out=add_time_features(df)
out.to_csv(FEATURE_DATA,index=False)
summary=out[['hour','day','month','dayofweek','is_weekend','traffic_level','wind_category']].head(10).to_string(index=False)
(REPORTS/'day06_feature_preview.txt').write_text('First 10 engineered rows:\n\n'+summary,encoding='utf-8')
print('Created:',FEATURE_DATA); print(summary)
