import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. 데이터 불러오기
df = pd.read_csv('../dataset/processed/hair_salon_data.csv')

# 2. 범주형 컬럼 정의
categorical_cols = ['book_tod', 'book_dow', 'book_category', 'book_staff',
                    'last_category', 'last_staff', 'last_dow', 'last_tod']

# 3. 각 컬럼별 LabelEncoder 적용 및 저장
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str)) 
    encoders[col] = le

# 4. 딕셔너리 형태로 저장
with open('../models/xgboost/label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
