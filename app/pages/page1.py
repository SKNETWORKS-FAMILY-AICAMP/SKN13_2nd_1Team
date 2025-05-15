import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 모델 불러오기
with open('../models/xgboost/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# LabelEncoder 사전 (학습 시 저장한 파일이 있다고 가정)
with open('../models/xgboost/label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 입력 받을 범주형 컬럼
categorical_cols = ['book_tod', 'book_dow', 'book_category', 'book_staff',
                    'last_category', 'last_staff', 'last_dow', 'last_tod']

numeric_cols = ['last_day_services', 'last_noshow',
                'last_cumrev', 'last_cumbook', 'last_cumstyle', 'last_cumcolor',
                'last_cumnoshow', 'recency',
                'first_visit', 'is_revisit_30days']

st.title("💇‍♀️ XGBoost 노쇼 예측기")

input_dict = {}

# Streamlit 입력 받기 (범주형)
st.header("📋 예약 정보 입력")
for col in categorical_cols:
    values = encoders[col].classes_
    input_dict[col] = st.selectbox(f"{col}", values)

# Streamlit 입력 받기 (수치형)
st.header("🔢 고객 수치 데이터 입력")
for col in numeric_cols:
    input_dict[col] = st.number_input(f"{col}", min_value=0)

# DataFrame 생성
input_df = pd.DataFrame([input_dict])

# 인코딩 적용
for col in categorical_cols:
    le = encoders[col]
    input_df[col] = le.transform(input_df[col])

# 예측
if st.button("노쇼 예측하기"):
    y_proba = model.predict_proba(input_df)[:, 1][0]
    y_pred = int(y_proba >= 0.4)  # threshold는 0.4 예시

    st.subheader("📊 예측 결과")
    st.write(f"노쇼 확률: **{y_proba:.2%}**")

    if y_pred:
        st.error("❌ 노쇼 가능성이 높습니다!")
    else:
        st.success("✅ 노쇼 가능성은 낮습니다.")
