import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 🔹 Pipeline 모델 로딩
with open('../models/logistic_regression/logistic_model_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
threshold = bundle['threshold']  # 예: 0.4

# 🔹 사용자 입력 받을 컬럼 목록
input_columns = [
    'book_dow', 'book_tod', 'book_category', 'book_staff',
    'last_category', 'last_staff', 'last_dow', 'last_tod',
    'last_day_services', 'last_noshow',
    'last_cumrev', 'last_cumbook', 'last_cumstyle',
    'last_cumcolor', 'last_cumnoshow',
    'recency', 'first_visit', 'is_revisit_30days'
]

# 🔹 Streamlit UI
st.title("📈 로지스틱 회귀 기반 노쇼 예측기")

st.subheader("📋 예약/고객 정보 입력")

# 예시 값들 (학습 시 사용한 값 기준)
input_dict = {
    'book_dow': st.selectbox("예약 요일", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
    'book_tod': st.selectbox("예약 시간대", ['morning', 'afternoon', 'evening', 'Unknown']),
    'book_category': st.selectbox("서비스 유형", ['STYLE', 'COLOR', 'MISC']),
    'book_staff': st.selectbox("담당자", ['JJ', 'JOANNE', 'KELLY', 'BECKY', 'HOUSE', 'SINEAD', 'TANYA']),
    'last_category': st.selectbox("이전 서비스", ['Unknown', 'STYLE', 'COLOR', 'MISC']),
    'last_staff': st.selectbox("이전 담당자", ['Unknown', 'JJ', 'JOANNE', 'KELLY', 'BECKY', 'HOUSE', 'SINEAD', 'TANYA']),
    'last_dow': st.selectbox("이전 요일", ['Unknown', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
    'last_tod': st.selectbox("이전 시간대", ['Unknown', 'morning', 'afternoon', 'evening']),
    'last_day_services': st.number_input("이전 시술 수", min_value=0),
    'last_noshow': st.selectbox("이전 노쇼 여부", [0, 1]),
    'last_cumrev': st.number_input("누적 매출", min_value=0),
    'last_cumbook': st.number_input("누적 예약 수", min_value=0),
    'last_cumstyle': st.number_input("누적 스타일 시술 수", min_value=0),
    'last_cumcolor': st.number_input("누적 컬러 시술 수", min_value=0),
    'last_cumnoshow': st.number_input("누적 노쇼 수", min_value=0),
    'recency': st.number_input("최근 방문 후 지난 일수", min_value=0),
    'first_visit': st.selectbox("첫 방문 여부", [0, 1]),
    'is_revisit_30days': st.selectbox("30일 이내 재방문 여부", [0, 1])
}

# 🔹 DataFrame 변환
input_df = pd.DataFrame([input_dict])

# 🔹 예측
if st.button("노쇼 예측하기"):
    y_proba = model.predict_proba(input_df)[:, 1][0]
    y_pred = int(y_proba >= threshold)

    st.subheader("📊 예측 결과")
    st.write(f"예상 노쇼 확률: **{y_proba:.2%}**")

    if y_pred:
        st.error("❌ 노쇼 가능성이 높습니다!")
    else:
        st.success("✅ 노쇼 가능성은 낮습니다.")
