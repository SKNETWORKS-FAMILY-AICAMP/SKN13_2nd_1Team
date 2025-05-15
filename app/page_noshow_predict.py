import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("../models/xgboost/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

threshold = 0.4

st.title("🔮 노쇼 예측기")

st.subheader("📋 고객 정보 입력")

book_dow = st.selectbox("예약 요일", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
book_tod = st.selectbox("예약 시간대", ['afternoon', 'morning', 'evening', 'Unknown'])
book_category = st.selectbox("예약 서비스 유형", ['STYLE', 'COLOR', 'MISC'])
book_staff = st.selectbox("예약 담당자", ['JJ', 'JOANNE', 'KELLY', 'BECKY', 'HOUSE', 'SINEAD', 'TANYA'])

last_dow = st.selectbox("이전 방문 요일", ['Unknown', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
last_tod = st.selectbox("이전 방문 시간대", ['Unknown', 'morning', 'afternoon', 'evening'])
last_category = st.selectbox("이전 서비스 유형", ['Unknown', 'COLOR', 'STYLE', 'MISC'])
last_staff = st.selectbox("이전 담당자", ['Unknown', 'JJ', 'JOANNE', 'KELLY', 'BECKY', 'HOUSE', 'SINEAD', 'TANYA'])

last_day_services = st.number_input("이전 방문 시 시술 수", min_value=0)
last_receipt_tot = st.number_input("이전 결제 금액", min_value=0.0)
last_cumbook = st.number_input("누적 예약 횟수", min_value=0)
last_cumstyle = st.number_input("누적 STYLE 횟수", min_value=0)
last_cumcolor = st.number_input("누적 COLOR 횟수", min_value=0)
last_cumprod = st.number_input("누적 제품 구매 횟수", min_value=0)
last_cumcancel = st.number_input("누적 취소 횟수", min_value=0)
last_cumnoshow = st.number_input("누적 노쇼 횟수", min_value=0)
last_noshow = st.selectbox("이전 방문 노쇼 여부", [0, 1])
last_prod_flag = st.selectbox("이전 방문 시 제품 구매 여부", [0, 1])
last_cumrev = st.number_input("누적 결제 금액", min_value=0)
recency = st.number_input("최근 방문 후 지난 일 수", min_value=0)
ID = st.text_input("고객 ID", value="TEMP_ID")
is_revisit_30days = st.selectbox("30일 이내 재방문 여부", [0, 1])
first_visit = st.selectbox("첫 방문 여부", [0, 1])

if st.button("노쇼 예측하기"):
    input_df = pd.DataFrame([{
        'book_dow': book_dow,
        'book_tod': book_tod,
        'book_category': book_category,
        'book_staff': book_staff,
        'last_dow': last_dow,
        'last_tod': last_tod,
        'last_category': last_category,
        'last_staff': last_staff,
        'last_day_services': last_day_services,
        'last_receipt_tot': last_receipt_tot,
        'last_cumbook': last_cumbook,
        'last_cumstyle': last_cumstyle,
        'last_cumcolor': last_cumcolor,
        'last_cumprod': last_cumprod,
        'last_cumcancel': last_cumcancel,
        'last_cumnoshow': last_cumnoshow,
        'last_noshow': last_noshow,
        'last_prod_flag': last_prod_flag,
        'last_cumrev': last_cumrev,
        'recency': recency,
        'ID': ID,
        'is_revisit_30days': is_revisit_30days,
        'first_visit': first_visit
    }])

    y_proba = model.probs(input_df)[:, 1][0]
    y_pred = int(y_proba >= threshold)

    st.subheader("예측 결과")
    st.write(f"🔢 노쇼 확률: **{y_proba:.2%}**")
    if y_pred:
        st.error("❌ 노쇼 가능성이 높습니다!")
    else:
        st.success("✅ 노쇼 가능성은 낮습니다.")
