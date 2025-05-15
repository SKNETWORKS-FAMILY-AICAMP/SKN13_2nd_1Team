import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 📦 모델 로드
with open("../models/logistic_regression/logistic_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle['model']
threshold = bundle['threshold']

# 📊 데이터 불러오기
df = pd.read_csv('../dataset/processed/hair_salon_data.csv')

# 🗂 탭 정의
tab1, tab2, tab3 = st.tabs(["📊 통계 분석", "💡 전략 제안", "🔮 노쇼 예측기"])

# =============================================
with tab1:
    st.title("📊 통계 분석")

    st.metric("전체 노쇼율", f"{df['noshow'].mean():.1%}")

    st.subheader("요일별 노쇼율")
    dow_rate = df.groupby('book_dow')['noshow'].mean().sort_values()
    st.bar_chart(dow_rate)

    st.subheader("시간대별 노쇼율")
    tod_rate = df.groupby('book_tod')['noshow'].mean().sort_values()
    st.bar_chart(tod_rate)

    st.subheader("누적 노쇼 횟수 vs 실제 노쇼율")
    df['noshow_bin'] = pd.cut(df['last_cumnoshow'], bins=[-1, 0, 1, 3, 10], labels=["0", "1", "2-3", "4+"])
    st.bar_chart(df.groupby('noshow_bin')['noshow'].mean())

# =============================================
with tab2:
    st.title("💡 전략 제안")

    st.markdown("""
    ### 📍 데이터 기반 인사이트
    - `화요일 오전`의 노쇼율이 가장 높음
    - `STYLE` 서비스 예약 고객의 노쇼율이 높음
    - `누적 노쇼 ≥ 2` 고객은 전체 평균의 **2.5배 이상** 노쇼함

    ### ✅ 추천 전략
    | 전략 | 설명 |
    |------|------|
    | 🔔 **리마인더 발송** | 노쇼 확률 50% 이상 고객에게 예약 하루 전 자동 알림 |
    | 💰 **예약금 제도** | 노쇼 누적 2회 이상 고객에 대해 사전 결제 도입 검토 |
    | ⛔ **예약 제한** | 최근 30일 내 3회 이상 노쇼한 고객은 온라인 예약 제한 |
    """)

# =============================================
with tab3:
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

        y_proba = model.predict_proba(input_df)[:, 1][0]
        y_pred = int(y_proba >= threshold)

        st.subheader("예측 결과")
        st.write(f"🔢 노쇼 확률: **{y_proba:.2%}**")
        if y_pred:
            st.error("❌ 노쇼 가능성이 높습니다!")
        else:
            st.success("✅ 노쇼 가능성은 낮습니다.")
