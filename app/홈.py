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
tab1, tab2 = st.tabs(["📊 통계 분석", "💡 전략 제안"])

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
