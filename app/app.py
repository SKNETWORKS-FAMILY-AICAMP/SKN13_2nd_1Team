'''
streamlit_app.py
'''

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.title("💇‍♀️ 미용실 노쇼 분석")

# 🗂 탭 정의
tab1, tab2 = st.tabs(["📊 통계 분석", "💡 모델 분석"])

# =============================================
with tab1:
    @st.cache_data
    def load_data(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.dropna(subset=['book_tod', 'book_dow'])
        return df

    hair_salon_data = load_data('dataset/raw/hair_salon_no_show_wrangled_df.csv')

    # 라디오 버튼으로 뷰 선택
    view = st.radio("Select view:", ("Time of Day", "Day of Week", "By Staff"))

    if view == "Time of Day":
        st.header("⏰ No-Show Rate by Time of Day")
        # 시간대별 집계
        stats_tod = (
            hair_salon_data
            .groupby('book_tod')['noshow']
            .agg(total_appointments='count', no_show_rate='mean')
            .reset_index()
        )
        order_tod = ['morning', 'afternoon', 'evening', 'night']
        stats_tod['book_tod'] = pd.Categorical(stats_tod['book_tod'], categories=order_tod, ordered=True)
        stats_tod = stats_tod.sort_values('book_tod')

        # 차트
        fig, ax = plt.subplots()
        ax.bar(stats_tod['book_tod'], stats_tod['no_show_rate'])
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('No-Show Rate')
        ax.set_ylim(0, stats_tod['no_show_rate'].max() * 1.1)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 테이블
        st.subheader("시간대별 통계 데이터")
        st.dataframe(stats_tod.style.format({
            "no_show_rate": "{:.2%}",
            "total_appointments": "{:,}"
        }))

    elif view == "Day of Week":
        st.header("📅 No-Show Rate by Day of Week")
        # 요일별 집계
        stats_dow = (
            hair_salon_data
            .groupby('book_dow')['noshow']
            .agg(total_appointments='count', no_show_rate='mean')
            .reset_index()
        )
        order_dow = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        stats_dow['book_dow'] = pd.Categorical(stats_dow['book_dow'], categories=order_dow, ordered=True)
        stats_dow = stats_dow.sort_values('book_dow')

        # 차트
        fig, ax = plt.subplots()
        ax.bar(stats_dow['book_dow'], stats_dow['no_show_rate'])
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('No-Show Rate')
        ax.set_ylim(0, stats_dow['no_show_rate'].max() * 1.1)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 테이블
        st.subheader("요일별 통계 데이터")
        st.dataframe(stats_dow.style.format({
            "no_show_rate": "{:.2%}",
            "total_appointments": "{:,}"
        }))

    elif view == "By Staff":
        st.header("🧑‍💼 No-Show Rate by Staff")
        st.markdown("각 스태프(`book_staff`)별 예약 건수와 노쇼율을 보여줍니다.")

        stats_staff = (
        hair_salon_data
        .groupby('book_staff')['noshow']
        .agg(total_appointments='count', no_show_rate='mean')
        .reset_index()
        .sort_values('no_show_rate', ascending=False)
    )
        st.dataframe(stats_staff.style.format({
            "no_show_rate": "{:.2%}",
            "total_appointments": "{:,}"
        }))
        fig, ax = plt.subplots()
        ax.bar(stats_staff['book_staff'], stats_staff['no_show_rate'])
        ax.set_xlabel('Staff')
        ax.set_ylabel('No-Show Rate')
        ax.set_ylim(0, stats_staff['no_show_rate'].max() * 1.1)
        plt.xticks(rotation=45)
        st.pyplot(fig)
# =============================================

with tab2:
    st.title("💡 모델 분석")
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
