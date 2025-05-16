'''
streamlit_app.py
'''

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 페이지 설정
# st.set_page_config(
#     page_title="💇‍♀️ 미용실 노쇼 대시보드",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# 🗂 탭 정의
tab1, tab2 = st.tabs(["📊 통계 분석", "💡 모델 분석"])

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=['book_tod', 'book_dow'])
    return df

hair_salon_data = load_data('dataset/raw/hair_salon_no_show_wrangled_df.csv')

def get_stats(data, group_col):
    stats = (
        data
        .groupby(group_col)['noshow']
        .agg(total_appointments='count', no_show_rate='mean')
        .reset_index()
    )
    

# =============================================
with tab1:

    # 라디오 버튼으로 뷰 선택
    view = st.radio("Select view:", ("시간대별", "요일별", "디자이너별"))

    if view == "시간대별":
        st.header("⏰ No-Show Rate by 시간대별")
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
        ax.set_xlabel('시간대별')
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

    elif view == "요일별":
        st.header("📅 No-Show Rate by 요일별")
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
        ax.set_xlabel('요일별')
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

    elif view == "디자이너별":
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

# 모델 분석 탭
with tab2:
    st.title("💡 모델 분석")

    st.subheader("📈 모델 성능 시각화")
    # 이미지 파일 경로 리스트 (실제 경로에 맞게 수정)
    image_info = [
        ("Confusion Matrix", 'models/xgboost/Evaluation Metrics/XGBoost_Threshold_ConfusionMatrix.png'),
        ("SHAP Feature Importance", 'models/xgboost/Evaluation Metrics/shap_summary_scatter.png'),
        ("Precision-Recall Curve", 'models/xgboost/Evaluation Metrics/PR Curve.png'),
        ("F1 Score vs Threshold", 'models/xgboost/Evaluation Metrics/XGBoost_thresholdvsF1score.png'),
        ("Precision/Recall/F1 vs Threshold", 'models/xgboost/Evaluation Metrics/XGBoost_Threshold_vs_Precision_Recall_F1Score.png')
    ]

    # 한 열에 두 개씩 이미지와 타이틀 배치
    cols = None
    for idx, (title, img_path) in enumerate(image_info):
        if idx % 2 == 0:
            cols = st.columns(2)
        with cols[idx % 2]:
            st.subheader(title)
            st.image(img_path, use_container_width=True)