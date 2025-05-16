import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 모델 & 인코더 로딩
with open('../models/xgboost/xgboost_model_with_threshold.pkl', 'rb') as f:
    model_bundle = pickle.load(f)
with open('../models/xgboost/label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

model = model_bundle['model']
threshold = model_bundle.get('threshold', 0.4)  # 기본값 0.4

# 컬럼 정의
categorical_cols = ['book_tod', 'book_dow', 'book_category', 'book_staff',
                    'last_category', 'last_staff', 'last_dow', 'last_tod']
numeric_cols = ['last_day_services', 'last_noshow',
                'last_cumrev', 'last_cumbook', 'last_cumstyle', 'last_cumcolor',
                'last_cumnoshow', 'recency',
                'first_visit', 'is_revisit_30days']

# 세션 초기화
if 'registered_customers' not in st.session_state:
    st.session_state.registered_customers = pd.DataFrame(columns=['name'] + categorical_cols + numeric_cols)

# 앱 제목
st.title("🌳 XGBoost 노쇼 예측기")

# ---------------------------
# 1. 고객 등록 영역
# ---------------------------
st.header("🧾 고객 등록")

with st.form("customer_form"):
    name = st.text_input("고객 이름을 입력하세요", key="name_input")
    input_dict = {'name': name}

    st.markdown("📋 **예약 정보 입력 (범주형)**")
    # 범주형 입력: 3개씩 열 분할
    cat_chunks = [categorical_cols[i:i+3] for i in range(0, len(categorical_cols), 3)]
    for chunk in cat_chunks:
        cols = st.columns(len(chunk))
        for i, col in enumerate(chunk):
            input_dict[col] = cols[i].selectbox(f"{col}", encoders[col].classes_, key=col)

    st.markdown("🔢 **고객 수치 데이터 입력**")
    # 수치형 입력: 3개씩 열 분할
    num_chunks = [numeric_cols[i:i+3] for i in range(0, len(numeric_cols), 3)]
    for chunk in num_chunks:
        cols = st.columns(len(chunk))
        for i, col in enumerate(chunk):
            input_dict[col] = cols[i].number_input(f"{col}", min_value=0, key=col)

    submitted = st.form_submit_button("고객 등록")
    if submitted:
        if not name:
            st.warning("고객 이름을 입력해주세요.")
        else:
            new_row = pd.DataFrame([input_dict])
            st.session_state.registered_customers = pd.concat(
                [st.session_state.registered_customers, new_row], ignore_index=True
            )
            st.success(f"{name} 고객이 등록되었습니다.")


# ---------------------------
# 2. 고객 선택 및 예측
# ---------------------------
st.header("🔍 등록 고객 예측")

if st.session_state.registered_customers.empty:
    st.info("등록된 고객이 없습니다.")
else:
    customer_names = st.session_state.registered_customers['name'].tolist()
    selected_name = st.selectbox("예측할 고객 선택", customer_names)

    selected_row = st.session_state.registered_customers[
        st.session_state.registered_customers['name'] == selected_name
    ].iloc[0]

    st.markdown("📋 **예약 정보 수정 가능**")
    cat_values = {}
    cols = st.columns(3)
    for idx, col in enumerate(categorical_cols):
        with cols[idx % 3]:
            cat_values[col] = st.selectbox(
                f"{col}", 
                encoders[col].classes_,
                index=list(encoders[col].classes_).index(selected_row[col])
            )

    st.markdown("🔢 **수치형 정보 수정 가능**")
    num_values = {}
    cols = st.columns(3)
    for idx, col in enumerate(numeric_cols):
        with cols[idx % 3]:
            num_values[col] = st.number_input(f"{col}", value=int(selected_row[col]), min_value=0)

    if st.button("노쇼 예측하기"):
        # 하나의 row로 구성된 예측용 DataFrame
        input_data = {**cat_values, **num_values}
        input_df = pd.DataFrame([input_data])

        # 범주형 인코딩
        for col in categorical_cols:
            input_df[col] = encoders[col].transform(input_df[col])

        y_proba = model.predict_proba(input_df)[:, 1][0]
        y_pred = int(y_proba >= threshold)

        st.subheader("📊 예측 결과")
        st.write(f"노쇼 확률: **{y_proba:.2%}**")
        if y_pred:
            st.error("❌ 노쇼 가능성이 높습니다!")
        else:
            st.success("✅ 노쇼 가능성은 낮습니다.")

