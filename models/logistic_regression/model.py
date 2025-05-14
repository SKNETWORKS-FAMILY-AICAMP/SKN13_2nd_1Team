# 로지스틱 회귀 모델 최종본

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score
)

df = pd.read_csv('../../dataset/processed/hair_salon_data.csv')

X = df.drop(columns='noshow')
y = df['noshow']

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_idx = [X.columns.get_loc(col) for col in categorical_cols]
numeric_idx = [X.columns.get_loc(col) for col in numeric_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx),
        ('num', StandardScaler(), numeric_idx)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(random_state=0))
])

param_grid = {
    'classifier__penalty': ['l2'],  # 또는 ['l1', 'l2'] (solver에 따라 제한됨)
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'],  # 'l1'은 'liblinear', 'saga'만 지원
    'classifier__class_weight': ['balanced', None],
    'classifier__max_iter': [100, 200, 500]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1', # 평가 기준
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


# 최적의 threshold 탐색
y_proba = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.6, 0.05)
recalls, precisions, f1s, accuracies = [], [], [], []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    
    recalls.append(recall_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    accuracies.append(accuracy_score(y_test, y_pred))

best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

# print("베스트 임계값: ", best_threshold)


# 모델 저장 ({모델, threshold값} 객체로 저장.)
import pickle

model_bundle = {
    'model': best_model,
    'threshold': best_threshold
}

with open('logistic_model_bundle.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)
