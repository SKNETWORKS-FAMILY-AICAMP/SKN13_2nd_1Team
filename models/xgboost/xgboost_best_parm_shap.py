import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve

# 데이터 로드
hair_salon_data = pd.read_csv('dataset/processed/hair_salon_data.csv')

# 카테고리형 컬럼별로 LabelEncoder 따로 적용
categorical_cols = ['book_tod', 'book_dow', 'book_category', 'book_staff', 
                    'last_staff', 'last_dow', 'last_tod']
for col in categorical_cols:
    le = LabelEncoder()
    hair_salon_data[col] = le.fit_transform(hair_salon_data[col].astype(str))

# 피처 컬럼 정의
feature_cols = [
    *categorical_cols,
    'last_receipt_tot', 'last_prod_flag', 'last_cumrev', 'last_cumbook', 
    'last_cumstyle', 'last_cumcolor', 'last_cumprod', 'last_cumcancel', 
    'last_cumnoshow', 'recency', 'first_visit'
]

X = hair_salon_data[feature_cols].values
y = hair_salon_data['noshow'].values

# 클래스 비율 계산 후 scale_pos_weight 계산
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos
print(f"Class ratio (neg:pos): {neg}:{pos} => scale_pos_weight: {scale_pos_weight:.2f}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 하이퍼파라미터 (예시)
best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 3,
    'n_estimators': 100,
    'subsample': 0.8
}

# 모델 생성
xgb_best = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    **best_params
)

# 교차검증 F1 점수 계산 및 출력 (학습 전에)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_best, X_train, y_train, scoring='f1', cv=cv)
print(f"📊 5-Fold F1 Scores: {scores}")
print(f"✅ 평균 F1 Score: {scores.mean():.4f}")

# 모델 학습
xgb_best.fit(X_train, y_train)

# 테스트셋 예측 및 평가
y_pred = xgb_best.predict(X_test)
y_proba = xgb_best.predict_proba(X_test)[:, 1]

print("===== Test Set Classification Report =====")
print(classification_report(y_test, y_pred, digits=4))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("ROC AUC:  ", roc_auc_score(y_test, y_proba))

# 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_best.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test Set)")
plt.show()

# SHAP 해석
explainer = shap.Explainer(xgb_best)
shap_values = explainer(X_test)

# SHAP summary plot 저장 (화면 출력 대신)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.savefig('shap_summary_scatter (변수 제거).png')
plt.close()

shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type='bar', show=False)
plt.savefig('shap_summary_bar (변수 제거).png')
plt.close()

# PR Curve 그리기
from sklearn.metrics import precision_recall_curve, auc
probs = xgb_best.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig('PR Curve (변수 제거)')
plt.close()

