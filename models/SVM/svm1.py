# import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv("C:\\SKN13SM\\SKN13_2nd_1Team\\SKN13_2nd_1Team\\dataset\\processed\\hair_salon_data.csv")
data.head()

X = data.drop(columns=['ID', 'noshow']) # 파생변수 'first_visit', 'is_revisit_30days' 포함
y = data['noshow']

# 범주형 변수 인코딩 (One-Hot Encoding)
categorical_cols = ['book_tod', 'book_dow', 'book_category', 'book_staff',
                    'last_category', 'last_staff', 'last_dow', 'last_tod',
                    'last_noshow', 'last_prod_flag', 'first_visit', 'is_revisit_30days']

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 수치형 변수 스케일링
# 수치형만 선택
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

# 스케일링
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# 범주형 변수 one-hot 인코딩 결과가 bool 형태이므로 int형으로 변경한다.
cate_cols = X_train.select_dtypes(include=["bool"]).columns
X_train_scaled[cate_cols] = X_train_scaled[cate_cols].astype(int)
X_test_scaled[cate_cols] = X_test_scaled[cate_cols].astype(int)

# Grid Search로 최적의 파라미터를 찾음
# 모델 1: recall이 높은 모델 (파생변수 포함)
svm1 = SVC(
    kernel="rbf", 
    C=0.1, 
    gamma=1,
    class_weight='balanced',
    probability=True)  

# 모델 학습
svm1.fit(X_train_scaled, y_train)

# 예측
y_pred = svm1.predict(X_test_scaled)
y_proba = svm1.predict_proba(X_test_scaled)[:, 1]

# 평가

print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_proba))

cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["visit", "noshow"])
cm_display.plot(cmap="Greens")
plt.title("Confusion Matrix - SVM Model", fontsize=20)
plt.show()

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - SVM Model")
plt.grid(True)
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title("Precision-Recall Curve - SVM Model")
plt.grid(True)
plt.show()


# 모델 저장
joblib.dump(svm1, "models/SVM/svm1_model.pkl")