import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
)
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle


hair_salon_data = pd.read_csv('dataset/processed/hair_salon_data.csv')

# 범주형/수치형 컬럼 정의 및 Label Encoding
categorical_cols = ['book_tod', 'book_dow', 'book_category', 'book_staff',
                    'last_category', 'last_staff', 'last_dow', 'last_tod']

numerical_cols = ['last_day_services', 'last_receipt_tot', 'last_noshow',
                  'last_prod_flag', 'last_cumrev', 'last_cumbook', 'last_cumstyle',
        'last_cumcolor', 'last_cumprod', 'last_cumcancel', 'last_cumnoshow',
        'recency','first_visit', 'is_revisit_30days']

# Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    hair_salon_data[col] = le.fit_transform(hair_salon_data[col].astype(str))

# Feature and target variable
X = hair_salon_data[categorical_cols + numerical_cols]
y = hair_salon_data['noshow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# SMOTE oversampling
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), numerical_cols)
], remainder='passthrough')

# KNN pipeline
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

knn_param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__p': [1, 2]
}

# GridSearchCV
knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_param_grid,
    cv=3,
    scoring='recall',
    verbose=1,
    n_jobs=-1
)

knn_grid.fit(X_train, y_train)
best_knn_model = knn_grid.best_estimator_
print("Best parameters found: ", knn_grid.best_params_)
print("Best recall score: ", knn_grid.best_score_)
print("Best KNN model: ", best_knn_model)

# 최적 모델로 다시 학습습
# best_knn_model.fit(X_train, y_train)
# y_pred = best_knn_model.predict(X_test)
# y_pred_proba = best_knn_model.predict_proba(X_test)[:, 1]

y_test_pred = best_knn_model.predict(X_test)
y_test_pred_proba = best_knn_model.predict_proba(X_test)[:, 1]
print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))
print("Test Precision: ", precision_score(y_test, y_test_pred))
print("Test Recall: ", recall_score(y_test, y_test_pred))
print("Test F1 Score: ", f1_score(y_test, y_test_pred))
print("Test ROC AUC: ", roc_auc_score(y_test, y_test_pred_proba))
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
disp.ax_.set_title('kNN Confusion Matrix')
plt.show()

# ROC Curve
roc_disp = RocCurveDisplay.from_predictions(y_test, y_test_pred)
roc_disp.ax_.set_title('kNN ROC Curve')
plt.show()

# Precision-Recall Curve
pr_disp = PrecisionRecallDisplay.from_predictions(y_test, y_test_pred)
pr_disp.ax_.set_title('kNN Precision-Recall Curve')
plt.show()

# 12. 모델 저장 (.pkl)
with open('models/kNN/best_knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn_model, f)
print("Saved kNN model to 'best_knn_model.pkl'")

