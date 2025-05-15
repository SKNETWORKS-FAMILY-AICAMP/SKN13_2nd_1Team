import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve
# from sklearn.pipeline import Pipeline  
from imblearn.pipeline import Pipeline  # pipeline 안에서 SMOTE 사용시 사용
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle
import os


# load the dataset
hair_salon_data = pd.read_csv('dataset/processed/hair_salon_data.csv')

# string data to numerical data
le = LabelEncoder()
categorical_cols = ['book_tod', 'book_dow', 'book_category', 'book_staff',
                       'last_category', 'last_staff', 'last_dow', 'last_tod']
for col in categorical_cols:
    hair_salon_data[col] = le.fit_transform(hair_salon_data[col].astype(str))

# Define feature columns
feature_cols = [
      *categorical_cols,
        'last_day_services', 'last_receipt_tot', 'last_noshow',
        'last_prod_flag', 'last_cumrev', 'last_cumbook', 'last_cumstyle',
        'last_cumcolor', 'last_cumprod', 'last_cumcancel', 'last_cumnoshow',
        'recency','first_visit', 'is_revisit_30days']

X = hair_salon_data[feature_cols]
y = hair_salon_data['noshow']

# caculate class ratio for scale_pos_weight
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos
print(f"Class ratio (neg:pos): {neg}:{pos} => scale_pos_weight: {scale_pos_weight:.2f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# XGBoost Classifier initialization
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42
    )

pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', xgb_clf)                
])

param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.75, 0.8, 1.0],
    'classifier__min_child_weight': [1, 3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
grid = GridSearchCV(
    estimator=pipeline,  # pipeline 사용시 pipeline 객체 사용
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)

# Fit the model
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Parameters: ", grid.best_params_)

# result analysis
print("====================================================")
print("Best Score: ", grid.best_score_)
print("Best Recall: ", grid.best_estimator_.score(X_train, y_train))
print("Best F1 Score: ", f1_score(y_train, grid.best_estimator_.predict(X_train)))
print("Best ROC AUC: ", roc_auc_score(y_train, grid.best_estimator_.predict_proba(X_train)[:, 1]))
print("Best Classification Report: ")
print(classification_report(y_train, grid.best_estimator_.predict(X_train)))
print("====================================================")

threshold_list = [0.35, 0.4, 0.45, 0.5]
results = []
probs = best_model.predict_proba(X_test)[:, 1]
# for thr in threshold_list:

#     y_pred = (probs >= thr).astype(int)

#     f1 = f1_score(y_test, y_pred)
#     results.append({
#         'threshold': thr,
#         'f1_score': f1,
#         'recall': recall_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred),
#         'accuracy': accuracy_score(y_test, y_pred)
#     })
#     print(f"\n=== Threshold: {thr} ===")
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))

#     results_df = pd.DataFrame(results)
#     print(results_df)
#     print("====================================================")
#     confusion = confusion_matrix(y_test, y_pred)
#     print('Confusion Matrix:')
#     print(confusion)

thresholds = np.arange(0.4, 0.61, 0.01)

results = []

for t in thresholds:
    y_pred = (probs >= t).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results.append({
        'threshold': round(t, 3),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df['f1_score'].idxmax()]
best_threshold = best_row['threshold']

print("✅ 최적 threshold 탐색 결과:")
print(results_df)
print("\n✅ F1 최고점 기준 최적 threshold:")
print(best_row)
    
model_package = {
    "model": best_model,
    "threshold": best_threshold
}

with open('models/xgboost/xgboost_model_with_threshold.pkl', 'wb') as f:
    pickle.dump(model_package, f)
    
print("✅ 모델과 threshold가 저장되었습니다.")


# import metrics
# metrics.plot_all_metrics(best_model, X_test, y_test, 'XGBoost')
# metrics.plot_precision_recall_curve(best_model, X_test, y_test, "XGBoost")

from sklearn.metrics import precision_recall_curve, auc

probs = best_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
pr_auc = auc(recall, precision)

# PR Curve 그리기
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()

# Save the model
# with open('models/xgboost/xgboost_model.pkl', 'wb') as f:
#     pickle.dump(best_model, f)
















