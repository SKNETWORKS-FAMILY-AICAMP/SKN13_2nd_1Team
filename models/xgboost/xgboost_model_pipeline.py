import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.pipeline import Pipeline  # pipeline 안에서 SMOTE 사용시 사용
# from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)



# preprocess and compose pipeline
# preprocessor = ColumnTransformer([
#     ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
# ])

# pipeline = Pipeline([
#     ('smote', SMOTE()),
#     ('classifier', XGBClassifier(eval_metric='logloss',
#                                  n_jobs=-1))
# ])

# SMOTE oversampling
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
X_test, y_test = SMOTE().fit_resample(X_test, y_test)

# XGBoost Classifier initialization
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__n_estimators': [50, 100, 200, 300, 400],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.5, 0.75, 0.8, 1.0],
    'classifier__min_child_weight': [1, 3, 5],
}

# GridSearchCV
grid = GridSearchCV(
    estimator=xgb_clf,  # pipeline 사용시 pipeline 객체 사용
    param_grid=param_grid,
    cv=3,
    scoring='recall',
    verbose=1,
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
for thr in threshold_list:

    y_pred = (probs >= thr).astype(int)

    f1 = f1_score(y_test, y_pred)
    results.append({
        'threshold': thr,
        'f1_score': f1,
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred)
    })

    results_df = pd.DataFrame(results)
    print(results_df)
    print("====================================================")
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(confusion)


# results_df.set_index('thr', inplace=True)
# results_df.plot(figsize=(10, 6), kind='line')
# plt.title('XGBoost Model Performance')
# plt.xlabel('Threshold')
# plt.ylabel('Score')
# plt.legend(loc='best')
# plt.grid()
# plt.show()

import metrics
metrics.plot_all_metrics(best_model, X_test, y_test, 'XGBoost')

# Save the model
with open('models/xgboost/xgboost_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
















