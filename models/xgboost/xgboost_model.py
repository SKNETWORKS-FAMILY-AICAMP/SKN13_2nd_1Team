import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np


# load the dataset
hair_salon_data = pd.read_csv('dataset/processed/hair_salon_data.csv')

# string data to numerical data
le = LabelEncoder()
for col in ['book_tod', 'book_dow', 'book_category', 'book_staff',
            'last_category', 'last_staff', 'last_dow', 'last_tod']:
        hair_salon_data[col] = le.fit_transform(hair_salon_data[col])

# Train-test split
X = hair_salon_data[['book_tod', 'book_dow', 'book_category', 'book_staff',
      'last_category', 'last_staff', 'last_day_services', 'last_receipt_tot',
      'last_dow', 'last_tod', 'last_noshow', 'last_prod_flag', 'last_cumrev',
      'last_cumbook', 'last_cumstyle', 'last_cumcolor', 'last_cumprod',
      'last_cumcancel', 'last_cumnoshow', 'recency','first_visit', 'is_revisit_30days']]

y = hair_salon_data['noshow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

# SMOTE oversampling
new_X_train, new_y_train = SMOTE().fit_resample(X_train, y_train)
new_X_val, new_y_val = SMOTE().fit_resample(X_val, y_val)
new_X_test, new_y_test = SMOTE().fit_resample(X_test, y_test)

# XGBoost Classifier initialization
xgb_clf = XGBClassifier(eval_metric='logloss')
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300, 400],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.5, 0.75, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}

# GridSearchCV
grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring='recall',
    verbose=1,
    n_jobs=-1
)

# print(thr_best_result)
grid.fit(new_X_train, new_y_train)

best_model = grid.best_estimator_
print("Best Parameters: ", grid.best_params_)
print("Best Score: ", grid.best_score_)
print("Best Recall: ", grid.best_estimator_.score(new_X_val, new_y_val))
print("Best F1 Score: ", f1_score(new_y_val, grid.best_estimator_.predict(new_X_val)))
print("Best ROC AUC: ", roc_auc_score(new_y_val, grid.best_estimator_.predict_proba(new_X_val)[:, 1]))
print("Best Classification Report: ")
print(classification_report(new_y_val, grid.best_estimator_.predict(new_X_val)))
print("====================================================")

# Plotting ROC curve
y_pred_proba = grid.best_estimator_.predict_proba(new_X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(new_y_val, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(new_y_val, y_pred_proba))

threshold_list = [0.35, 0.4, 0.45, 0.5]
results = []
probs = best_model.predict_proba(new_X_val)[:, 1]

for thr in threshold_list:
        y_pred = (probs >= thr).astype(int)
        results.append({
                'threshold': thr,
                'f1_score': f1_score(new_y_test, y_pred),
                'recall': recall_score(new_y_test, y_pred),
                'precision': precision_score(new_y_test, y_pred),
                'accuracy': accuracy_score(new_y_test, y_pred)
        })

results_df = pd.DataFrame(results)
print(results_df)

# Plotting the ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Save the model
import pickle
with open('models/xgboost/xgboost_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)



















