import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot the ROC curve and calculate the AUC score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Evaluate the model using various metrics.
    """
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_pred_proba))

    # Plot ROC curve
    plot_roc_curve(y_true, y_pred_proba)


import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

def plot_model_performance(model, X_test, y_test, threshold=0.5):
    """
    모델의 성능을 한 번에 시각화하는 함수:
      1) ROC Curve
      2) Confusion Matrix (threshold 적용)
      3) Precision-Recall Curve

    Parameters:
    - model: 학습된 분류기 (predict_proba 지원)
    - X_test, y_test: 테스트 데이터
    - threshold: 이진 분류 임계값 (default=0.5)
    """
    # 1) 확률 예측 및 이진화
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    # 2) 서브플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_test, y_proba, ax=axes[0]
    )
    axes[0].set_title("ROC Curve")

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=axes[1]
    )
    axes[1].set_title(f"Confusion Matrix\n(threshold = {threshold})")

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, ax=axes[2]
    )
    axes[2].set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.show()

# 사용 예시:
# plot_model_performance(best_model, X_test, y_test, threshold=0.4)

