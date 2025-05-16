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
    roc_curve,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)


def plot_roc_curve(model, X, y, model_name):
    """
    ROC 커브를 그리는 함수

    Parameters:
    - model: 학습된 분류 모델 (predict_proba 지원)
    - X: 특징 데이터
    - y: 실제 레이블
    - model_name: 시각화에 사용할 모델 이름
    """
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(
        model, X, y, ax=ax, name=model_name
    )
    ax.set_title(f"ROC Curve ({model_name})")
    ax.legend()
    plt.show()


def plot_confusion_matrix(model, X, y, model_name):
    """
    혼동 행렬을 그리는 함수

    Parameters:
    - model: 학습된 분류 모델 (predict 메서드 사용)
    - X: 특징 데이터
    - y: 실제 레이블
    - model_name: 시각화에 사용할 모델 이름
    """
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, ax=ax
    )
    ax.set_title(f"Confusion Matrix ({model_name})")
    plt.show()


def plot_precision_recall_curve(model, X, y, model_name):
    """
    Precision-Recall 커브를 그리는 함수

    Parameters:
    - model: 학습된 분류 모델 (predict_proba 지원)
    - X: 특징 데이터
    - y: 실제 레이블
    - model_name: 시각화에 사용할 모델 이름
    """
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(
        model, X, y, ax=ax, name=model_name
    )
    ax.set_title(f"Precision-Recall Curve ({model_name})")
    ax.legend()
    plt.show()


def plot_all_metrics(model, X, y, model_name):
    """
    ROC, Confusion Matrix, Precision-Recall 커브를 한 번에 그리는 함수

    Parameters:
    - model: 학습된 분류 모델
    - X: 특징 데이터
    - y: 실제 레이블
    - model_name: 시각화에 사용할 모델 이름
    """
    plot_roc_curve(model, X, y, model_name)
    plot_confusion_matrix(model, X, y, model_name)
    plot_precision_recall_curve(model, X, y, model_name)