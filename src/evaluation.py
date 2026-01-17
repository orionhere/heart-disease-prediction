from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the trained model on test data and plots results.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\nFinal Test Accuracy ({model_name}):", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title(f"Confusion Matrix ({model_name})")
    ax[0].set_ylabel("True Label")
    ax[0].set_xlabel("Predicted Label")

    # ROC Curve
    if y_proba is not None:
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1])
        ax[1].set_title(f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.3f})")
        ax[1].plot([0, 1], [0, 1], 'k--')
    else:
        ax[1].text(0.5, 0.5, "ROC not available", ha='center')

    plt.tight_layout()
    output_path = f"reports/figures/evaluation_{model_name.replace(' ', '_')}.png"
    plt.savefig(output_path)
    print(f"Evaluation plot saved to {output_path}")
    plt.close()
