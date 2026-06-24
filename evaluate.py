import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


CLASS_NAMES = ['Low', 'Medium', 'High']


def get_predictions(model: tf.keras.Model,
                    X: np.ndarray) -> tuple:
    """
    Run inference and return both probability arrays and integer class indices.

    Args:
        model: Trained Keras model.
        X:     Scaled input features.

    Returns:
        Tuple (y_proba: shape (n, 3), y_pred: shape (n,) integer indices).
    """
    y_proba = model.predict(X, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)
    return y_proba, y_pred


def evaluate_model(model: tf.keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   class_names: list = CLASS_NAMES) -> dict:
    """

    Args:
        model:       Trained Keras model.
        X_test:      Scaled test features.
        y_test:      One-hot encoded true labels.
        class_names: String names for each class.

    Returns:
        Dictionary with all metrics and raw arrays for downstream use.
    """

    y_true         = np.argmax(y_test, axis=1)
    y_proba, y_pred = get_predictions(model, X_test)

    # Compute aggregate metrics (weighted by class support)
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(   y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(       y_true, y_pred, average='weighted', zero_division=0)

    # Print a clean results block
    print("\n" + "=" * 55)
    print("TEST SET:  MODEL EVALUATION")
    print("=" * 55)
    print(f"Samples evaluated : {len(y_true)}")
    print(f"Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Precision (wtd) : {precision:.4f}")
    print(f"Recall (wtd) : {recall:.4f}")
    print(f"F1-Score (wtd) : {f1:.4f}")
    print("=" * 55)
    print("\n  Per-class breakdown:")
    print(classification_report(y_true, y_pred,target_names=class_names,digits=4))

    return {
        'accuracy':  acc,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'y_true':    y_true,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'report':    classification_report(y_true, y_pred,target_names=class_names,output_dict=True),
    }


def analyze_by_risk_level(y_true:np.ndarray, y_pred:np.ndarray, class_names:  list = CLASS_NAMES) -> pd.DataFrame:
    """
    Break down classification performance by true risk category.
    By grouping test patients by their actual
    risk level and measuring class-specific accuracy, we know if the model reliably catches High-risk patients, the risk tier is most often misclassified,
    and into the adjacent category it falls into
    we also know if Low-risk patients ever incorrectly escalated to High.

    Args:
        y_true:      Integer true labels (0=Low, 1=Medium, 2=High).
        y_pred:      Integer predicted labels.
        class_names: List of class name strings.

    Returns:
        DataFrame with per-class accuracy, counts, and misclassification totals.
    """
    records = []
    for i, name in enumerate(class_names):
        mask    = (y_true == i)
        total   = int(mask.sum())
        correct = int((y_pred[mask] == i).sum())
        wrong   = total - correct
        pct     = (correct / total * 100) if total > 0 else 0.0
        records.append({
            'Risk Level':              name,
            'True Positives in Test':  total,
            'Correctly Classified':    correct,
            'Misclassified':           wrong,
            'Class Accuracy (%)':      round(pct, 2),
        })

    df = pd.DataFrame(records)
    print("\n  Per-risk-level accuracy (categorical breakdown):")
    print("  " + df.to_string(index=False).replace('\n', '\n  '))
    return df



def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = CLASS_NAMES,save_path: str  = 'confusion_matrix.png') -> None:
    """
    Plot and save a labelled confusion matrix as a heatmap.

    Args:
        y_true:      Integer true labels.
        y_pred:      Integer predicted labels.
        class_names: Axis tick labels.
        save_path:   File path for saving the figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                linecolor='white',
                ax=ax)

    ax.set_title('Confusion Matrix. Cancer Risk ANN\n(Test Set)', fontsize=13, pad=14)
    ax.set_xlabel('Predicted Risk Level', fontsize=11, labelpad=8)
    ax.set_ylabel('True Risk Level', fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Confusion matrix saved → '{save_path}'.")


def plot_training_history(history:tf.keras.callbacks.History, save_path: str = 'training_history.png') -> None:
    """
    Plot training and validation accuracy/loss curves side by side.

    Args:
        history:   Keras History from model.fit().
        save_path: File path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history.history['accuracy'],
                 label='Train', linewidth=2, color='steelblue')
    axes[0].plot(history.history['val_accuracy'],
                 label='Validation', linewidth=2, linestyle='--', color='darkorange')
    axes[0].set_title('Accuracy Over Epochs', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Accuracy', fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(history.history['loss'],
                 label='Train', linewidth=2, color='steelblue')
    axes[1].plot(history.history['val_loss'],
                 label='Validation', linewidth=2, linestyle='--', color='darkorange')
    axes[1].set_title('Loss Over Epochs', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Categorical Crossentropy Loss', fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.suptitle('ANN Training History.  Cancer Risk Classification',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Training history plot saved → '{save_path}'.")


def plot_class_probability_distribution(y_proba: np.ndarray,
                                         y_true: np.ndarray,
                                         class_names: list = CLASS_NAMES,
                                         save_path: str  = 'probability_distribution.png') -> None:
    """
    Plot the distribution of predicted probabilities for each true class.

    Two rows per class:
      Top row: full range (0.0 – 1.0) to show the overall picture.
      Bottom row: zoomed to (min - 0.01, 1.01) so high-confidence
        distributions are actually visible instead of all cramming
        into the rightmost histogram bin.



    Args:
        y_proba:     Softmax probability arrays shape (n, 3).
        y_true:      Integer true labels (0/1/2).
        class_names: Class name strings for titles.
        save_path:   File path to save the figure.
    """
    COLORS = {'Low': 'steelblue', 'Medium': 'darkorange', 'High': 'crimson'}
    n_classes = len(class_names)

    fig, axes = plt.subplots(2, n_classes, figsize=(14, 7))

    for i, name in enumerate(class_names):
        mask            = (y_true == i)
        proba_for_class = y_proba[mask, i]
        mean_p          = proba_for_class.mean()
        color           = COLORS.get(name, 'steelblue')

        # Determine the zoom range for the bottom row.
        # If everything is above 0.9, zoom into [0.85, 1.02] so the
        # distribution shape is actually visible.
        p_min = float(proba_for_class.min())
        if p_min > 0.85:
            zoom_low  = max(0.0, p_min - 0.05)
            zoom_bins = np.linspace(zoom_low, 1.005, 20)
        else:
            zoom_low  = max(0.0, p_min - 0.05)
            zoom_bins = 20

        ax_full = axes[0, i]
        ax_full.hist(proba_for_class, bins=np.linspace(0, 1, 21),
                     color=color, edgecolor='white', alpha=0.8)
        ax_full.axvline(mean_p, color='black', linestyle='--',
                        linewidth=1.8, label=f'Mean = {mean_p:.3f}')
        ax_full.set_title(f'True {name} Risk  (n={int(mask.sum())})',
                          fontsize=11, fontweight='bold')
        ax_full.set_xlabel('Predicted Probability', fontsize=9)
        ax_full.set_ylabel('Count', fontsize=9)
        ax_full.set_xlim([0, 1])
        ax_full.legend(fontsize=8)
        ax_full.grid(alpha=0.3)
        if i == 0:
            ax_full.text(0.02, 0.96, 'Full range', transform=ax_full.transAxes,
                         fontsize=8, color='grey', va='top')

        ax_zoom = axes[1, i]
        ax_zoom.hist(proba_for_class, bins=zoom_bins,
                     color=color, edgecolor='white', alpha=0.8)
        ax_zoom.axvline(mean_p, color='black', linestyle='--',
                        linewidth=1.8, label=f'Mean = {mean_p:.3f}')
        ax_zoom.set_xlabel('Predicted Probability (zoomed)', fontsize=9)
        ax_zoom.set_ylabel('Count', fontsize=9)
        ax_zoom.set_xlim([zoom_low, 1.005])
        ax_zoom.legend(fontsize=8)
        ax_zoom.grid(alpha=0.3)
        if i == 0:
            ax_zoom.text(0.02, 0.96, 'Zoomed', transform=ax_zoom.transAxes,
                         fontsize=8, color='grey', va='top')

        # Annotate % of samples above 0.99 confidence (useful metric)
        pct_certain = (proba_for_class >= 0.99).mean() * 100
        ax_zoom.text(0.98, 0.92, f'{pct_certain:.0f}% ≥ 0.99',
                     transform=ax_zoom.transAxes, fontsize=8,
                     ha='right', color='darkgreen', fontweight='bold')

    fig.suptitle('Confidence Distribution by True Risk Class\n (Top: full 0–1 range  |  Bottom: zoomed to actual distribution)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Probability distribution plot saved → '{save_path}'.")