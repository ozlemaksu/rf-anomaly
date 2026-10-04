from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(y_true, y_pred, y_prob=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        print("ROC-AUC:", roc_auc)
    else:
        print("ROC-AUC: y_prob verilmedi")

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc if y_prob is not None else None,
        "confusion_matrix": cm.tolist()
    }
def evaluate_by_sir(y_true, y_pred, sir_db):
    import pandas as pd

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "sir_db": sir_db
    })

    results = []

    for sir in sorted(df["sir_db"].dropna().unique()):
        part = df[df["sir_db"] == sir]

        f1 = f1_score(
            part["y_true"],
            part["y_pred"],
            zero_division=0
        )

        results.append({
            "sir_db": sir,
            "count": len(part),
            "f1": f1
        })

    result_df = pd.DataFrame(results)

    print("\nSIR bazlı sonuçlar:")
    print(result_df)

    return result_df
def plot_sir_f1(result_df, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(
        result_df["sir_db"],
        result_df["f1"],
        marker="o"
    )

    plt.xlabel("SIR (dB)")
    plt.ylabel("F1 Score")
    plt.title("SIR - F1 Performance")
    plt.grid(True)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Grafik oluşturuldu: {output_path}")