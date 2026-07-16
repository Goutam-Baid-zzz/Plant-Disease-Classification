import csv
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import MODELS_DIR, PLOTS_DIR, METADATA_DIR, ensure_dirs
from data_loader import get_dataloaders
from models.improved_cnn import ImprovedCNN


@torch.no_grad()
def get_predictions(model, loader, device):
    
    model.eval()
    all_true, all_pred, all_conf, all_paths = [], [], [], []

    # loader.dataset.rows gives us filepaths in the same order the DataLoader
    # would traverse them ONLY if shuffle=False — true for val/test loaders.
    dataset_rows = loader.dataset.rows
    row_idx = 0

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)

        batch_size = labels.size(0)
        batch_paths = [dataset_rows[row_idx + i]["filepath"] for i in range(batch_size)]
        row_idx += batch_size

        all_true.extend(labels.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_conf.extend(confs.cpu().tolist())
        all_paths.extend(batch_paths)

    return np.array(all_true), np.array(all_pred), np.array(all_conf), all_paths


def save_predictions_csv(y_true, y_pred, confidences, filepaths, idx_to_class, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_class", "pred_class", "confidence", "correct"])
        for path, t, p, c in zip(filepaths, y_true, y_pred, confidences):
            writer.writerow([
                path, idx_to_class[int(t)], idx_to_class[int(p)],
                f"{c:.4f}", int(t == p),
            ])


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm, annot=False, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Improved CNN (Test Set)")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return cm


def print_top_confusions(cm, class_names, top_n=10):
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_n} confused class pairs (true -> predicted : count):")
    for true_c, pred_c, count in confusions[:top_n]:
        print(f"  {true_c:45s} -> {pred_c:45s} : {count}")


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_mapping = get_dataloaders()
    idx_to_class = {v: k for k, v in class_mapping.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = ImprovedCNN(num_classes=len(class_mapping)).to(device)
    ckpt_path = MODELS_DIR / "improved" / "improved_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}\n")

    y_true, y_pred, confidences, filepaths = get_predictions(model, test_loader, device)

    # ── Classification report ────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    print(report)

    report_path = PLOTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # ── Confusion matrix ──────────────────────────────────
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    cm = plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")

    print_top_confusions(cm, class_names, top_n=10)

    # ── Raw predictions CSV (feeds Phase 7 & 8) ──────────
    preds_csv_path = METADATA_DIR / "test_predictions.csv"
    save_predictions_csv(y_true, y_pred, confidences, filepaths, idx_to_class, preds_csv_path)
    print(f"\nPer-sample predictions saved to: {preds_csv_path}")

    # ── Summary JSON ──────────────────────────────────────
    summary = {
        "test_accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_test_samples": int(len(y_true)),
    }
    summary_path = METADATA_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()