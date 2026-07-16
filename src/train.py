import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import (
    MODELS_DIR, LOGS_DIR, PLOTS_DIR, SEED,
    LEARNING_RATE, BASELINE_EPOCHS, EARLY_STOPPING_PATIENCE, ensure_dirs,
)
from data_loader import get_dataloaders
from models.baseline_cnn import BaselineCNN


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def plot_curves(history: dict, save_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_training(
    model,
    model_name: str,
    epochs: int,
    train_loader,
    val_loader,
    device,
    checkpoint_dir: Path,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
):
    
    ensure_dirs()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = checkpoint_dir / f"{model_name}_best.pt"

    print(f"Training {model_name} on {device} for up to {epochs} epochs "
          f"(early stopping patience={patience})\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> val_loss improved, checkpoint saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val_loss improvement for {patience} epochs)")
                break

    # Save training history + curves
    history_path = LOGS_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_path = PLOTS_DIR / f"{model_name}_curves.png"
    plot_curves(history, plot_path, title=model_name)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"History saved to: {history_path}")
    print(f"Curves saved to : {plot_path}")
    print(f"Best checkpoint : {best_ckpt_path}")

    return history, best_ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=BASELINE_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_mapping = get_dataloaders()
    num_classes = len(class_mapping)

    model = BaselineCNN(num_classes=num_classes).to(device)

    run_training(
        model=model,
        model_name="baseline",
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=MODELS_DIR / "baseline",
        lr=args.lr,
    )


if __name__ == "__main__":
    main()