import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import MODELS_DIR, GRADCAM_DIR, METADATA_DIR, IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, ensure_dirs
from models.improved_cnn import ImprovedCNN
from augmentations import get_eval_transforms


class GradCAM:

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool gradients over spatial dims -> per-channel weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()  # normalize to [0, 1]

        return cam, class_idx, torch.softmax(output, dim=1)[0, class_idx].item()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(NORMALIZE_MEAN).reshape(3, 1, 1)
    std = np.array(NORMALIZE_STD).reshape(3, 1, 1)
    img = tensor.detach().cpu().numpy() * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))  # CHW -> HWC


def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR
    )) / 255.0

    heatmap = cm.jet(cam_resized)[:, :, :3]  # drop alpha channel from colormap
    overlaid = (1 - alpha) * image_np + alpha * heatmap
    return np.clip(overlaid, 0, 1)


def explain_sample(gradcam: GradCAM, filepath: str, true_class: str, pred_class: str,
                    device, save_path: Path):
    transform = get_eval_transforms()
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    cam, class_idx, confidence = gradcam.generate(input_tensor)
    image_np = denormalize_image(input_tensor.squeeze(0))
    overlay = overlay_heatmap(image_np, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    correct = "CORRECT" if true_class == pred_class else "INCORRECT"
    fig.suptitle(f"[{correct}] True: {true_class} | Pred: {pred_class} ({confidence:.2f})", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedCNN(num_classes=38).to(device)
    ckpt_path = MODELS_DIR / "improved" / "improved_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")

    # Target the last conv block's conv layer (before pooling) — standard
    # Grad-CAM practice: deepest conv layer for class-discriminative features,
    # but before spatial info is fully collapsed by GAP.
    target_layer = model.features[-1].block[0]  # last ConvBlock's Conv2d

    gradcam = GradCAM(model, target_layer)

    preds_csv = METADATA_DIR / "test_predictions.csv"
    if not preds_csv.exists():
        raise FileNotFoundError(
            f"{preds_csv} not found — run evaluate.py first (Phase 6) to generate it."
        )

    with open(preds_csv) as f:
        rows = list(csv.DictReader(f))

    correct_rows = [r for r in rows if r["correct"] == "1"]
    incorrect_rows = [r for r in rows if r["correct"] == "0"]

    print(f"Total predictions: {len(rows)} | Correct: {len(correct_rows)} | Incorrect: {len(incorrect_rows)}")

    n_correct_samples = min(10, len(correct_rows))
    n_incorrect_samples = min(10, len(incorrect_rows))

    import random
    random.seed(42)
    sampled_correct = random.sample(correct_rows, n_correct_samples)
    sampled_incorrect = random.sample(incorrect_rows, n_incorrect_samples) if incorrect_rows else []

    for i, row in enumerate(sampled_correct):
        save_path = GRADCAM_DIR / f"correct_{i:02d}.png"
        explain_sample(gradcam, row["filepath"], row["true_class"], row["pred_class"], device, save_path)

    for i, row in enumerate(sampled_incorrect):
        save_path = GRADCAM_DIR / f"incorrect_{i:02d}.png"
        explain_sample(gradcam, row["filepath"], row["true_class"], row["pred_class"], device, save_path)

    print(f"\nSaved {n_correct_samples} correct + {n_incorrect_samples} incorrect Grad-CAM examples to {GRADCAM_DIR}")


if __name__ == "__main__":
    main()