import sys
import json
from pathlib import Path

import torch
from PIL import Image

from config import MODELS_DIR, CLASS_MAPPING_JSON
from models.improved_cnn import ImprovedCNN
from augmentations import get_eval_transforms


def load_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CLASS_MAPPING_JSON) as f:
        class_mapping = json.load(f)
    idx_to_class = {v: k for k, v in class_mapping.items()}

    model = ImprovedCNN(num_classes=len(class_mapping)).to(device)
    ckpt_path = MODELS_DIR / "improved" / "improved_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return model, idx_to_class, device


@torch.no_grad()
def predict_image(model, idx_to_class, device, image: Image.Image, top_k: int = 3):
    
    transform = get_eval_transforms()
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).squeeze(0)

    top_probs, top_idxs = probs.topk(min(top_k, len(idx_to_class)))

    results = [
        (idx_to_class[idx.item()], prob.item())
        for idx, prob in zip(top_idxs, top_probs)
    ]
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    model, idx_to_class, device = load_model()
    image = Image.open(image_path)

    results = predict_image(model, idx_to_class, device, image, top_k=3)

    print(f"\nPredictions for {image_path.name}:\n")
    for class_name, prob in results:
        print(f"  {class_name:45s} {prob:.4f}")


if __name__ == "__main__":
    main()