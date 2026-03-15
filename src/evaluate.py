import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN
from utils import get_device, ensure_dir


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # 확률이 한 클래스에 몰리면 entropy 낮고, 여러 클래스에 퍼지면 entropy 높음
    
    return -(probs * torch.log(probs + eps)).sum(dim=1)


@torch.no_grad()
def main():
    ensure_dir("../outputs")
    ensure_dir("../data")

    device = get_device()
    print(f"Using device: {device}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = datasets.CIFAR10(
        root="../data",
        train=False,
        download=True,
        transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = SimpleCNN(num_classes=10).to(device)
    model_path = os.path.join("../outputs", "best_cnn.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []
    sample_id = 0

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)

        confidences, preds = torch.max(probs, dim=1)  # softmax 확률 중 가장 큰 값
        entropies = entropy_from_probs(probs)
        corrects = (preds == labels).long()

        probs_np = probs.cpu().numpy()

        for i in range(images.size(0)):
            row = {
                "sample_id": sample_id,
                "true_label": int(labels[i].cpu().item()),
                "pred_label": int(preds[i].cpu().item()),
                "correct": int(corrects[i].cpu().item()),
                "confidence": float(confidences[i].cpu().item()),
                "uncertainty_entropy": float(entropies[i].cpu().item()),
            }

            for cls_idx in range(probs_np.shape[1]):
                row[f"prob_class_{cls_idx}"] = float(probs_np[i, cls_idx])

            rows.append(row)
            sample_id += 1

    df = pd.DataFrame(rows)
    save_path = os.path.join("../outputs", "test_predictions.csv")
    df.to_csv(save_path, index=False)

    accuracy = df["correct"].mean()
    print(f"Saved predictions to: {save_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(df.head())


if __name__ == "__main__":
    main()