"""
Train Stage 1: TBImageClassifier on real chest X-rays only.

Data: data/tb_image_manifest.csv (built by prepare_stage1_data.py) -- each of
the 4,200 real images in TB_Chest_Radiography_Database used exactly once,
labeled directly from its source folder. No synthetic duplication, no
clinical/genomic data involved.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import config
from model import TBImageClassifier

BASE_DIR = Path(__file__).parent
MANIFEST_CSV = BASE_DIR / "data" / "tb_image_manifest.csv"
MODELS_DIR = BASE_DIR / "results" / "models"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)


class CXRDataset(Dataset):
    def __init__(self, img_paths, labels, train=False):
        self.img_paths = img_paths
        self.labels = labels
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(BASE_DIR / self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def best_threshold(probs, labels):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main():
    df = pd.read_csv(MANIFEST_CSV)
    img_paths = df["img_path"].values
    labels = df["label_tb"].values.astype(np.float32)
    print(f"Loaded {len(labels)} images. TB positive rate: {labels.mean():.3f}")

    p_train, p_temp, y_train, y_temp = train_test_split(
        img_paths, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    train_ds = CXRDataset(p_train, y_train, train=True)
    val_ds = CXRDataset(p_val, y_val, train=False)
    test_ds = CXRDataset(p_test, y_test, train=False)

    class_counts = np.bincount(y_train.astype(int))
    sample_weights = 1.0 / class_counts[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TBImageClassifier(num_classes=1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1
    best_state = None
    epochs = 25
    patience, patience_counter = 7, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for images, batch_labels in train_loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)

        val_probs, val_labels = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            # Save after every improvement so an interrupted run still leaves
            # a usable checkpoint behind instead of losing all progress.
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": best_state,
                "validation_auc": best_val_auc,
                "epoch": epoch,
            }, MODELS_DIR / f"{config.TB_MODEL_PREFIX}.pth")
        else:
            patience_counter += 1

        dt = time.time() - t0
        print(f"Epoch {epoch:3d} | train_loss={total_loss/len(train_loader.dataset):.4f} | "
              f"val_auc={val_auc:.4f} | best_val_auc={best_val_auc:.4f} | {dt:.1f}s", flush=True)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no val AUC improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)

    val_probs, val_labels = evaluate(model, val_loader, device)
    threshold, val_f1 = best_threshold(val_probs, val_labels)

    test_probs, test_labels = evaluate(model, test_loader, device)
    test_preds = (test_probs >= threshold).astype(int)

    metrics = {
        "optimal_threshold": float(threshold),
        "validation_auc": float(best_val_auc),
        "validation_f1": float(val_f1),
        "test_auc": float(roc_auc_score(test_labels, test_probs)),
        "test_accuracy": float(accuracy_score(test_labels, test_preds)),
        "test_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "test_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "test_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "image_size": config.IMG_SIZE,
        "note": (
            "Compact CNN trained from scratch (no ImageNet pretraining -- "
            "download.pytorch.org and huggingface.co are blocked in this "
            "training environment). Each image used exactly once, label "
            "taken directly from its source folder (Tuberculosis/ vs Normal/)."
        ),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{config.TB_MODEL_PREFIX}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "validation_auc": best_val_auc,
        "validation_f1": val_f1,
    }, checkpoint_path)

    metrics_path = MODELS_DIR / f"{config.TB_MODEL_PREFIX}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
