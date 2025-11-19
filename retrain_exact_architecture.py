"""
Retrain the DR-TB multimodal model using the exact architecture defined in model.py.
This script loads the merged dataset, applies heavy class weighting + Focal+Dice loss,
and saves a strictly-compatible checkpoint at results/models/exact_match_nov2025.pth.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

# Project setup
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from model import MultimodalFusionModel

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = config.IMG_SIZE
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 35
EARLY_STOP_PATIENCE = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
RANDOM_SEED = 42
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "models" / "exact_match_nov2025.pth"
METRICS_PATH = PROJECT_ROOT / "results" / "models" / "exact_match_nov2025_metrics.json"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset
# -----------------------------


class MultimodalDRTBDataset(Dataset):
    def __init__(self, dataframe, transform, image_root, clinical_stats, genomic_stats):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_root = Path(image_root)
        self.clinical_cols = config.CLINICAL_FEATURES
        self.genomic_cols = config.GENOMIC_FEATURES

        self.clinical_mean = torch.tensor(clinical_stats["mean"], dtype=torch.float32)
        self.clinical_std = torch.tensor(clinical_stats["std"], dtype=torch.float32)
        self.genomic_mean = torch.tensor(genomic_stats["mean"], dtype=torch.float32)
        self.genomic_std = torch.tensor(genomic_stats["std"], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_path):
        full_path = self.image_root / img_path
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_image(row["img_path"])

        clinical = torch.tensor(
            row[self.clinical_cols].values.astype(np.float32), dtype=torch.float32
        )
        clinical = (clinical - self.clinical_mean) / self.clinical_std

        genomic = torch.tensor(
            row[self.genomic_cols].values.astype(np.float32), dtype=torch.float32
        )
        genomic = (genomic - self.genomic_mean) / self.genomic_std

        label = torch.tensor(row["label_drtb"], dtype=torch.float32)

        return image, clinical, genomic, label


# -----------------------------
# Transforms
# -----------------------------

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
        ),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
)

# -----------------------------
# Loss Functions
# -----------------------------


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets = targets.view_as(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        targets = targets.view_as(logits)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_score = (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice_score


class CombinedLoss(nn.Module):
    def __init__(self, focal_alpha, focal_gamma, pos_weight, focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


# -----------------------------
# Utility Functions
# -----------------------------


def compute_feature_stats(df, columns):
    values = df[columns].values.astype(np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std}


def build_dataloaders(df):
    labels = df["label_drtb"].values
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label_drtb"],
        random_state=RANDOM_SEED,
    )

    clinical_stats = compute_feature_stats(train_df, config.CLINICAL_FEATURES)
    genomic_stats = compute_feature_stats(train_df, config.GENOMIC_FEATURES)

    train_dataset = MultimodalDRTBDataset(
        train_df, train_transform, PROJECT_ROOT, clinical_stats, genomic_stats
    )
    val_dataset = MultimodalDRTBDataset(
        val_df, eval_transform, PROJECT_ROOT, clinical_stats, genomic_stats
    )
    test_dataset = MultimodalDRTBDataset(
        test_df, eval_transform, PROJECT_ROOT, clinical_stats, genomic_stats
    )

    class_counts = train_df["label_drtb"].value_counts()
    neg = class_counts.get(0, 1)
    pos = class_counts.get(1, 1)
    weight_for_class = {0: 1.0, 1: neg / pos}
    sample_weights = train_df["label_drtb"].map(weight_for_class).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    stats = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "pos_weight": torch.tensor(float(neg / pos), device=DEVICE),
    }
    return train_loader, val_loader, test_loader, stats


def evaluate(model, loader, threshold=0.5):
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for images, clinical, genomic, labels in loader:
            images = images.to(DEVICE)
            clinical = clinical.to(DEVICE)
            genomic = genomic.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs, _ = model(images, clinical, genomic)
            batch_probs = torch.sigmoid(outputs).view(-1)

            probs.extend(batch_probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    probs = np.array(probs)
    targets = np.array(targets)

    if len(np.unique(targets)) > 1:
        auc = roc_auc_score(targets, probs)
    else:
        auc = 0.5

    best_threshold = 0.5
    best_f1 = 0.0
    for th in np.linspace(0.2, 0.9, 71):
        preds = (probs >= th).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    preds = (probs >= threshold).astype(int)
    metrics = {
        "auc": float(auc),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
    }
    return metrics, probs, targets


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (images, clinical, genomic, labels) in enumerate(loader):
        images = images.to(DEVICE)
        clinical = clinical.to(DEVICE)
        genomic = genomic.to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast(enabled=torch.cuda.is_available()):
            logits, _ = model(images, clinical, genomic)
            logits = logits.view(-1)
            loss = criterion(logits, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * GRAD_ACCUM_STEPS

    return running_loss / len(loader)


# -----------------------------
# Main Training Logic
# -----------------------------


def main():
    merged_path = PROJECT_ROOT / "data" / "merged_dataset.csv"
    if not merged_path.exists():
        print(f"❌ Dataset not found at {merged_path}.")
        sys.exit(1)

    df = pd.read_csv(merged_path)
    if "label_drtb" not in df.columns or "img_path" not in df.columns:
        print("❌ merged_dataset.csv missing required columns.")
        sys.exit(1)

    train_loader, val_loader, test_loader, stats = build_dataloaders(df)
    print(
        f"📊 Dataset split → Train: {stats['train_size']}, "
        f"Val: {stats['val_size']}, Test: {stats['test_size']}"
    )

    model = MultimodalFusionModel(
        num_clinical_features=len(config.CLINICAL_FEATURES),
        num_genomic_features=len(config.GENOMIC_FEATURES),
        num_classes=1,
    ).to(DEVICE)

    criterion = CombinedLoss(
        focal_alpha=0.75,
        focal_gamma=2.5,
        pos_weight=stats["pos_weight"],
        focal_weight=0.7,
        dice_weight=0.3,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_auc = 0.0
    best_metrics = None
    patience_counter = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_metrics, _, _ = evaluate(model, val_loader, threshold=0.5)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_metrics["auc"],
                "val_f1": val_metrics["f1"],
            }
        )

        print(
            f"[Epoch {epoch}/{NUM_EPOCHS}] "
            f"Loss={train_loss:.4f} | Val AUC={val_metrics['auc']:.4f} | "
            f"Val F1={val_metrics['f1']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = val_metrics
            patience_counter = 0
            save_checkpoint(model, best_metrics)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    if best_metrics is None:
        print("❌ Training failed to improve. No checkpoint saved.")
        sys.exit(1)

    # Load best checkpoint for testing
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state_dict"])
    test_metrics, _, _ = evaluate(model, test_loader, threshold=best_metrics["best_threshold"])

    save_metrics(history, best_metrics, test_metrics)
    print("✅ Training complete.")
    print(f"   • Best Val AUC: {best_metrics['auc']:.4f}")
    print(f"   • Test AUC: {test_metrics['auc']:.4f}")
    print(f"   • Checkpoint saved to: {CHECKPOINT_PATH}")


def save_checkpoint(model, val_metrics):
    os.makedirs(CHECKPOINT_PATH.parent, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_clinical_features": len(config.CLINICAL_FEATURES),
        "num_genomic_features": len(config.GENOMIC_FEATURES),
        "validation_auc": val_metrics["auc"],
        "validation_f1": val_metrics["f1"],
        "validation_precision": val_metrics["precision"],
        "validation_recall": val_metrics["recall"],
        "threshold": val_metrics["best_threshold"],
        "saved_at": datetime.utcnow().isoformat(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"💾 Saved new best checkpoint with AUC {val_metrics['auc']:.4f}")


def save_metrics(history, val_metrics, test_metrics):
    metrics = {
        "history": history,
        "best_val": val_metrics,
        "test": test_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"📁 Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()

