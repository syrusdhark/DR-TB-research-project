"""
Train Stage 2: DRTBRiskModel on clinical + genomic data only (no image).

Data: data/drtb_risk_dataset.csv (built by prepare_stage2_data.py), 4,200
unique patients, label_drtb derived deterministically from
mdr_tb/xdr_tb/rifampin_resistance/isoniazid_resistance/mutation_count.

Because the label is a deterministic function of a subset of the inputs,
expect this model to reach very high validation AUROC/F1 -- that reflects
the label's construction, not a claim of some deeper learned signal beyond
that rule. What this model actually adds over the raw rule is generalizing
across which specific genomic mutations are present and blending that with
the clinical variables that aren't part of the label formula (age, HIV,
diabetes, smoking, region, etc.) via the attention fusion.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config
from model import DRTBRiskModel

DATA_CSV = REPO_ROOT / "data" / "drtb_risk_dataset.csv"
MODELS_DIR = REPO_ROOT / "results" / "models"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class TabularDataset(Dataset):
    def __init__(self, clinical, genomic, labels):
        self.clinical = torch.tensor(clinical, dtype=torch.float32)
        self.genomic = torch.tensor(genomic, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.clinical[idx], self.genomic[idx], self.labels[idx]


def load_dataset():
    df = pd.read_csv(DATA_CSV)
    clinical = df[config.CLINICAL_FEATURES].values.astype(np.float32)
    genomic = df[config.GENOMIC_FEATURES].values.astype(np.float32)
    labels = df["label_drtb"].values.astype(np.float32)
    return clinical, genomic, labels


def make_loader(clinical, genomic, labels, batch_size, shuffle, sampler=None):
    dataset = TabularDataset(clinical, genomic, labels)
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for clinical, genomic, labels in loader:
            clinical, genomic = clinical.to(device), genomic.to(device)
            output, _ = model(clinical, genomic)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def best_threshold(probs, labels):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main():
    clinical, genomic, labels = load_dataset()
    print(f"Loaded {len(labels)} patients. Positive rate: {labels.mean():.3f}")

    c_train, c_temp, g_train, g_temp, y_train, y_temp = train_test_split(
        clinical, genomic, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    c_val, c_test, g_val, g_test, y_val, y_test = train_test_split(
        c_temp, g_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    class_counts = np.bincount(y_train.astype(int))
    sample_weights = 1.0 / class_counts[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = make_loader(c_train, g_train, y_train, batch_size=32, shuffle=False, sampler=sampler)
    val_loader = make_loader(c_val, g_val, y_val, batch_size=64, shuffle=False)
    test_loader = make_loader(c_test, g_test, y_test, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRTBRiskModel(
        num_clinical_features=config.NUM_CLINICAL_FEATURES,
        num_genomic_features=config.NUM_GENOMIC_FEATURES,
        num_classes=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1
    best_state = None
    epochs = 60
    patience, patience_counter = 15, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for clinical_batch, genomic_batch, label_batch in train_loader:
            clinical_batch = clinical_batch.to(device)
            genomic_batch = genomic_batch.to(device)
            label_batch = label_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output, _ = model(clinical_batch, genomic_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(label_batch)

        val_probs, val_labels = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={total_loss/len(train_loader.dataset):.4f} | val_auc={val_auc:.4f}")

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
        "note": (
            "label_drtb is derived deterministically from mdr_tb/xdr_tb/"
            "rifampin_resistance/isoniazid_resistance/mutation_count (see "
            "prepare_stage2_data.py). High metrics reflect that construction, "
            "not a claim of predictive power beyond it."
        ),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{config.DRTB_RISK_MODEL_PREFIX}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_clinical_features": config.NUM_CLINICAL_FEATURES,
        "num_genomic_features": config.NUM_GENOMIC_FEATURES,
        "validation_auc": best_val_auc,
        "validation_f1": val_f1,
    }, checkpoint_path)

    metrics_path = MODELS_DIR / f"{config.DRTB_RISK_MODEL_PREFIX}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
