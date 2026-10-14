"""Training loop for pneumonia classifier."""
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from .dataset import ChestXRayDataset, get_train_transforms, get_val_transforms
from .model import get_model


# ---------- Config ----------
DATA_DIR = Path("data/chest_xray")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # Windows pe 0 rakho, warna issues aate hain
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
    """Handle class imbalance — Normal aur Pneumonia ko barabar weight do."""
    labels = [label for _, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label].item() for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders():
    """Build train/val/test dataloaders."""
    train_ds = ChestXRayDataset(
        DATA_DIR / "train",
        transform=get_train_transforms(IMAGE_SIZE),
    )
    val_ds = ChestXRayDataset(
        DATA_DIR / "val",
        transform=get_val_transforms(IMAGE_SIZE),
    )
    test_ds = ChestXRayDataset(
        DATA_DIR / "test",
        transform=get_val_transforms(IMAGE_SIZE),
    )

    train_sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Train class dist: {train_ds.get_class_distribution()}")

    return train_loader, val_loader, test_loader


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_true, all_pred, all_prob = [], [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        all_true.extend(labels.cpu().numpy().flatten().astype(int))
        all_pred.extend(preds)
        all_prob.extend(probs)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_true, all_pred, all_prob)
    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    running_loss = 0.0
    all_true, all_pred, all_prob = [], [], []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        all_true.extend(labels.cpu().numpy().flatten().astype(int))
        all_pred.extend(preds)
        all_prob.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_true, all_pred, all_prob)
    return epoch_loss, metrics


def train():
    """Full training pipeline."""
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders()

    model = get_model(pretrained=True, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_loss = float("inf")
    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")

        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Time: {elapsed:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | AUC: {val_metrics['auc']:.4f}")

        # Save best model (based on val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_metrics["auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }, MODEL_DIR / "best_model.pth")
            print(f"Saved best model (val_loss={val_loss:.4f}, val_auc={val_metrics['auc']:.4f})")

    # Final test evaluation
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    checkpoint = torch.load(MODEL_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1:        {test_metrics['f1']:.4f}")
    print(f"AUC:       {test_metrics['auc']:.4f}")


if __name__ == "__main__":
    train()