"""
TIMO — Fitzpatrick17k Fast Training Script (Feature Extraction + Classifier)
=============================================================================
Strategy for CPU-only training (production-quality, fast):

  Phase 1 — Feature Extraction (one-time, ~5 min):
    - Run all images through frozen EfficientNet-B3 backbone ONCE
    - Save feature vectors (1536-dim) to disk
    - This avoids re-running the backbone every epoch

  Phase 2 — Classifier Training (fast, ~2 min for 15 epochs):
    - Train a 2-layer MLP on the pre-extracted features
    - Class-weighted loss, dropout, cosine LR schedule
    - This is equivalent to fine-tuning the head only, but 100x faster

  Phase 3 — Checkpoint Assembly:
    - Reassemble full EfficientNet-B3 + trained classifier
    - Save in exact format expected by image_model.py
    - class_names, architecture, normalization, model_state_dict

This produces a valid, production-ready checkpoint that image_model.py
can load directly without any changes.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, f1_score

WORKSPACE = Path(r"C:\Users\hp\Desktop\AI assistant")
BACKEND_DIR = WORKSPACE / "backend"
MANIFEST_PATH = BACKEND_DIR / "data/image_dataset_combined/manifest.jsonl"
OUT_DIR = BACKEND_DIR / "models"
FEATURES_DIR = WORKSPACE / "data/processed/fitzpatrick_features"

ARCHITECTURE = "efficientnet_b3"
EPOCHS = 30          # Fast since we're only training MLP on features
BATCH_SIZE = 256     # Large batch fine for MLP
LR = 1e-3
IMAGE_SIZE = 224
SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def load_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({
                "image_path": str(d["image_path"]),
                "label": str(d["label"]).strip(),
                "split": str(d.get("split", "train")).strip().lower(),
            })
    return pd.DataFrame(rows)


def extract_features(df: pd.DataFrame, model, transform, device, split_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract EfficientNet-B3 features for all images in a split."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ImgDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                return self.transform(img), row["label"]
            except Exception:
                # Return black image on error
                return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), row["label"]

    ds = ImgDataset(df, transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_feats = []
    all_labels = []
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (xb, labels) in enumerate(loader):
            feats = model(xb.to(device)).cpu().numpy()
            all_feats.append(feats)
            all_labels.extend(list(labels))
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{split_name}] batch {i+1}/{len(loader)} | {elapsed:.0f}s elapsed")
                sys.stdout.flush()

    features = np.vstack(all_feats)
    labels_arr = np.array(all_labels)
    elapsed = time.time() - t0
    print(f"  [{split_name}] Done: {features.shape} features in {elapsed:.0f}s")
    return features, labels_arr


def main():
    seed_everything(SEED)

    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Loading manifest: {MANIFEST_PATH}")

    df = load_manifest(MANIFEST_PATH)
    classes = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes: {num_classes}")
    print(f"Split counts: {df['split'].value_counts().to_dict()}")

    # ── Phase 1: Build feature extractor ──────────────────────────────────────
    print("\n[Phase 1] Building EfficientNet-B3 feature extractor...")
    weights = models.EfficientNet_B3_Weights.DEFAULT
    full_model = models.efficientnet_b3(weights=weights)

    # Feature extractor = everything except the final classifier
    feature_extractor = nn.Sequential(
        full_model.features,
        full_model.avgpool,
        nn.Flatten(),
    )
    feature_extractor.eval()
    feature_extractor.to(device)

    # Get feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        feat_dim = feature_extractor(dummy).shape[1]
    print(f"Feature dimension: {feat_dim}")

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # ── Phase 1b: Extract or load cached features ──────────────────────────────
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    features_cache = {}

    for split in splits:
        cache_path = FEATURES_DIR / f"{split}_features.npz"
        split_df = df[df["split"] == split].copy()

        if cache_path.exists():
            print(f"  Loading cached features for {split}...")
            data = np.load(cache_path, allow_pickle=True)
            features_cache[split] = {
                "X": data["X"],
                "y": data["y"],
                "labels": data["labels"].tolist(),
            }
            print(f"  Loaded: {features_cache[split]['X'].shape}")
        else:
            print(f"  Extracting features for {split} ({len(split_df)} images)...")
            X, y_labels = extract_features(split_df, feature_extractor, eval_transform, device, split)
            y = np.array([class_to_idx[l] for l in y_labels])
            np.savez_compressed(cache_path, X=X, y=y, labels=np.array(y_labels))
            features_cache[split] = {"X": X, "y": y, "labels": y_labels}
            print(f"  Saved cache: {cache_path}")

    X_train = features_cache["train"]["X"]
    y_train = features_cache["train"]["y"]
    X_val   = features_cache["val"]["X"]
    y_val   = features_cache["val"]["y"]
    X_test  = features_cache["test"]["X"]
    y_test  = features_cache["test"]["y"]

    print(f"\nFeature shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ── Phase 2: Train MLP classifier on features ─────────────────────────────
    print(f"\n[Phase 2] Training MLP classifier for {EPOCHS} epochs...")

    # Class weights
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    weight_tensor = torch.tensor(w, dtype=torch.float32)

    # MLP: 2 hidden layers with dropout
    # Note: features are already flat (1536-dim), no Flatten needed
    classifier = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    classifier.to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def evaluate_mlp(X, y):
        classifier.eval()
        with torch.no_grad():
            logits = classifier(X)
            preds = torch.argmax(logits, dim=1).numpy()
        y_np = y.numpy()
        acc = float(np.mean(preds == y_np))
        f1  = float(f1_score(y_np, preds, average="macro", zero_division=0))
        return acc, f1, y_np.tolist(), preds.tolist()

    best_val_f1 = -1.0
    best_state  = None
    best_epoch  = -1
    epoch_log   = []

    print("-" * 70)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        classifier.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.shape[0]
        scheduler.step()

        train_loss = running_loss / max(1, len(X_tr))
        val_acc, val_f1, _, _ = evaluate_mlp(X_v, y_v)
        elapsed = time.time() - t0

        print(
            f"epoch={epoch:02d} | loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | "
            f"time={elapsed:.2f}s"
        )
        sys.stdout.flush()

        epoch_log.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            best_state  = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            print(f"  ✓ New best val_f1={best_val_f1:.4f} at epoch {best_epoch}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nLoading best classifier checkpoint...")
    classifier.load_state_dict(best_state)
    test_acc, test_f1, y_true, y_pred = evaluate_mlp(X_te, y_te)
    train_acc, train_f1, _, _ = evaluate_mlp(X_tr, y_tr)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Train  acc={train_acc:.4f}  macro-F1={train_f1:.4f}")
    print(f"  Val    macro-F1={best_val_f1:.4f}  (best epoch={best_epoch})")
    print(f"  Test   acc={test_acc:.4f}  macro-F1={test_f1:.4f}")
    print(f"  Baseline (DermaMNIST 7-class): test macro-F1=0.449, test acc=0.5796")
    delta_f1 = test_f1 - 0.449
    print(f"  Delta vs baseline: F1 {delta_f1:+.4f}")
    print("=" * 70)

    # ── Phase 3: Assemble full model checkpoint ────────────────────────────────
    print("\n[Phase 3] Assembling full EfficientNet-B3 + trained classifier...")

    # Rebuild full model with the SAME classifier architecture as the standalone MLP
    # The standalone MLP (classifier) has these layers:
    #   0: Linear(feat_dim→512)
    #   1: BatchNorm1d(512)
    #   2: ReLU
    #   3: Dropout(0.4)
    #   4: Linear(512→256)
    #   5: BatchNorm1d(256)
    #   6: ReLU
    #   7: Dropout(0.3)
    #   8: Linear(256→num_classes)
    # EfficientNet's avgpool already flattens, so we don't need Flatten() here.
    full_model_final = models.efficientnet_b3(weights=weights)
    full_model_final.classifier = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    # Load trained classifier weights (keys match exactly now)
    full_model_final.classifier.load_state_dict(best_state)
    full_model_final.eval()

    # Verify the assembled model produces same predictions as feature extractor + MLP
    print("Verifying assembled model...")
    with torch.no_grad():
        # Test on a small batch
        test_imgs = []
        from torchvision import transforms as T
        transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
        test_df_sample = df[df["split"] == "test"].head(4)
        for _, row in test_df_sample.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                test_imgs.append(transform(img))
            except Exception:
                test_imgs.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
        if test_imgs:
            batch = torch.stack(test_imgs)
            # Full model prediction
            full_preds = torch.argmax(full_model_final(batch), dim=1).numpy()
            # Feature extractor + MLP prediction
            feats = feature_extractor(batch)
            mlp_preds = torch.argmax(classifier(feats), dim=1).numpy()
            match = np.mean(full_preds == mlp_preds)
            print(f"  Prediction match (full vs feature+MLP): {match:.0%}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "backend": "torchvision",
        "architecture": ARCHITECTURE,
        "model_state_dict": {k: v.detach().cpu() for k, v in full_model_final.state_dict().items()},
        "num_classes": num_classes,
        "class_names": classes,
        "input_size": IMAGE_SIZE,
        "normalization": {"mean": MEAN, "std": STD},
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "train_macro_f1": round(train_f1, 4),
        "train_accuracy": round(train_acc, 4),
        "training_strategy": "feature_extraction_mlp",
        "dataset": "fitzpatrick17k_atlas_subset",
        "epoch_log": epoch_log,
    }
    torch.save(checkpoint, OUT_DIR / "skin_cnn_torch.pt")
    print(f"Checkpoint saved: {OUT_DIR / 'skin_cnn_torch.pt'}")

    # ── Save labels ───────────────────────────────────────────────────────────
    (OUT_DIR / "image_labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    # ── Save training metrics ─────────────────────────────────────────────────
    metrics = {
        "dataset": "fitzpatrick17k_atlas_subset",
        "architecture": ARCHITECTURE,
        "training_strategy": "feature_extraction_mlp",
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "classes": num_classes,
        "train_macro_f1": round(train_f1, 4),
        "train_accuracy": round(train_acc, 4),
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "best_epoch": best_epoch,
        "epochs_trained": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "baseline_test_f1": 0.449,
        "baseline_test_acc": 0.5796,
        "improvement_f1": round(test_f1 - 0.449, 4),
        "improvement_acc": round(test_acc - 0.5796, 4),
        "epoch_log": epoch_log,
        "classification_report": report,
    }
    (OUT_DIR / "image_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ── Save metadata ─────────────────────────────────────────────────────────
    metadata = {
        "backend": "torchvision",
        "architecture": ARCHITECTURE,
        "input_size": IMAGE_SIZE,
        "classes": classes,
        "normalization": {"mean": MEAN, "std": STD},
        "num_classes": num_classes,
        "dataset": "fitzpatrick17k_atlas_subset",
        "training_strategy": "feature_extraction_mlp",
    }
    (OUT_DIR / "image_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nAll artifacts saved to: {OUT_DIR}")
    print("✅ Training complete!")
    return metrics


if __name__ == "__main__":
    main()
