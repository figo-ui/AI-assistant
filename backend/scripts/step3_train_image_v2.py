"""
STEP 3: Image Classification (v2 — CPU-optimized)
Use 28x28 DermaMNIST with an improved CNN (deeper than DermCNN baseline).
Key improvements over baseline DermCNN:
  - Deeper architecture with BatchNorm + Dropout
  - Class-weighted loss (58x imbalance)
  - Data augmentation (flip, rotation)
  - Cosine annealing LR scheduler
  - Early stopping
Target: beat Test F1=0.307, Test Acc=0.5022
"""
import sys, os, warnings, json, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, "data", "dataset_v1.0", "imaging")
MODEL_DIR = os.path.join(BASE, "backend", "models")

DIVIDER = "=" * 70
def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load metadata ──────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "imaging_metadata.json")) as f:
    meta = json.load(f)

class_names = meta["class_names"]
class_weights_list = meta["class_weights"]
norm_mean = meta["normalization"]["mean"]
norm_std  = meta["normalization"]["std"]
NUM_CLASSES = 7

# ── Load 28x28 data directly from NPZ ─────────────────────────────────────
section("Loading 28x28 DermaMNIST Data")
npz_path = os.path.join(BASE, "data", "raw", "imaging_legacy", "dermamnist.npz")
data = np.load(npz_path)
train_images = data["train_images"]  # (7007, 28, 28, 3) uint8
train_labels = data["train_labels"].flatten()
val_images   = data["val_images"]
val_labels   = data["val_labels"].flatten()
test_images  = data["test_images"]
test_labels  = data["test_labels"].flatten()

print(f"Train: {train_images.shape}, Val: {val_images.shape}, Test: {test_images.shape}")

# ── Dataset ────────────────────────────────────────────────────────────────
class DermDataset28(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (28, 28, 3)
        # Convert to tensor (C, H, W)
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, 28, 28)
        # Normalize
        mean = torch.tensor(norm_mean, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(norm_std,  dtype=torch.float32).view(3, 1, 1)
        img_t = (img_t - mean) / (std + 1e-6)
        # Augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_t = torch.flip(img_t, dims=[2])  # horizontal flip
            if torch.rand(1).item() > 0.5:
                img_t = torch.flip(img_t, dims=[1])  # vertical flip
            # Random rotation (90 degree increments)
            k = torch.randint(0, 4, (1,)).item()
            img_t = torch.rot90(img_t, k, dims=[1, 2])
        return img_t, self.labels[idx]

train_dataset = DermDataset28(train_images, train_labels, augment=True)
val_dataset   = DermDataset28(val_images,   val_labels,   augment=False)
test_dataset  = DermDataset28(test_images,  test_labels,  augment=False)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ── Improved CNN Architecture ──────────────────────────────────────────────
class ImprovedDermCNN(nn.Module):
    """
    Improved CNN for 28x28 DermaMNIST.
    Deeper than baseline DermCNN with BatchNorm + Dropout.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = ImprovedDermCNN(num_classes=NUM_CLASSES).to(DEVICE)
print(f"ImprovedDermCNN params: {sum(p.numel() for p in model.parameters()):,}")

# ── Class weights ──────────────────────────────────────────────────────────
class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ── Training ───────────────────────────────────────────────────────────────
section("Training ImprovedDermCNN (30 epochs)")

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

def evaluate(loader, model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

best_val_f1 = 0.0
best_state = None
PATIENCE = 8
no_improve = 0

for epoch in range(1, 31):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    preds_val, labels_val = evaluate(val_loader, model)
    val_f1  = f1_score(labels_val, preds_val, average="macro")
    val_acc = accuracy_score(labels_val, preds_val)
    print(f"  Epoch {epoch:2d}/30 | Loss: {total_loss/len(train_loader):.4f} | "
          f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
        print(f"    ✓ New best Val F1: {best_val_f1:.4f}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

# ── Final evaluation ───────────────────────────────────────────────────────
section("Final Evaluation")
model.load_state_dict(best_state)

preds_train, labels_train = evaluate(train_loader, model)
preds_val,   labels_val   = evaluate(val_loader,   model)
preds_test,  labels_test  = evaluate(test_loader,  model)

train_f1  = f1_score(labels_train, preds_train, average="macro")
val_f1    = f1_score(labels_val,   preds_val,   average="macro")
test_f1   = f1_score(labels_test,  preds_test,  average="macro")
train_acc = accuracy_score(labels_train, preds_train)
val_acc   = accuracy_score(labels_val,   preds_val)
test_acc  = accuracy_score(labels_test,  preds_test)

print(f"Train — F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
print(f"Val   — F1: {val_f1:.4f},   Acc: {val_acc:.4f}")
print(f"Test  — F1: {test_f1:.4f},  Acc: {test_acc:.4f}")
print(f"Overfitting gap (train-val F1): {train_f1 - val_f1:.4f}")
print(f"\nBaseline: Test F1 = 0.307, Test Acc = 0.5022")
print(f"Improvement: F1 {test_f1 - 0.307:+.4f}, Acc {test_acc - 0.5022:+.4f}")

print(f"\nPer-class report (test set):")
print(classification_report(labels_test, preds_test, target_names=class_names))

# ── Save ───────────────────────────────────────────────────────────────────
section("Saving Image Model")

checkpoint = {
    "model_state_dict": best_state,
    "architecture": "improved_dermcnn",
    "num_classes": NUM_CLASSES,
    "class_names": class_names,
    "input_size": 28,
    "normalization": {"mean": norm_mean, "std": norm_std},
    "val_macro_f1": round(float(val_f1), 4),
    "test_macro_f1": round(float(test_f1), 4),
}
torch.save(checkpoint, os.path.join(MODEL_DIR, "skin_cnn_torch.pt"))
print(f"Saved: skin_cnn_torch.pt")

with open(os.path.join(MODEL_DIR, "image_labels.json"), "w") as f:
    json.dump(class_names, f, indent=2)

metrics = {
    "dataset": "dermamnist_28",
    "architecture": "improved_dermcnn",
    "train_samples": int(len(train_labels)),
    "val_samples": int(len(val_labels)),
    "test_samples": int(len(test_labels)),
    "classes": NUM_CLASSES,
    "train_macro_f1": round(float(train_f1), 4),
    "best_val_macro_f1": round(float(val_f1), 4),
    "test_macro_f1": round(float(test_f1), 4),
    "train_accuracy": round(float(train_acc), 4),
    "val_accuracy": round(float(val_acc), 4),
    "test_accuracy": round(float(test_acc), 4),
    "image_input_size": 28,
    "baseline_test_f1": 0.307,
    "baseline_test_acc": 0.5022,
    "improvement_f1": round(float(test_f1) - 0.307, 4),
    "improvement_acc": round(float(test_acc) - 0.5022, 4),
}
with open(os.path.join(MODEL_DIR, "image_training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: image_training_metrics.json")

print(f"\n✓ Step 3 (Image v2) complete.")
print(f"  ImprovedDermCNN | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")
