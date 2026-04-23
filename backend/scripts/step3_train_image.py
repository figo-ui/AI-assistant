"""
STEP 3: Model Training — Task 3: Skin Lesion Image Classification
Strategy:
  - EfficientNet-B0 with pretrained ImageNet weights (transfer learning)
  - Input: 64x64 dermamnist images (upsampled to 224x224 for EfficientNet)
  - Class-weighted loss to handle 58x imbalance
  - Data augmentation: random flip, rotation, color jitter
  - Target: beat Test Macro F1 = 0.307, Test Accuracy = 0.5022
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
import torchvision.models as tv_models
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

print(f"Classes: {NUM_CLASSES}")
print(f"Class weights: {[round(w, 3) for w in class_weights_list]}")

# ── Load data ──────────────────────────────────────────────────────────────
section("Loading Image Data")
train_images = np.load(os.path.join(DATA_DIR, "train_images.npy"))  # (7007, 64, 64, 3) float32 [0,1]
train_labels = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
val_images   = np.load(os.path.join(DATA_DIR, "val_images.npy"))
val_labels   = np.load(os.path.join(DATA_DIR, "val_labels.npy"))
test_images  = np.load(os.path.join(DATA_DIR, "test_images.npy"))
test_labels  = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

print(f"Train: {train_images.shape}, Val: {val_images.shape}, Test: {test_images.shape}")

# ── Dataset class ──────────────────────────────────────────────────────────
class DermDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # images: (N, H, W, C) float32 [0,1]
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, C) float32
        # Convert to uint8 PIL for transforms
        img_uint8 = (img * 255).astype(np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(img_uint8)
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)
        return img_tensor, self.labels[idx]

# ── Transforms ────────────────────────────────────────────────────────────
# EfficientNet-B0 expects 224x224
TARGET_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])

val_transform = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])

train_dataset = DermDataset(train_images, train_labels, transform=train_transform)
val_dataset   = DermDataset(val_images,   val_labels,   transform=val_transform)
test_dataset  = DermDataset(test_images,  test_labels,  transform=val_transform)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ── Build EfficientNet-B0 ──────────────────────────────────────────────────
section("Building EfficientNet-B0 (Transfer Learning)")

model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# Replace classifier head
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, NUM_CLASSES),
)
model = model.to(DEVICE)
print(f"EfficientNet-B0 loaded. Classifier in_features: {in_features}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Class weights ──────────────────────────────────────────────────────────
class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ── Phase 1: Train only the classifier head (5 epochs) ────────────────────
section("Phase 1: Train Classifier Head Only (5 epochs)")

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

optimizer_p1 = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=5)

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

for epoch in range(1, 6):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer_p1.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer_p1.step()
        total_loss += loss.item()
    scheduler_p1.step()

    preds_val, labels_val = evaluate(val_loader, model)
    val_f1  = f1_score(labels_val, preds_val, average="macro")
    val_acc = accuracy_score(labels_val, preds_val)
    print(f"  Epoch {epoch}/5 | Loss: {total_loss/len(train_loader):.4f} | "
          f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"Phase 1 best Val F1: {best_val_f1:.4f}")

# ── Phase 2: Fine-tune full model (10 epochs) ──────────────────────────────
section("Phase 2: Fine-tune Full Model (10 epochs)")

# Unfreeze all
for param in model.parameters():
    param.requires_grad = True

optimizer_p2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=10, eta_min=1e-6)

PATIENCE = 4
no_improve = 0

for epoch in range(1, 11):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer_p2.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_p2.step()
        total_loss += loss.item()
    scheduler_p2.step()

    preds_val, labels_val = evaluate(val_loader, model)
    val_f1  = f1_score(labels_val, preds_val, average="macro")
    val_acc = accuracy_score(labels_val, preds_val)
    print(f"  Epoch {epoch}/10 | Loss: {total_loss/len(train_loader):.4f} | "
          f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
        print(f"    ✓ New best Val F1: {best_val_f1:.4f}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# ── Evaluate best model ────────────────────────────────────────────────────
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

# ── Save model ─────────────────────────────────────────────────────────────
section("Saving Image Model")

checkpoint = {
    "model_state_dict": best_state,
    "architecture": "efficientnet_b0",
    "num_classes": NUM_CLASSES,
    "class_names": class_names,
    "input_size": TARGET_SIZE,
    "normalization": {"mean": norm_mean, "std": norm_std},
    "val_macro_f1": round(float(val_f1), 4),
    "test_macro_f1": round(float(test_f1), 4),
}
torch.save(checkpoint, os.path.join(MODEL_DIR, "skin_cnn_torch.pt"))
print(f"Saved: skin_cnn_torch.pt")

with open(os.path.join(MODEL_DIR, "image_labels.json"), "w") as f:
    json.dump(class_names, f, indent=2)
print(f"Saved: image_labels.json")

metrics = {
    "dataset": "dermamnist_64",
    "architecture": "efficientnet_b0",
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
    "image_input_size": TARGET_SIZE,
    "baseline_test_f1": 0.307,
    "baseline_test_acc": 0.5022,
    "improvement_f1": round(float(test_f1) - 0.307, 4),
    "improvement_acc": round(float(test_acc) - 0.5022, 4),
}
with open(os.path.join(MODEL_DIR, "image_training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved: image_training_metrics.json")

print(f"\n✓ Step 3 (Image) complete.")
print(f"  EfficientNet-B0 | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")
