import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, f1_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class ManifestRow:
    image_path: str
    label: str
    split: str
    source: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    backend_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train an upgraded skin/image model on a combined manifest dataset."
    )
    parser.add_argument(
        "--manifest",
        default=str(backend_dir / "data" / "image_dataset_combined" / "manifest.jsonl"),
        help="Combined image manifest produced by download_medical_image_datasets.py",
    )
    parser.add_argument(
        "--out-dir",
        default=str(backend_dir / "models"),
        help="Directory for saving model artifacts.",
    )
    parser.add_argument(
        "--architecture",
        choices=["efficientnet_b3", "resnet50"],
        default="efficientnet_b3",
        help="Primary image backbone.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torchvision", "keras"],
        default="auto",
        help="Prefer torchvision first or force Keras fallback.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest(path: Path) -> List[ManifestRow]:
    if not path.exists():
        raise FileNotFoundError(f"Combined image manifest not found: {path}")

    rows: List[ManifestRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            image_path = str(payload["image_path"])
            label = str(payload["label"]).strip()
            split = str(payload.get("split", "train")).strip().lower()
            source = str(payload.get("source", "unknown")).strip().lower()
            if not image_path or not label:
                continue
            if split not in {"train", "val", "test"}:
                split = "train"
            rows.append(ManifestRow(image_path=image_path, label=label, split=split, source=source))
    if not rows:
        raise RuntimeError(f"Combined image manifest is empty: {path}")
    return rows


def summarize_manifest(rows: Sequence[ManifestRow]) -> Dict[str, object]:
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    class_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    for row in rows:
        split_counts[row.split] = split_counts.get(row.split, 0) + 1
        class_counts[row.label] = class_counts.get(row.label, 0) + 1
        source_counts[row.source] = source_counts.get(row.source, 0) + 1
    return {
        "rows": len(rows),
        "splits": split_counts,
        "classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "sources": dict(sorted(source_counts.items())),
    }


def try_import_torchvision():
    try:
        import torch
        from torchvision import models, transforms

        return torch, models, transforms
    except Exception:
        return None, None, None


def _keras_architecture(architecture: str):
    import tensorflow as tf

    if architecture == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input
    else:
        base = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(300, 300, 3),
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    return tf, base, preprocess


def choose_backend(preference: str) -> str:
    torch, models, transforms = try_import_torchvision()
    if preference == "torchvision":
        if torch is None or models is None or transforms is None:
            raise RuntimeError("Requested torchvision backend, but torchvision is unavailable in this environment.")
        return "torchvision"
    if preference == "keras":
        return "keras"
    if torch is not None and models is not None and transforms is not None:
        return "torchvision"
    return "keras"


def rows_to_frame(rows: Sequence[ManifestRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_path": row.image_path,
                "label": row.label,
                "split": row.split,
                "source": row.source,
            }
            for row in rows
        ]
    )


def train_torchvision(
    *,
    rows: Sequence[ManifestRow],
    architecture: str,
    image_size: int,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    seed: int,
) -> Dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted({row.label for row in rows})
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    def _get_norm(w):
        """Extract mean/std from weights — handles both old and new torchvision APIs."""
        if "mean" in w.meta and "std" in w.meta:
            return list(w.meta["mean"]), list(w.meta["std"])
        # Newer torchvision: normalization lives in transforms()
        try:
            t = w.transforms()
            return list(t.mean), list(t.std)
        except Exception:
            pass
        # ImageNet defaults as fallback
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        mean, std = _get_norm(weights)
        backbone = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, len(classes))
    else:
        weights = models.EfficientNet_B3_Weights.DEFAULT
        mean, std = _get_norm(weights)
        backbone = models.efficientnet_b3(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, len(classes))

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    class ManifestDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform) -> None:
            self.frame = frame.reset_index(drop=True)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            image = Image.open(row["image_path"]).convert("RGB")
            tensor = self.transform(image)
            return tensor, class_to_idx[str(row["label"])]

    frame = rows_to_frame(rows)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Combined image manifest must contain train/val/test rows.")

    train_ds = ManifestDataset(train_df, train_transform)
    val_ds = ManifestDataset(val_df, eval_transform)
    test_ds = ManifestDataset(test_df, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    counts = train_df["label"].value_counts().reindex(classes, fill_value=0).astype(float).to_numpy()
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    model = backbone.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_f1 = -1.0
    best_epoch = -1

    def evaluate(loader: DataLoader) -> Tuple[float, float, List[int], List[int]]:
        model.eval()
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                all_pred.extend(preds)
                all_true.extend(yb.numpy().tolist())
        accuracy = float(np.mean(np.asarray(all_true) == np.asarray(all_pred)))
        macro_f1 = float(f1_score(all_true, all_pred, average="macro"))
        return accuracy, macro_f1, all_true, all_pred

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.shape[0]
        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_acc, val_f1, _, _ = evaluate(val_loader)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Image training failed: no best checkpoint captured.")

    model.load_state_dict(best_state)
    test_acc, test_f1, y_true, y_pred = evaluate(test_loader)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    checkpoint = {
        "backend": "torchvision",
        "architecture": architecture,
        "model_state_dict": best_state,
        "num_classes": len(classes),
        "class_names": classes,
        "input_size": int(image_size),
        "normalization": {"mean": mean, "std": std},
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
    }
    torch.save(checkpoint, out_dir / "skin_cnn_torch.pt")

    return {
        "backend": "torchvision",
        "architecture": architecture,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
        "classification_report": report,
        "input_size": int(image_size),
        "normalization": {"mean": mean, "std": std},
        "classes": classes,
    }


def train_keras(
    *,
    rows: Sequence[ManifestRow],
    architecture: str,
    image_size: int,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, object]:
    tf, base_model, preprocess_input = _keras_architecture(architecture)
    seed_everything(seed)
    tf.random.set_seed(seed)
    classes = sorted({row.label for row in rows})
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    input_size = 300 if architecture == "efficientnet_b3" else image_size
    frame = rows_to_frame(rows)
    if frame.empty:
        raise RuntimeError("Combined image manifest is empty.")

    def load_array(path: str) -> np.ndarray:
        image = Image.open(path).convert("RGB").resize((input_size, input_size))
        return np.asarray(image, dtype=np.float32)

    def encode_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
        subset = frame[frame["split"] == split].copy()
        if subset.empty:
            raise RuntimeError(f"Combined image manifest is missing '{split}' rows.")
        images = np.stack([load_array(path) for path in subset["image_path"].tolist()])
        labels = np.asarray([class_to_idx[label] for label in subset["label"].tolist()], dtype=np.int32)
        return images, labels

    x_train, y_train = encode_split("train")
    x_val, y_val = encode_split("val")
    x_test, y_test = encode_split("test")

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.08),
        ]
    )

    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = augment(inputs)
    x = preprocess_input(x)
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    counts = pd.Series(y_train).value_counts().reindex(range(len(classes)), fill_value=0).astype(float)
    mean_count = max(counts.mean(), 1.0)
    class_weight = {idx: float(mean_count / max(value, 1.0)) for idx, value in counts.items()}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    probs = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    test_acc = float(np.mean(y_pred == y_test))
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    model.save(out_dir / "skin_cnn.keras")
    return {
        "backend": "keras",
        "architecture": architecture,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(float(test_f1), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
        "classification_report": report,
        "input_size": int(input_size),
        "classes": classes,
        "normalization": {"mode": "keras_preprocess"},
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    dataset_summary = summarize_manifest(rows)
    backend = choose_backend(args.backend)
    print(json.dumps({"selected_backend": backend, "dataset_summary": dataset_summary}, indent=2))

    if backend == "torchvision":
        metrics = train_torchvision(
            rows=rows,
            architecture=args.architecture,
            image_size=args.image_size,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    else:
        metrics = train_keras(
            rows=rows,
            architecture=args.architecture,
            image_size=args.image_size,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

    labels = list(metrics["classes"])
    (out_dir / "image_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (out_dir / "image_training_metrics.json").write_text(
        json.dumps(
            {
                "dataset_summary": dataset_summary,
                **{key: value for key, value in metrics.items() if key != "classes"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "image_model_metadata.json").write_text(
        json.dumps(
            {
                "backend": metrics["backend"],
                "architecture": metrics["architecture"],
                "input_size": metrics["input_size"],
                "classes": labels,
                "normalization": metrics.get("normalization", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"saved_dir": str(out_dir), "backend": metrics["backend"], "classes": len(labels)}, indent=2))


if __name__ == "__main__":
    main()
