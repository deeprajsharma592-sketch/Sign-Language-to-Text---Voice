"""Training loop, validation, metrics calculation, and confusion matrix export."""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import (
    SignDataset,
    load_dataset_from_disk,
    split_dataset,
    generate_synthetic_dataset,
)
from src.model import build_model
from src.features import TOTAL_RAW_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: Path
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")


def get_top_confused_pairs(cm: np.ndarray, classes: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    pairs = []
    n = len(classes)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i, j])
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def run_training(
    config_path: str = "config/labels.yaml",
    use_synthetic: bool = False,
    override_model: str = None,
    eval_only: bool = False
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    vocab = cfg["vocabulary"]
    model_type = override_model or cfg["training"].get("model_type", "gru")
    include_velocity = cfg["sequence"].get("include_velocity", False)
    input_dim = TOTAL_RAW_DIM * 2 if include_velocity else TOTAL_RAW_DIM
    batch_size = cfg["training"].get("batch_size", 16)
    epochs = cfg["training"].get("epochs", 40)
    lr = cfg["training"].get("learning_rate", 0.001)

    logger.info(f"Vocabulary: {vocab}")
    logger.info(f"Model: {model_type.upper()} | Input Dim: {input_dim} (Velocity: {include_velocity})")

    if use_synthetic:
        logger.info("[SYNTHETIC MODE] Generating synthetic dataset for CI/pipeline test...")
        samples, labels, sessions, label_to_idx = generate_synthetic_dataset(vocab, samples_per_class=30)
    else:
        samples, labels, sessions, label_to_idx = load_dataset_from_disk("data/raw", vocab)
        if len(samples) == 0:
            logger.error("No samples found in data/raw! Provide pre-recorded data or pass '--synthetic' to test.")
            return

    train_idx, val_idx = split_dataset(samples, labels, sessions, test_size=0.2, seed=seed)
    logger.info(f"Dataset split: Train samples={len(train_idx)}, Val samples={len(val_idx)}")

    train_samples = [samples[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = SignDataset(train_samples, train_labels, augment=True, include_velocity=include_velocity)
    val_dataset = SignDataset(val_samples, val_labels, augment=False, include_velocity=include_velocity)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class weighting to counter imbalance
    class_counts = np.bincount(train_labels, minlength=len(vocab))
    weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    weights = weights / (weights.sum() + 1e-8) * len(vocab)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=len(vocab),
        hidden_dim=cfg["training"].get("hidden_dim", 64),
        num_layers=cfg["training"].get("num_layers", 2),
        dropout=cfg["training"].get("dropout", 0.2),
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_checkpoint_path = models_dir / f"best_model_{model_type}.pt"

    logger.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                logits = model(x_v)
                loss = criterion(logits, y_v)
                val_loss += loss.item() * len(y_v)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_v).sum().item()
                val_total += len(y_v)

        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "input_dim": input_dim,
                "num_classes": len(vocab),
                "vocabulary": vocab,
                "config": cfg,
                "val_acc": val_acc,
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f})")
                break

    # Save standard checkpoint link
    default_checkpoint = models_dir / "best_model.pt"
    torch.save(torch.load(best_checkpoint_path), default_checkpoint)

    # Save label map
    label_map_path = models_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({"classes": vocab, "label_to_idx": label_to_idx}, f, indent=2)

    logger.info(f"Best model saved to {best_checkpoint_path} and {default_checkpoint}")

    # Final Evaluation on Val Set using Best Checkpoint
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for x_v, y_v in val_loader:
            preds = model(x_v).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y_v.cpu().numpy())

    cm = confusion_matrix(all_trues, all_preds, labels=list(range(len(vocab))))
    report_dict = classification_report(all_trues, all_preds, target_names=vocab, output_dict=True, zero_division=0)
    top_confused = get_top_confused_pairs(cm, vocab)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    cm_path = reports_dir / f"confusion_matrix_{model_type}.png"
    plot_and_save_confusion_matrix(cm, vocab, cm_path)

    summary_metrics = {
        "synthetic": use_synthetic,
        "model_type": model_type,
        "val_accuracy": float(report_dict["accuracy"]),
        "classification_report": report_dict,
        "top_confused_pairs": top_confused,
    }

    metrics_path = reports_dir / f"metrics_{model_type}.json"
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Evaluation Complete! Accuracy: {report_dict['accuracy']:.4f}")
    if use_synthetic:
        logger.info("[NOTE] These metrics are on SYNTHETIC data and reflect pipeline verification only.")
    logger.info(f"Top confused pairs: {top_confused}")
    logger.info("=" * 60)
    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Model Training")
    parser.add_argument("--config", type=str, default="config/labels.yaml", help="Path to config file")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic trajectory dataset")
    parser.add_argument("--model", type=str, default=None, help="Override model type ('gru' or 'cnn')")
    args = parser.parse_args()

    run_training(config_path=args.config, use_synthetic=args.synthetic, override_model=args.model)
