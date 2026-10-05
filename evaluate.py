"""Evaluate category retrieval/classification and a separate synthetic same-image color test."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CategoryFolderDataset, eval_transform, resolve_data_root, synthetic_recolor_transform
from model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def embed_dataset(model, dataset, device, batch_size=64, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    model.eval()
    all_embeddings, all_logits, all_labels = [], [], []
    use_amp = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        for images, labels in loader:
            embeddings, logits = model(images.to(device, non_blocking=True))
            all_embeddings.append(embeddings.float().cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_logits), np.concatenate(all_labels)


def retrieval_metrics(gallery_embeddings, query_embeddings, gallery_ids, query_ids, ks=(1, 5, 10)):
    """Query hit-rate Recall@K and mAP for the supplied relevance IDs."""
    similarities = query_embeddings @ gallery_embeddings.T
    recalls = {k: [] for k in ks}
    average_precisions = []
    for row_idx, scores in enumerate(similarities):
        order = np.argsort(-scores)
        relevant = gallery_ids[order] == query_ids[row_idx]
        total_relevant = int(np.sum(gallery_ids == query_ids[row_idx]))
        if total_relevant == 0:
            continue
        for k in ks:
            recalls[k].append(float(relevant[:min(k, len(relevant))].any()))
        positions = np.flatnonzero(relevant) + 1
        precision_at_relevant = np.arange(1, len(positions) + 1) / positions
        average_precisions.append(float(precision_at_relevant.mean()))
    result = {f"Recall@{k}": float(np.mean(values)) for k, values in recalls.items() if values}
    result["mAP"] = float(np.mean(average_precisions)) if average_precisions else float("nan")
    result["query_count"] = int(len(query_ids))
    result["gallery_count"] = int(len(gallery_ids))
    return result


def classification_metrics(logits, labels, class_names):
    predictions = logits.argmax(axis=1)
    accuracy = float(np.mean(predictions == labels))
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, pred in zip(labels, predictions):
        confusion[int(truth), int(pred)] += 1
    per_class = {}
    for idx, name in enumerate(class_names):
        denom = int(confusion[idx].sum())
        per_class[name] = {"accuracy": float(confusion[idx, idx] / denom) if denom else None,
                           "support": denom}
    return {"accuracy": accuracy, "per_class": per_class,
            "confusion_matrix_rows_true_columns_predicted": confusion.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Folder containing train/, valid/, test/.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = checkpoint["classes"]
    class_to_idx = checkpoint["class_to_idx"]
    if len(class_names) != 4:
        raise ValueError(f"Expected four broad category classes, got {class_names}")
    data_root = resolve_data_root(args.data_root)
    train_ds = CategoryFolderDataset(data_root, "train", class_to_idx=class_to_idx, transform=eval_transform())
    query_ds = CategoryFolderDataset(data_root, args.split, class_to_idx=class_to_idx, transform=eval_transform())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names), pretrained=False,
                        embedding_dim=checkpoint.get("embedding_dim", 256))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    gallery_embeddings, _, gallery_labels = embed_dataset(model, train_ds, device,
        batch_size=args.batch_size, num_workers=args.num_workers)
    query_embeddings, query_logits, query_labels = embed_dataset(model, query_ds, device,
        batch_size=args.batch_size, num_workers=args.num_workers)
    if not set(np.unique(query_labels)).issubset(set(np.unique(gallery_labels))):
        raise ValueError("Every query category must have at least one train gallery image")

    # Separate synthetic protocol: one source test/valid image is the gallery item, and
    # its deterministically seeded color-perturbed copy is the paired query. Pairing is
    # known from construction, not an inferred design/colorway label.
    seed_everything(args.seed + 10_000)
    synthetic_ds = CategoryFolderDataset(data_root, args.split, class_to_idx=class_to_idx,
                                         transform=synthetic_recolor_transform())
    synthetic_embeddings, synthetic_logits, synthetic_labels = embed_dataset(model, synthetic_ds, device,
        batch_size=args.batch_size, num_workers=0)
    source_ids = np.arange(len(query_ds), dtype=np.int64)
    synthetic_ids = np.arange(len(synthetic_ds), dtype=np.int64)
    if len(source_ids) != len(synthetic_ids):
        raise RuntimeError("Source and synthetic datasets differ in size/order")

    result = {
        "scope": "4-way broad category classification and category-level retrieval",
        "split": args.split,
        "classes": class_names,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": args.seed,
        "device": str(device),
        "classification": classification_metrics(query_logits, query_labels, class_names),
        "category_level_retrieval_gallery_train": retrieval_metrics(
            gallery_embeddings, query_embeddings, gallery_labels, query_labels),
        "synthetic_same_image_color_transform": {
            "scope": "synthetic transform-consistency only; same source file paired by construction; not real cross-colorway design recognition",
            "transform": "ColorJitter brightness/contrast/saturation/hue followed by occasional grayscale",
            "source_image_count": len(query_ds),
            "transformed_category_accuracy_diagnostic": classification_metrics(
                synthetic_logits, synthetic_labels, class_names)["accuracy"],
            "same_source_image_retrieval": retrieval_metrics(
                query_embeddings, synthetic_embeddings, source_ids, synthetic_ids),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
