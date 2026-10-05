"""Train a ResNet-50 category classifier and category-level image embedding."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CategoryFolderDataset, eval_transform, resolve_data_root, train_transform
from losses import supervised_contrastive_loss
from model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.cuda.amp.autocast(enabled=use_amp):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Folder containing train/, valid/, and test/.")
    parser.add_argument("--output-dir", default="runs/resnet50_category")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet ResNet-50 weights.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be >=1 and --batch-size must be >=2")

    seed_everything(args.seed)
    data_root = resolve_data_root(args.data_root)
    train_ds = CategoryFolderDataset(data_root, "train", transform=train_transform())
    valid_ds = CategoryFolderDataset(data_root, "valid", class_to_idx=train_ds.class_to_idx,
                                     transform=eval_transform())
    # Confirm the held-out test split is present and shares the exact same categories.
    CategoryFolderDataset(data_root, "test", class_to_idx=train_ds.class_to_idx, transform=eval_transform())
    if len(train_ds.classes) != 4:
        raise ValueError(f"Expected the four verified category folders; found {train_ds.classes}")

    generator = torch.Generator().manual_seed(args.seed)
    common_loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                              generator=generator)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common_loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **common_loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = build_model(num_classes=len(train_ds.classes), pretrained=not args.no_pretrained).to(device)
    labels_np = np.asarray([label for _, label in train_ds.samples], dtype=np.int64)
    frequencies = np.bincount(labels_np, minlength=len(train_ds.classes)).astype(np.float32)
    class_weights = frequencies.sum() / np.maximum(frequencies, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"class_to_idx": train_ds.class_to_idx, "classes": train_ds.classes,
                           "architecture": "torchvision ResNet-50", "embedding_dim": 256,
                           "pretrained_weights": None if args.no_pretrained else "ResNet50_Weights.IMAGENET1K_V2",
                           "amp_enabled": use_amp, "seed": args.seed}
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    history = []
    best_accuracy = -1.0
    best_valid_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                embeddings, logits = model(images)
                cls_loss = criterion(logits, labels)
            # Keep pairwise similarities, exponentiation, and log reductions in fp32.
            metric_loss = supervised_contrastive_loss(embeddings.float(), labels)
            loss = cls_loss.float() + args.contrastive_weight * metric_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().item() * labels.size(0)
            examples += labels.size(0)

        scheduler.step()
        valid_loss, valid_accuracy = validate(model, valid_loader, criterion, device, use_amp)
        row = {"epoch": epoch, "train_loss": loss_sum / max(examples, 1),
               "valid_loss": valid_loss, "valid_accuracy": valid_accuracy,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print("epoch {epoch:02d}/{total} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} valid_accuracy={valid_accuracy:.4f}".format(
            total=args.epochs, **row))

        if valid_accuracy > best_accuracy or (valid_accuracy == best_accuracy and valid_loss < best_valid_loss):
            best_accuracy, best_valid_loss = valid_accuracy, valid_loss
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes,
                        "class_to_idx": train_ds.class_to_idx, "embedding_dim": 256,
                        "architecture": "torchvision ResNet-50", "epoch": epoch,
                        "pretrained_weights": None if args.no_pretrained else "ResNet50_Weights.IMAGENET1K_V2",
                        "valid_accuracy": valid_accuracy, "valid_loss": valid_loss,
                        "seed": args.seed, "input_size": 224}, output_dir / "best_model.pt")

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"Best checkpoint: {output_dir / 'best_model.pt'} (valid accuracy={best_accuracy:.4f})")


if __name__ == "__main__":
    main()
