"""Classify a query image and optionally retrieve similar gallery images by cosine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import eval_transform, list_image_paths
from model import build_model


class PathImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            return self.transform(image.convert("RGB"))


@torch.no_grad()
def encode_paths(model, paths, transform, device, batch_size=64):
    loader = DataLoader(PathImageDataset(paths, transform), batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    model.eval()
    embeddings, logits = [], []
    use_amp = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        for images in loader:
            z, scores = model(images.to(device, non_blocking=True))
            embeddings.append(z.float().cpu())
            logits.append(scores.float().cpu())
    return torch.cat(embeddings), torch.cat(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="Query image path")
    parser.add_argument("--gallery-dir", help="Optional folder; images are recursively searched")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = checkpoint["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), pretrained=False,
                        embedding_dim=checkpoint.get("embedding_dim", 256))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    query_path = Path(args.image)
    if not query_path.is_file():
        raise FileNotFoundError(query_path)
    query_embedding, query_logits = encode_paths(model, [query_path], eval_transform(), device)
    probabilities = query_logits[0].softmax(dim=0)
    predicted_idx = int(probabilities.argmax())
    output = {
        "query": str(query_path),
        "predicted_category": class_names[predicted_idx],
        "category_probabilities": {name: float(probabilities[i]) for i, name in enumerate(class_names)},
        "match_semantics": "Embedding similarities are visual similarity scores; category training does not establish exact design identity.",
        "retrieval": [],
    }
    if args.gallery_dir:
        gallery_paths = list_image_paths(args.gallery_dir)
        if not gallery_paths:
            raise ValueError(f"No supported images found under {args.gallery_dir}")
        gallery_embeddings, gallery_logits = encode_paths(model, gallery_paths, eval_transform(), device,
                                                            batch_size=args.batch_size)
        similarities = (query_embedding @ gallery_embeddings.T).squeeze(0)
        topk = min(args.top_k, len(gallery_paths))
        values, indices = torch.topk(similarities, k=topk)
        gallery_predictions = gallery_logits.softmax(dim=1).argmax(dim=1)
        output["retrieval"] = [
            {"rank": rank + 1, "path": str(gallery_paths[int(index)]),
             "cosine_similarity": float(values[rank]),
             "predicted_category": class_names[int(gallery_predictions[int(index)])]}
            for rank, index in enumerate(indices)
        ]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
