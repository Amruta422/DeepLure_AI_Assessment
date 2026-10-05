# Saree image retrieval baseline (four broad categories)

This is a PyTorch ResNet-50 baseline for image classification and cosine-similarity retrieval using the four folder categories in the local Roboflow export: **Banarasi, Bandhani, Ikat, and Pichwai**.

## Label scope and limitations

These four labels are broad fabric/pattern-style categories only. They are **not** design IDs, product IDs, motif identities, or colorway labels. An image that shares a folder with another image is not thereby known to show the same design. Accordingly:

- Recall@K and mAP below measure retrieval of the same **broad category**, not the same saree design.
- Classification accuracy measures prediction of those four categories.
- The synthetic recoloring result pairs a transformed image with the exact source image from which it was generated. It measures only synthetic same-image transform consistency; it is **not** real alternate-colorway or cross-design evaluation.
- No `design_id` or `colorway_id` values are fabricated or inferred.
- The 165 unlabeled proprietary DeepLure images are not used for supervised training or evaluation and must not be copied to repository outputs.

The inspected archive is currently at `data/archive/`, with `train/`, `valid/`, and `test/` subfolders. Its readme identifies it as a Roboflow export named Indian Fabric Patterns; verify provenance if associating it with the assessment's Kaggle slug. See [DATASET_PLAN.md](DATASET_PLAN.md) for verified counts, splits, and limitations.

## Model and training

- Torchvision ResNet-50 with the pinned ImageNet-1K V2 pretrained weights.
- 256-dimensional L2-normalized projection embedding and four-way linear classifier.
- Training augmentations: random resized crop, horizontal flip, hue, saturation, brightness, contrast, and occasional grayscale.
- Objective: class-weighted cross entropy plus supervised contrastive loss. Contrastive positives mean same folder category only; this objective intentionally represents the dataset labels and does not create design identity labels.
- Uses only the supplied `train/` split for gradient updates. `valid/` selects the best checkpoint by validation accuracy (validation loss breaks ties); `test/` is not used for checkpoint selection.
- Fixed seed defaults to 42. CUDA mixed precision is enabled when a CUDA device is available.
- Best checkpoint saved as `best_model.pt`, with `history.csv` and `run_config.json` alongside it.

The `--data-root` directory can either directly contain `train/`, `valid/`, and `test/`, or contain an `archive/` subfolder with those three splits. Each split must have the same four class subfolders.

## Recommended Kaggle commands

Attach the dataset input in Kaggle, enable a GPU, put this `submission/` directory at `/kaggle/working/submission`, and ensure the attached input contains the three split directories either directly or under `archive/`. With the Kaggle input slug mounted at `/kaggle/input/indian-saree-patterns`, run these as separate Kaggle notebook cells:

```bash
!python /kaggle/working/submission/train.py --data-root /kaggle/input/indian-saree-patterns --output-dir /kaggle/working/runs/resnet50_category --epochs 15 --batch-size 32 --num-workers 2 --seed 42
!python /kaggle/working/submission/evaluate.py --data-root /kaggle/input/indian-saree-patterns --checkpoint /kaggle/working/runs/resnet50_category/best_model.pt --split valid --output /kaggle/working/runs/resnet50_category/valid_metrics.json --seed 42
!python /kaggle/working/submission/evaluate.py --data-root /kaggle/input/indian-saree-patterns --checkpoint /kaggle/working/runs/resnet50_category/best_model.pt --split test --output /kaggle/working/runs/resnet50_category/test_metrics.json --seed 42
```

The mounted Kaggle input directory name can differ from the slug. If so, change only the `/kaggle/input/indian-saree-patterns` prefix to the actual mounted directory. The data-root resolver handles either a direct split layout or an `archive/` child. For the inspected local layout, use `--data-root data/archive`.

Install dependencies if needed with `pip install -r /kaggle/working/submission/requirements.txt`. Torchvision downloads pretrained weights on first use, so Kaggle Internet must be enabled or the weights must already be cached.

## Metrics and protocol

For `--split valid` or `--split test`, the gallery is all training images, and the query set is the requested held-out split. Query labels are category folders. A gallery item is relevant exactly when it has the same one of the four category labels as the query. Recall@K is the fraction of queries with at least one same-category gallery item in the top K; mAP is mean average precision over the same category relevance labels. Classification accuracy and a confusion matrix are also reported.

The separate `synthetic_same_image_color_transform` result applies seeded ColorJitter (hue/saturation/brightness/contrast) and occasional grayscale to each held-out image, then retrieves against the untransformed versions of those exact held-out source files. The source-query pairing is known because the transformed query is generated from that exact file. This protocol does not test whether two separate saree photos show the same physical design and does not support a real color-invariance claim. It is reported separately from category retrieval.

Metrics are written only when the evaluation script is run. No accuracy or retrieval result is asserted in this repository before a run.

## Inference

Classify a query and optionally retrieve visually similar files from a directory:

```bash
python inference.py --checkpoint runs/resnet50_category/best_model.pt --image path/to/query.jpg --gallery-dir data/archive/train --top-k 5
```

Similarity results from this category-trained embedding are visual neighbors, not verified same-design matches.

## Repository hygiene

The project `.gitignore` excludes `/data/`, model checkpoints, and run outputs. Do not force-add dataset images or place proprietary DeepLure material in a GitHub repository.
