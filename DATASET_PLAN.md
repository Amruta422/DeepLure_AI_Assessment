# Dataset findings and plan

**Inspected:** 2026-10-05  
**Training status:** Not trained. No evaluation has been run and no labels have been created or inferred.

## Location and provenance

The requested path `data/indian_saree_patterns/` is absent. The extracted archive found in the project is `data/archive/`. Its structure is `train/`, `valid/`, and `test/`, each containing four class folders.

The included `README.dataset.txt` identifies the package as **Indian Fabric Patterns**, links to Roboflow Universe project `div-szivu/indian-fabric-patterns`, says it was provided by a Roboflow user, and states an MIT license. `README.roboflow.txt` identifies a Roboflow export and says it includes 1,468 images. The local files therefore verify the contents and this stated Roboflow provenance; they do not independently verify that this archive is the precise Kaggle slug `div456/indian-saree-patterns` named in the assessment.

## Verified files and labels

- **Images:** 1,468 JPG files.
- **Other files:** Two TXT readme files. No CSV, JSON, XML, Excel, YAML, or separate annotation/metadata table was present.
- **Folder labels:** The four leaf-folder names are `Banarasi`, `Bandhani`, `Ikat`, and `Pichwai`. The included Roboflow README explicitly says these categories are annotated in folder format.
- **Label interpretation:** These are usable labels for the archive's four broad fabric/pattern-style categories. They are **not** exact motif/design IDs or product-instance IDs. The readmes provide no basis to claim that images within one category depict the same design.
- **Images per category:** Every category has many images, with raw totals from 316 to 489. The Roboflow README says three augmented versions were created from each source image; raw image counts therefore include generated variants and are not counts of unique design identities.
- **Colorway labels:** None found. There are no color columns, color annotation files, or colorway folders. Filename text is not treated as a verified color label.

The dataset README says images were resized to 640x640 and that salt-and-pepper noise was applied to create three versions of each source image. This is synthetic noise augmentation, not a colorway annotation or color transformation.

## Image counts by supplied split and folder class

| Class folder | Train | Valid | Test | Total |
|---|---:|---:|---:|---:|
| Banarasi | 432 | 43 | 14 | 489 |
| Bandhani | 279 | 22 | 15 | 316 |
| Ikat | 303 | 26 | 13 | 342 |
| Pichwai | 279 | 24 | 18 | 321 |
| **Total** | **1293** | **115** | **60** | **1468** |

As a limited split-integrity check, exact SHA-256 duplicates and repeated normalized source-name stems were not found crossing the supplied train/valid/test boundaries. This does not rule out visually similar or near-duplicate products with different names, and it does not turn a category label into an identity label.

## What tasks the labels support

### Supported: four-category classification

Use the provided `train` split for fitting, `valid` for model selection, and `test` once for final category classification. Report the four-category task by its actual labels (Banarasi/Bandhani/Ikat/Pichwai), with accuracy, macro-F1, per-class precision/recall, and the counts above. The test set has only 60 images (13 to 18 per class), so report that small test size and avoid overstating precision.

### Supported with careful wording: category-level retrieval

The train images can form a category gallery; valid/test images can be queries, with a match defined only as the same one of the four folder categories. Report it as **fabric/pattern-category retrieval**, not saree design identification. The provided class folders cannot establish exact same-design positives.

### Not supported: exact design retrieval or verification

Neither this Kaggle archive nor the inspected 165-image DeepLure folder has verified per-image design IDs. The DeepLure folder has no labels or metadata (`submission/DATASET_REPORT.md`). Thus a same-design/different-colorway positive pair and an exact-design gallery/query benchmark cannot be derived from the current annotations.

### Not supported: real color-invariance evaluation

There are no colorway labels or verified alternate-colorway examples. The supplied noise augmentation does not change palette. Therefore the available labels do not support a defensible real-world claim that the same design matches across colorways while different designs in the same palette remain distinct.

One limited experiment is possible later: create deterministic synthetic color transformations from held-out images and pair each generated view with the exact source image that produced it. This gives known **same-source-image** pairs by construction and can measure robustness to those transformations only. It does not provide real alternate-colorway evidence; negatives across different source images are not guaranteed different designs. Any such result must be called a synthetic transform-consistency experiment and kept separate from design recognition metrics.

## Recommended strategy

1. **First confirm provenance.** The local README names a Roboflow export, while the assessment names a Kaggle dataset. Confirm this archive is the intended Kaggle download before treating it as the stated external dataset.
2. **Use the four folder labels only for the auxiliary four-category task.** Keep the supplied train/valid/test split. Do not rename those class labels to `design_id` or use them as colorway labels.
3. **For the assessment's design-matching objective, do not claim supervised identity results from these classes.** Obtain verified design IDs and colorway IDs for at least an evaluation subset, or clearly disclose that only category-level and synthetic-view experiments are available.
4. **If proceeding without new annotations, report two separate outcomes:** category-level classification/retrieval on the supplied splits, and optionally a synthetic same-source color-transform robustness experiment. State explicitly that neither establishes real color-invariant design recognition.
5. **Train only after the task and manifest columns reflect these actual semantics.** A category-class manifest must not be passed to the design-identity training pipeline as if its category IDs were exact design IDs.
