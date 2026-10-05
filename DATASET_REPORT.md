# Dataset inspection report

**Inspected:** `data/handloom_sarees/`  
**Scope:** Filesystem and image metadata inspection only. No labels were created, and no training or evaluation was run.

## Contents

- **JPG images:** 165 (`.jpg`), including recursive file count.
- **Subfolders:** 0. All images are directly in the dataset folder.
- **Other files:** None found. In particular, there are no CSV, JSON, TXT, XML, Excel (`.xls`, `.xlsx`, `.xlsm`), YAML, Parquet, database, or annotation/metadata-named files.
- **Filename patterns:** 156 files use `img_<number>.jpg`; 9 use `h_img_<number>.jpg`.
- **Folder names:** No subfolders, hence no folder-based labels.

The numeric filename suffixes may be image or item identifiers, but there is no evidence in the inspected files that they represent design IDs or colorway IDs. A small amount of embedded JPEG coding-table metadata was present in inspected images; no semantic design/colorway annotation was found.

## Label availability

**Design labels:** Not available in the inspected folder or filenames.  
**Colorway labels:** Not available in the inspected folder or filenames.

The images alone do not provide a verified mapping from image to design identity or colorway. No labels or results are inferred in this report. A labeled manifest or authoritative annotation source is needed before supervised training or a valid identity-based evaluation.

## Image dimensions and file sizes

There are **78 distinct image dimension pairs**. The most common dimensions are 641x641 (6 images), 1511x1512 (6), 598x598 (5), 610x610 (5), 662x662 (5), and 730x730 (5). Across all 165 JPGs, file sizes range from **4,081 bytes** to **572,630 bytes**, with a mean of approximately **114,788 bytes**.

Sample of the first 12 images in filesystem listing order:

| File | Dimensions (W x H) | Size (bytes) |
|---|---:|---:|
| `h_img_132981.jpg` | 1511x1512 | 213,184 |
| `h_img_149526.jpg` | 730x730 | 84,672 |
| `h_img_31665.jpg` | 1512x1512 | 438,421 |
| `h_img_34590.jpg` | 1185x1185 | 246,688 |
| `h_img_35259.jpg` | 1259x1259 | 266,479 |
| `h_img_44440.jpg` | 1367x1366 | 339,393 |
| `h_img_593526.jpg` | 610x610 | 39,479 |
| `h_img_76108.jpg` | 875x874 | 154,392 |
| `h_img_93057.jpg` | 1286x1286 | 475,546 |
| `img_106019.jpg` | 589x589 | 22,469 |
| `img_10959.jpg` | 486x486 | 23,104 |
| `img_110820.jpg` | 708x707 | 59,992 |

## Inspection outcome

The folder currently supports image preprocessing experiments only. It does not support supervised design recognition or identity-level retrieval/verification evaluation until genuine design and colorway annotations are supplied or established through a documented labeling process.
