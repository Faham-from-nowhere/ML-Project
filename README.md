# Star Dataset: K-Means + PCA

## Files
- `star_clustering_pca.py` — complete implementation.
- `requirements.txt` — Python dependencies.
- `Stars.csv` — add the Kaggle dataset CSV here (not included in this ZIP).

## Dataset
Download the Star dataset from:
https://www.kaggle.com/datasets/deepu1109/star-dataset

Rename the downloaded CSV to `Stars.csv` and place it in this folder.

## Setup
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python star_clustering_pca.py
```

## Implementation notes
- K-Means is implemented from scratch: random centroid initialization, distance calculation, assignment, centroid updates, and WCSS.
- K values 2 through 6 are tested.
- PCA is implemented from scratch using the covariance matrix and eigenvalue decomposition.
- `StandardScaler` is used to standardize the four numeric features.
- `Star type` is not used as a clustering feature; it is used only after clustering for interpretation.
- Scikit-learn is used for cluster evaluation metrics, not for K-Means or PCA.

## Outputs
The script creates an `outputs/` directory containing the elbow curve, convergence plots, cluster-size plots, evaluation metrics, PCA variance plots, 2D/3D visualizations, and CSV result tables.

**Important:** `SELECTED_K = 3` is only a starting setting. Examine the elbow curve and metric table, then change it if the results support another value. Do not report invented metric values; use the values produced by your run.
