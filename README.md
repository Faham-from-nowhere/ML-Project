# Star Dataset — 1D CNN + ANN

## Task
Six-class supervised classification: predict `Star type` from the four numerical attributes:
- Temperature (K)
- Luminosity(L/Lo)
- Radius(R/Ro)
- Absolute magnitude(Mv)

`Star color` and `Spectral Class` are not model inputs. `Star type` is the target label.

## Dataset location
The script automatically checks:
1. `archive/6 class csv.csv`
2. `6 class csv.csv`
3. `Stars.csv`

Place the downloaded Kaggle CSV in one of those locations.

## Setup and run
```bash
pip install -r requirements.txt
python star_cnn_classification.py
```

## CNN architecture and feature-map accounting
- Input tensor per sample: 1 channel × 4 feature positions
- Conv1D layer 1: 16 filters, kernel size 2; output feature maps: 16 × 4
- Conv1D layer 2: 32 filters, kernel size 2; output feature maps: 32 × 4
- Total learned convolution filters: 48
- Global average pooling converts 32 × 4 activations into 32 numbers
- The ANN receives 32 values per sample
- ANN hidden layer: 32 units
- Output layer: 6 units (one per Star Type class)

## Evaluation
The code makes stratified train/validation/test splits (70/15/15), fits standardization only on training data, trains with early stopping, and evaluates the held-out test set. It reports accuracy, macro/weighted precision, recall and F1, ROC-AUC, classification report, and confusion matrix. It saves training curves, ROC and precision-recall curves, learned feature maps, learned kernel weights, and the ANN input activations.

## Important scientific limitation
This dataset contains only four tabular numerical features, not images. The code uses Conv1D as an educational demonstration by treating the four feature positions as a short sequence. Their order is not naturally temporal or spatial, so this is not necessarily the most appropriate model. A conventional MLP should be used as a baseline for a fair comparison. The reported CNN performance must be obtained by running the script; no result is assumed in advance.
