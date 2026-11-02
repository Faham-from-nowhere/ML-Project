# Fashion-MNIST CNN + ANN Image Classification

This project trains a real 2D CNN on Fashion-MNIST, a 28x28 grayscale image dataset with 10 clothing categories.

## Dataset
Fashion-MNIST is downloaded automatically through `torchvision.datasets.FashionMNIST` the first time you run the script. Internet access is required for the initial download. The dataset contains 60,000 training images and 10,000 test images.

The official training partition is split into:
- 54,000 training images
- 6,000 validation images

The official 10,000-image test partition is held out for final evaluation.

## Architecture
- Input: 1 x 28 x 28
- Conv2D 1 -> 32, kernel 3x3, ReLU, MaxPool 2x2
- Conv2D 32 -> 64, kernel 3x3, ReLU, MaxPool 2x2
- Conv2D 64 -> 128, kernel 3x3, ReLU
- Adaptive global average pooling: 128 feature values
- ANN: 128 -> 64 -> 10 classes

There are 224 convolution output filters in total (32 + 64 + 128). The ANN receives 128 values per image.

## Setup (Windows PowerShell)
Open PowerShell in this project folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python fashion_mnist_cnn.py
```

If your `.venv` already exists and works, activate it and just install the requirements.

## CPU/GPU
The script automatically uses CUDA if available; otherwise it runs on CPU. CPU training will take longer. Start with the default 12 epochs. You can lower `EPOCHS` near the top of the script to 5 for a quicker smoke test.

## Outputs
The script creates `outputs/` with:
- Training/validation loss and accuracy curves
- Confusion matrix
- Multiclass ROC and precision-recall curves
- Feature-map visualizations for all three convolution blocks (first 32 maps shown for layer 3)
- First-layer kernel visualizations
- 128 ANN input activations for a sample
- Classification metrics and report
- Model parameter details and saved PyTorch weights

## Notes
This is genuine image classification using Conv2D. The feature maps correspond to spatial image locations, unlike the earlier tabular Star Dataset experiment. Do not compare its accuracy directly with the Star Dataset result because the datasets and tasks are different.
