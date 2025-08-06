# Toy hybrid Quantum Neural Network (hQNN) for image classification tasks

In this repository, we show the code used to model a hybrid quantum neural network (hQNN) for image classification tasks. We employed a simple, parametrized quantum circuit (PQC) with $L$ quantum layers and a final classical linear layer. We took inspiration from the following GitHub repository: [Image_classification_with_CNN_and_QNN](https://github.com/ArunSehrawat/Image_classification_with_CNN_and_QNN)

## Requirements

### Python version

To replicate the results of this code, use Python 3.10 or more recent versions (we tested with Python 3.12.0).

### Dependencies
- torch>=1.13.0
- torchvision>=0.14.0
- numpy>=1.21.0
- pandas>=1.3.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- Pillow>=8.0.0
- matplotlib>=3.5.0

## Installation of required dependencies

### Option 1: Using pip
```bash
pip install torch torchvision numpy pandas scipy scikit-learn Pillow matplotlib
```

### Option 2: Using requirements.txt
1. Create a `requirements.txt` file with the following content:
```
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
Pillow>=8.0.0
matplotlib>=3.5.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 3: Using conda
```bash
conda install pytorch torchvision numpy pandas scipy scikit-learn pillow matplotlib -c pytorch -c conda-forge
```

## GPU Support (Optional)

For CUDA-enabled GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Supported Datasets

For this project, we used the following datasets:
- [**MNIST**](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)): Handwritten digits (28x28 grayscale);
- [**CIFAR-10 and CIFAR-100**]((https://www.cs.toronto.edu/~kriz/cifar.html)): Natural images, 10 and 100 classes (32x32 RGB), respectively;
- [**MSL**]((https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)): Mars Science Laboratory images;
- [**ImageNet**](https://www.image-net.org/update-mar-11-2021.php): Large-scale image dataset.

## System Requirements

- **Memory**: 8GB+ RAM recommended.
- **Storage**: Variable by dataset (170MB for CIFAR, several GB for ImageNet).
- **GPU**: Optional CUDA-compatible GPU for faster training.
