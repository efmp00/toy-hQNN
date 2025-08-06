# Toy hybrid Quantum Neural Network (hQNN) for image classification tasks

The code shown here can be used to model a hybrid quantum neural network (hQNN) for image classification tasks. We employed a simple, parametrized quantum circuit (PQC) consisting of rotation and entangling gates with $L$ quantum layers and a final classical linear layer. 

We took inspiration from the following GitHub repository to build the PQC: [Image_classification_with_CNN_and_QNN](https://github.com/ArunSehrawat/Image_classification_with_CNN_and_QNN).

## Requirements

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
- [**MNIST**](https://www.kaggle.com/datasets/hojjatk/mnist-dataset): Handwritten digits (28x28 grayscale);
- [**CIFAR-10 and CIFAR-100**](https://www.cs.toronto.edu/~kriz/cifar.html): Natural images, 10 and 100 classes (32x32 RGB), respectively;
- [**MSL**](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl): Mars Science Laboratory images;
- [**ImageNet**](https://www.image-net.org/update-mar-11-2021.php): Large-scale image dataset.

> [!NOTE]
> Some datasets are downloaded automatically, while others require manual installation. You can access each dataset in the link provided. We suggest creating a folder called 'Datasets' to store all the training data. 

## System Requirements

- **Memory**: 8GB+ RAM recommended.
- **Storage**: Variable by dataset (170MB for CIFAR, several GB for ImageNet).
- **GPU**: Optional CUDA-compatible GPU for faster training.

## References
- Karkera, P., R, S. (2024). Optimization Techniques of Quantum Neural Network for Image Classification. In: Abraham, A., Bajaj, A., Hanne, T., Hong, TP. (eds) Intelligent Systems Design and Applications. ISDA 2023. Lecture Notes in Networks and Systems, vol 1047. Springer, Cham. [https://doi.org/10.1007/978-3-031-64836-6_41](https://doi.org/10.1007/978-3-031-64836-6_41).
- Benedetti, M., Lloyd, E., Sack, S., Fiorentini, M. (2019). Parametrized quantum circuits as machine learning models. Quantum Science and Technology, 4(4), 043001. [https://doi.org/10.1088/2058-9565/ab4eb5](https://doi.org/10.1088/2058-9565/ab4eb5)
- Mitarai, K., Negoro, M., Kitagawa, M., Fujii, K. (2018). Quantum circuit learning. Physical Review A, 98(3), 032309. [https://doi.org/10.1103/PhysRevA.98.032309](https://doi.org/10.1103/PhysRevA.98.032309).

> [!IMPORTANT]
> To replicate the results of this code, use Python 3.10 or more recent versions (we tested with Python 3.12.0).
