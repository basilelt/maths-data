# Multi-Class Logistic Regression for Digit Classification

This project implements a multi-class logistic regression model from scratch for handwritten digit classification (0-9) using the scikit-learn digits dataset. The implementation includes gradient descent optimization, regularization, and comprehensive analysis comparing performance with scikit-learn's built-in logistic regression.

## Project Overview

The main goal is to classify handwritten digits from 8x8 pixel grayscale images. The project demonstrates:

- **From-scratch implementation**: Custom `MultiClassLogisticRegression` class using gradient descent
- **Performance comparison**: Benchmarking against scikit-learn's `LogisticRegression`
- **Hyperparameter analysis**: Studying the impact of learning rate and iterations
- **Model interpretability**: Visualizing learned coefficients
- **Error analysis**: Examining misclassified samples
- **Regularization**: L2 regularization implementation
- **Generalization testing**: Evaluation on MNIST dataset

## Dataset

- **Source**: scikit-learn's `digits` dataset
- **Size**: 1,797 samples
- **Features**: 64 pixel values (8x8 images)
- **Classes**: 10 (digits 0-9)
- **Split**: 80% training, 20% testing

## Project Structure

```
projet/
├── documentation/          # Project report and documentation
│   ├── rapport_projet.md   # Detailed project report (French)
│   ├── projet.pdf          # Project PDF
│   └── grille_evaluation_2A_projet_mds.csv  # Evaluation criteria
├── models/                 # Saved trained models
│   └── trained_model.pkl   # Pickled custom model
├── results/                # Analysis results and visualizations
│   ├── confusion_matrix_*.png          # Confusion matrices
│   ├── coefficient_visualization.png   # Learned weights visualization
│   ├── parameter_analysis.png          # Hyperparameter analysis
│   ├── misclassified_samples.png       # Error analysis
│   └── mnist_test_results.txt          # MNIST generalization results
└── scripts/                # Python scripts
    ├── multi_class_logistic.py         # Custom logistic regression implementation
    ├── train_digits.py                 # Train custom model
    ├── train_digits_regularized.py     # Train with regularization
    ├── compare_models.py               # Compare custom vs sklearn
    ├── analyze_custom_model.py         # Detailed analysis
    ├── explore_digits.py               # Dataset exploration
    ├── test_mnist.py                   # MNIST generalization test
    └── sklearn_logistic.py             # Scikit-learn baseline
```

## Requirements

Install dependencies using:
```bash
pip install -r ../requirements.txt
```

Required packages:
- numpy
- scikit-learn
- matplotlib
- seaborn
- pandas

## Usage

### Training the Custom Model

Run the main training script:
```bash
cd scripts
python train_digits.py
```

This will:
- Load the digits dataset
- Train the custom multi-class logistic regression model
- Print training and test accuracies
- Save the trained model to `../models/trained_model.pkl`

### Comparing with Scikit-Learn

Compare custom implementation with scikit-learn:
```bash
python compare_models.py
```

### Analyzing the Model

Perform detailed analysis including parameter studies and visualizations:
```bash
python analyze_custom_model.py
```

### Testing Generalization on MNIST

Evaluate model generalization on MNIST dataset:
```bash
python test_mnist.py
```

### Training with Regularization

Train the model with L2 regularization:
```bash
python train_digits_regularized.py
```

## Key Results

- **Custom Model Accuracy**: 97.50% on test set
- **Scikit-Learn Accuracy**: 97.50% on test set
- **MNIST Generalization**: 24.67% (significant drop due to dataset differences)

## Implementation Details

### Custom MultiClassLogisticRegression Class

- **Softmax**: Numerically stable softmax implementation
- **Loss**: Cross-entropy loss
- **Optimization**: Gradient descent with configurable learning rate and iterations
- **Regularization**: Optional L2 regularization
- **Parameter handling**: Flattened parameter vector for optimization

### Gradient Descent

- Basic gradient descent implementation
- Configurable learning rate and maximum iterations
- Support for regularization

## Mathematical Background

The model implements multi-class logistic regression using softmax activation:

$$P(Y = k | X = x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

where $z_k = x^T w_k + b_k$

Optimized using gradient descent on cross-entropy loss:

$$J(W, b) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik})$$

## Authors

- Basile LE THIEC
- Lilian NOACCO

## License

This project is part of an academic assignment for MDS (Data Science) coursework.