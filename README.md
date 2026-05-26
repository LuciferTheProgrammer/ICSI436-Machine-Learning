# ICSI436 - Machine Learning
Applications are contained in the MyFiles folder.

## Neural Networks

By Rheinard Zadanowsky, Edmond Wong, Lucas Depaola, Chaoji Yang, and Matthew Niemczyk.

This repository contains a Python-based Machine Learning project focused on implementing neural networks for classification tasks. The project includes binary classification, feedforward multi-class classification, and convolutional neural network classification.

The project uses the Iris dataset for binary and feedforward multi-class classification, and the MNIST dataset for CNN-based handwritten digit classification.

## Project Overview

This project implements neural network models from scratch using Python and NumPy. The program trains and evaluates models using forward propagation, loss computation, backpropagation, gradient descent, activation functions, and data preprocessing.

The project includes three main classification modes:

- Binary classification using the Iris dataset
- Feedforward multi-class classification using the Iris dataset
- CNN multi-class classification using the MNIST dataset

The program also generates output text files and visual graphs showing loss curves and actual vs. predicted classification results.

## Features

- Python neural network implementation
- Binary classification
- Feedforward multi-class classification
- Convolutional neural network classification
- Iris dataset support
- MNIST dataset support
- Local MNIST dataset caching using `mnist_784.npz`
- OpenML MNIST dataset fetching if local file is unavailable
- Forward propagation
- Backpropagation
- Gradient descent
- Xavier initialization for feedforward neural networks
- He/Kaiming initialization for CNN layers
- Sigmoid activation
- ReLU activation
- Softmax activation
- Cross-entropy loss
- One-hot encoding
- Mini-batch training for CNN
- Max pooling
- Convolution operations
- Loss curve plotting
- Actual vs. predicted result plotting
- Output result files for model loss records

## Project Files

- `NeuralNetwork.py` - main Python source code containing the neural network implementations, training logic, dataset loading, plotting, and program menu
- `mnist_784.npz` - local cached MNIST dataset file used by the CNN model, if included in the repo/submission
- Output text files are generated after running the models
- Output graph images are generated after running the models

## IDE and Environment

IDE used:

```text
IntelliJ IDEA 2025.1 (Ultimate Edition)
Build #IU-251.23774.435, built on April 14, 2025
```

Plugins used:

```text
Python - version 251.23774.460
Python Community Edition - version 251.23774.460
```

Virtual environment:

```text
.venv
```

Python interpreter:

```text
Python 3.13
```

## Python Libraries Used

```text
numpy - Version: 2.2.3
scikit-learn - Version: 1.6.1
matplotlib - Version: 3.10.1
```

## Imports Used

```python
import numpy as number_array
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import os
```

## Import Purposes

- `numpy` - numerical computations and basic data processing
- `load_iris` - loads the Iris dataset
- `train_test_split` - splits the dataset into training and testing data
- `fetch_openml` - fetches the MNIST dataset from OpenML
- `matplotlib.pyplot` - creates graphs and plots
- `sliding_window_view` - supports CNN convolution operations
- `os` - checks for and saves the local MNIST dataset file

## Datasets

### Iris Dataset

The Iris dataset is used for:

- Binary classification
- Feedforward multi-class classification

For binary classification, the program filters the Iris dataset to use two classes.

For multi-class classification, the program uses the full Iris dataset with all three classes.

### MNIST Dataset

The MNIST dataset is used for CNN multi-class classification.

Website for MNIST dataset:

```text
https://www.openml.org/search?type=data&sort=runs&id=554&status=active
```

Please use the local `mnist_784.npz` file in the compressed submission for the source code for the Neural Network. Make sure both files are in the same directory so the Neural Network Python file can use and pick up the MNIST dataset.

If this does not work, please make sure to have an internet connection so the source code can fetch and retrieve the MNIST dataset from the OpenML website.

## How It Works

When the program runs, the user chooses between binary classification and multi-class classification.

For binary classification, the program trains a feedforward neural network on a filtered version of the Iris dataset. It uses forward propagation, sigmoid output activation, loss calculation, backpropagation, and gradient descent.

For feedforward multi-class classification, the program trains a neural network on the full Iris dataset. It uses one-hot encoding, softmax output activation, cross-entropy loss, and gradient descent.

For CNN multi-class classification, the program trains a convolutional neural network on the MNIST dataset. The CNN includes convolutional layers, ReLU activation, max pooling, a dense layer, an output layer, and softmax classification.

After training, the program prints loss values, computes experimental loss on test data, calculates accuracy, saves output text files, and creates plots for training loss and actual vs. predicted results.

## Running the Program

1. Open the project in IntelliJ IDEA or another Python-compatible IDE.

2. Make sure the virtual environment is active.

3. Make sure the required Python libraries are installed:

```bash
pip install numpy scikit-learn matplotlib
```

4. Place `mnist_784.npz` in the same directory as `NeuralNetwork.py` if using the local MNIST dataset file.

5. Run:

```bash
python NeuralNetwork.py
```

6. Follow the prompts in the terminal.

The program first asks:

```text
For Binary Classification enter 'b' or for Multi-class Classification enter 'm':
```

If multi-class classification is selected, it then asks:

```text
For CNN enter 'c' or Feedforward enter 'f':
```

## Suggested Parameters

When running the program, please follow the suggested parameters so the expected results are presented.

### Binary Classification

```text
learning_rate = 0.4
hidden_layer_size = 4
upper_boundary = 0.0001
num_iterations = 200
```

### CNN Multi-class Classification

```text
learning_rate = 0.33
number_iterations = 30
batch_size = 64
upper_boundary = 0.000001
```

### Feedforward Multi-class Classification

```text
learning_rate = 0.01
hidden_layer_size = 4
upper_boundary = 0.000001
num_iterations = 5000
```

## Output Files

The program generates text output files containing loss records and experimental loss values.

Possible generated output files include:

```text
Output_FF_Binary_Classification.txt
Output_CNN_Multi_Class_Classification.txt
Output_FF_Multi_Class_Classification.txt
```

The program also generates graph images showing loss curves and actual vs. predicted values.

Possible generated graph files include:

```text
[Iris Dataset - Binary] Loss.png
[Iris Dataset - Binary] Actual vs. Predicted values.png
[Iris Dataset - Multi-class] Loss.png
[Iris Dataset - Multi-class] Actual vs. Predicted values.png
[MNIST - Multi-class] Loss.png
[MNIST - Multi-class] Actual vs. Predicted values.png
```

## Technologies Used

- Python
- NumPy
- scikit-learn
- matplotlib
- OpenML
- Neural networks
- Feedforward neural networks
- Convolutional neural networks
- Backpropagation
- Gradient descent
- Softmax classification
- Binary classification
- Multi-class classification
- Data visualization

## Purpose

This project was created for the ICSI436 Machine Learning course to practice building and training neural networks.

The project demonstrates how neural networks can be implemented using lower-level numerical operations instead of relying on high-level deep learning frameworks. It also shows how different neural network models can be applied to binary classification, multi-class classification, and image classification tasks.
