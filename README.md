ICSI 436/536 - Machine Learning Project [Neural Networks]

By Rheinard Zadanowsky, Edmond Wong, Lucas Depaola, Chaoji Yang,
and Matthew Niemczyk.

IDE used is [IntelliJ IDEA]:
IntelliJ IDEA 2025.1 (Ultimate Edition)
Build #IU-251.23774.435, built on April 14, 2025

Plugins used:
[Python] version - 251.23774.460
[Python Community Edition] version - 251.23774.460

Used (.venv) - Virtual Environment.


Python Interpreter - Version: python 3.13
numpy - Version: 2.2.3
scikit-learn - Version: 1.6.1
matplotlib - Version: 3.10.1

All imports used in the project:

import numpy as number_array                           - Numerical computations/Basic data processing
from sklearn.datasets import load_iris                 - Loads Iris Dataset
from sklearn.model_selection import train_test_split   - Splits the Dataset for training and testing 
from sklearn.datasets import fetch_openml              - Fetches MNIST dataset from OpenML website (need internet connection)
import matplotlib.pyplot as plt                        - For graphs
from numpy.lib.stride_tricks import sliding_window_view - Basic data processing - used on CNN
import os                                               - To check/save local copy of MNIST dataset in current directory 

Website for MNIST dataset: https://www.openml.org/search?type=data&sort=runs&id=554&status=active

Please use the local "mnist_784.npz" file in the compressed submission for the source code for the
Neural Network and make sure both files are in the same directory for the Neural Network python file to 
use and pick up the mnist dataset. If this doesn't work, please make sure to have an internet connection so 
the source code (Neural Network python file) can fetch and retrieve the mnist dataset from the OpenML website.

When running the program, please also follow the "Suggested Parameters" so that the actual 
results are presented. 

For Binary Classification:
Suggested Parameters: learning_rate = 0.4, hidden_layer_size = 4, upper_boundary = 0.0001, num_iterations = 200

For CNN Multi-class Classification:
Suggested Parameters: learning_rate = 0.33, number_iterations = 30, batch_size = 64, upper_boundary = 0.000001

For Feedforward Multi-class Classification:
Suggested Parameters: learning_rate = 0.01, hidden_layer_size = 4, upper_boundary = 0.000001, num_iterations = 5000





