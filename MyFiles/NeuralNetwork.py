import numpy as numberArray

# Used in the output layer.
def sigmoidFunction(x):
    value = (1 / (1 + numberArray.exp(-x)))
    return value
# Initializing weights and biases for the layers of the NN.
def parameters(x , y, sizeInHidden):
    numberArray.random.seed(8)
    inputLayerSize = x.shape[0]
    outputLayerSize = y.shape[0]
    weight1 = (numberArray.random.randn(sizeInHidden, inputLayerSize)) * (numberArray.sqrt(1 / inputLayerSize))
    bias1 = numberArray.zeros((sizeInHidden, 1))
    weight2 = (numberArray.random.randn(outputLayerSize, sizeInHidden)) * (numberArray.sqrt(1 / sizeInHidden))
    bias2 = numberArray.zeros((outputLayerSize, 1))
    return weight1, bias1, weight2, bias2











