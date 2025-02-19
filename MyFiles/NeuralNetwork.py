import numpy as numberArray

# An activation function and is used in the output layer. This outputs
# a prediction probability value between 0 and 1.
def sigmoidFunction(x):
    value = (1 / (1 + numberArray.exp(-x)))
    return value

# Initializing weights and biases for the layers of the neural network.
def initializeParameters(x , y, sizeInHidden):
    numberArray.random.seed(8)
    inputLayerSize = x.shape[0]
    outputLayerSize = y.shape[0]
    weight1 = (numberArray.random.randn(sizeInHidden, inputLayerSize)) * (numberArray.sqrt(1 / inputLayerSize))
    bias1 = numberArray.zeros((sizeInHidden, 1))
    weight2 = (numberArray.random.randn(outputLayerSize, sizeInHidden)) * (numberArray.sqrt(1 / sizeInHidden))
    bias2 = numberArray.zeros((outputLayerSize, 1))
    parametersTuple = (weight1, bias1, weight2, bias2)
    return parametersTuple

# The neural network takes the input and passes it through its layers which undergoes
# linear transformations followed by non-linear activations to output predictions.
def forwardPropagation(x, parametersTuple):
    result1 = numberArray.dot(parametersTuple[0] , x) + parametersTuple[1]
    result1Activated = numberArray.tanh(result1)
    result2 = numberArray.dot(parametersTuple[2], result1Activated) + parametersTuple[3]
    result2Activated = sigmoidFunction(result2)
    forward_Prop = {"R1" : result1, "R2" : result2, "R1Activated" : result1Activated, "R2Activated" : result2Activated}
    return forward_Prop

# To measure the size of the error between the neural network's
# prediction value and the actual value.
def minimizeCostFunction(foward_Prop, trueLabels):
    accumulatedCost = 0
    length = len(trueLabels)
    for i in range(length):
        predictedValue = foward_Prop["R2Activated"][i]
        actualValue = trueLabels[i]
        cost = (actualValue * numberArray.log(predictedValue)) + ((1 - actualValue) * numberArray.log(1 - predictedValue))
        accumulatedCost += cost
    averageCostValue = -(accumulatedCost / length)
    return averageCostValue

def main():
    print("Welcome to the Neural Network Project.")
main()











