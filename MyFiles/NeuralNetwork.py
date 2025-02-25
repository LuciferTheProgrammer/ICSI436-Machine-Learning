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

# To find the derivatives for the weights and biases of the layers of the neural
# network.
def backwardPropagation(x, y, parametersTuple, forward_Prop):
    n  = x.shape[1]
    d_result_2 = forward_Prop["R2Activated"] - y
    d_weight_2 = numberArray.dot(d_result_2, forward_Prop["R1Activated"].T) * (1/n)
    d_bias_2 = numberArray.sum(d_result_2, axis = 1, keepdims = True) * (1/n)
    d_result_1_activated = numberArray.dot(parametersTuple[2].T, d_result_2)
    d_tanh = 1 - (forward_Prop["R1Activated"])**2
    d_result_1 = d_result_1_activated * d_tanh
    d_weight_1 = numberArray.dot(d_result_1, x.T) * (1/n)
    d_bias_1 = numberArray.sum(d_result_1, axis = 1, keepdims = True) * (1/n)
    backward_Prop = {"DW1": d_weight_1, "DW2": d_weight_2, "DBias1": d_bias_1,"DBias2": d_bias_2}
    return backward_Prop
def main():
    print("Welcome to the Neural Network Project.")
main()











