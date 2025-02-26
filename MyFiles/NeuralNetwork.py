import numpy as number_array

# An activation function and is used in the output layer. This outputs
# a prediction probability value between 0 and 1.
def sigmoid_function(x):
    value = (1 / (1 + number_array.exp(-x)))
    return value

# Initializing weights and biases for the layers of the neural network.
def initialize_parameters(x, y, size_in_hidden):
    number_array.random.seed(8)
    input_layer_size = x.shape[0]
    output_layer_size = y.shape[0]
    weight1 = (number_array.random.randn(size_in_hidden, input_layer_size)) * (number_array.sqrt(1 / input_layer_size))
    bias1 = number_array.zeros((size_in_hidden, 1))
    weight2 = (number_array.random.randn(output_layer_size, size_in_hidden)) * (number_array.sqrt(1 / size_in_hidden))
    bias2 = number_array.zeros((output_layer_size, 1))
    parameters_tuple = (weight1, bias1, weight2, bias2)
    return parameters_tuple

# The neural network takes the input and passes it through its layers which undergoes
# linear transformations followed by non-linear activations to output predictions.
def forward_propagation(x, parameters_tuple):
    result1 = number_array.dot(parameters_tuple[0], x) + parameters_tuple[1]
    result1_activated = number_array.tanh(result1)
    result2 = number_array.dot(parameters_tuple[2], result1_activated) + parameters_tuple[3]
    result2_activated = sigmoid_function(result2)
    forward_prop = {"R1" : result1, "R2" : result2, "R1Activated" : result1_activated, "R2Activated" : result2_activated}
    return forward_prop

# Uses loop to measure the size of the error between the neural network's
# prediction value and the actual value.
def cost_function_loop(forward_prop, true_labels):
    accumulated_cost = 0
    length = len(true_labels)
    for i in range(length):
        predicted_value = forward_prop["R2Activated"][i]
        actual_value = true_labels[i]
        cost = (actual_value * number_array.log(predicted_value)) + ((1 - actual_value) * number_array.log(1 - predicted_value))
        accumulated_cost += cost
    average_cost_value = -(accumulated_cost / length)
    return average_cost_value

# Uses optimized vector operation to measure the size of the error between the neural network's
# prediction value and the actual value.
def cost_function_vector(forward_prop, true_labels):
    n = forward_prop["R1Activated"].shape[1]
    log_r2_activated = number_array.log(forward_prop["R2Activated"])
    log_r2_complement = number_array.log(1 - forward_prop["R2Activated"])
    combined_cost = true_labels * log_r2_activated + (1 - true_labels) * log_r2_complement
    total_cost = -(1/n) * number_array.sum(combined_cost)
    return total_cost

# To find the derivatives for the weights and biases of the layers of the neural
# network.
def backward_propagation(x, y, parameters_tuple, forward_prop):
    n  = x.shape[1]
    d_result_2 = forward_prop["R2Activated"] - y
    d_weight_2 = number_array.dot(d_result_2, forward_prop["R1Activated"].T) * (1 / n)
    d_bias_2 = number_array.sum(d_result_2, axis = 1, keepdims = True) * (1 / n)
    d_result_1_activated = number_array.dot(parameters_tuple[2].T, d_result_2)
    d_tanh = 1 - (forward_prop["R1Activated"]) ** 2
    d_result_1 = d_result_1_activated * d_tanh
    d_weight_1 = number_array.dot(d_result_1, x.T) * (1 / n)
    d_bias_1 = number_array.sum(d_result_1, axis = 1, keepdims = True) * (1 / n)
    backward_prop = {"DW1": d_weight_1, "DW2": d_weight_2, "DBias1": d_bias_1,"DBias2": d_bias_2}
    return backward_prop
def main():
    print("Welcome to the Neural Network Project.")
main()











