import numpy as number_array                        # Numerical computations/Basic data processing
from sklearn.datasets import load_iris              # Loads Iris Dataset
from sklearn.model_selection import train_test_split # Splits the Dataset for training and testing
from sklearn.datasets import fetch_openml            # Fetches MNIST dataset from OpenML website (need internet connection)
import matplotlib.pyplot as plt                      # For graphs
from numpy.lib.stride_tricks import sliding_window_view # Basic data processing - used on CNN
import os                                               # To check/save local copy of MNIST dataset in current directory

# An activation function that handles multi-class classifications, outputting a probability
# distribution where each entry corresponds to a probability for a specific class.
def softmax_cnn(x):
    maximum = number_array.max(x, axis = 1, keepdims = True)
    adjusted = number_array.subtract(x, maximum)
    holder = number_array.exp(adjusted)
    total = number_array.sum(holder, axis = 1, keepdims = True)
    return holder / total

# An activation function that handles multi-class classifications, outputting a probability
# distribution where each entry corresponds to a probability for a specific class.
def softmax(x):
    maximum = number_array.max(x, axis = 0, keepdims = True)
    adjusted = number_array.subtract(x, maximum)
    holder = number_array.exp(adjusted)
    total = number_array.sum(holder, axis = 0, keepdims = True)
    return holder / total

# An activation function that outputs the given input directly if its positive, otherwise it outputs
# the value 0 instead.
def relu(x):
    value = number_array.maximum(0, x)
    return value

# An activation function and is used in the output layer. This outputs
# a prediction probability value between 0 and 1.
def sigmoid_function(x):
    value = (1 / (1 + number_array.exp(-x)))
    return value

# Initializing weights and biases for the layers of the neural network (binary classification).
# Uses Xavier Initialization.
def initialize_parameters(x, y, size_in_hidden):
    number_array.random.seed(8)
    input_layer_size = x.shape[0]
    output_layer_size = y.shape[0]
    weight1 = (number_array.sqrt(1 / input_layer_size)) * (number_array.random.randn(size_in_hidden, input_layer_size))
    bias1 = number_array.zeros((size_in_hidden, 1))
    weight2 = (number_array.sqrt(1 / size_in_hidden)) * (number_array.random.randn(output_layer_size, size_in_hidden))
    bias2 = number_array.zeros((output_layer_size, 1))
    parameters_tuple = (weight1, bias1, weight2, bias2)
    return parameters_tuple

# The neural network takes the input and passes it through its layers which undergoes
# linear transformations followed by non-linear activations to output predictions (binary classification).
def forward_propagation(x, parameters_tuple):
    result1 = number_array.dot(parameters_tuple[0], x) + parameters_tuple[1]
    result1_activated = number_array.tanh(result1)
    result2 = number_array.dot(parameters_tuple[2], result1_activated) + parameters_tuple[3]
    result2_activated = sigmoid_function(result2)
    forward_prop = {"R1" : result1, "R2" : result2, "R1Activated" : result1_activated, "R2Activated" : result2_activated}
    return forward_prop

# Uses optimized vector operation to measure the size of the error between the neural network's
# prediction value and the actual value (binary classification).
def loss_function_vector(forward_prop, true_labels):
    n = true_labels.shape[1]
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

# To update parameters with the use of gradient descent formula: x(t + 1) = x(t) - learning_rate * f'(x(t))
# to the neural network.
def gradient_descent(parameters_tuple, backward_prop, learning_rate):
    updated_weight1 = parameters_tuple[0] - learning_rate * backward_prop["DW1"]
    updated_bias1 = parameters_tuple[1] - learning_rate * backward_prop["DBias1"]
    updated_weight2 = parameters_tuple[2] - learning_rate * backward_prop["DW2"]
    updated_bias2 = parameters_tuple[3] - learning_rate * backward_prop["DBias2"]
    updated_weights_biases = (updated_weight1, updated_bias1, updated_weight2, updated_bias2)
    return updated_weights_biases

# To train our neural network model, updates accordingly if predicted value is a far estimate from the
# true/actual label (i.e. learning from the mistakes it makes) (binary classification).
def training_neural_network(x, y, learning_rate, hidden_size, upper_boundary, num_iterations):
    param = initialize_parameters(x, y, hidden_size)
    cost = []
    for i in range(num_iterations):
        updated_forward_prop = forward_propagation(x, param)
        current_cost = loss_function_vector(updated_forward_prop, y)
        cost.append(current_cost)
        if i > 0 :
            if upper_boundary > abs(cost[-1] - cost[-2]):
                break
        gradients = backward_propagation(x, y, param, updated_forward_prop)
        param = gradient_descent(param, gradients, learning_rate)
    return cost, param

# Method to write and save output of the model to a text file.
def output_text(list_collector, experimental_cost, case):
    entry = 0
    if case == 1:
        output_file = "Output_FF_Binary_Classification.txt"
    elif case == 2:
        output_file = "Output_CNN_Multi_Class_Classification.txt"
    else:
        output_file = "Output_FF_Multi_Class_Classification.txt"
    with open(output_file, "w") as output:
        for i in list_collector :
            entry += 1
            string_holder = "Current loss in record " + str(entry) + ": " + str(i) + "\n"
            output.write(string_holder)
        output.write("Experimental loss: " + str(experimental_cost) + "\n")

# To plot and save the cost curve of the neural network through binary classification.
def plot_save_loss_curve(cost_record_holder, learning_rate, hidden_layer, up, n_iterations):
    plt.plot(cost_record_holder)
    plt.title("Loss Curve for Iris Dataset - Binary Classification", color = "red")
    plt.xlabel("Number of Iterations", color = "blue")
    plt.ylabel("Loss Size", color = "green")
    plt.legend(["Training Loss"], loc = "upper left", bbox_to_anchor = (0.1, 0.95))
    container = (f"Learning Rate = {learning_rate}\n"
                 f"Hidden Size = {hidden_layer}\n"
                 f"Upper Boundary = {up}\n"
                 f"# of Iterations = {n_iterations}")
    plt.gca().text(0.95, 0.95, container, transform = plt.gca().transAxes, fontsize = 12, verticalalignment
    = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'square, pad = 0.4', facecolor = 'white',
                                                        alpha = 0.9))
    plt.savefig("[Iris Dataset - Binary] Loss.png")
    plt.show()

# To plot and save the binary classification task of the neural network regarding the implementation
# of the Iris Dataset.
def plot_save_binary_classification_iris_dataset(exp_forward_values, y_test_holder):
    predicted_outputs = exp_forward_values["R2Activated"]
    predicted_binary = (predicted_outputs >= 0.5).astype(int)
    comparison_holder = predicted_binary == y_test_holder
    counter = number_array.sum(comparison_holder)
    sample_size = y_test_holder.size
    accuracy = float(counter / sample_size)
    print("Accuracy: ", str(accuracy))
    plt.figure(figsize = (10,10))
    plt.xticks(range(y_test_holder.shape[1]))
    plt.scatter(range(y_test_holder.shape[1]), y_test_holder.flatten(), label = "= Actual (0 - Iris setosa: 1 - Iris versicolor)", color = "green", marker = "o")
    plt.scatter(range(y_test_holder.shape[1]), predicted_binary.flatten(), label = "= Predicted", color = "red", marker = "x")
    plt.xlabel("Data Test Samples", color = "blue")
    plt.ylabel("Output - Types Of Iris Flower", color = "blue")
    plt.title(f"Actual vs. Predicted Values\n Iris Dataset - Binary Classification\n Accuracy: {accuracy:.4f}", color = "blue")
    plt.legend()
    plt.savefig("[Iris Dataset - Binary] Actual vs. Predicted values.png")
    plt.show()

# Shows how well the neural network does on training with sample data sets based on binary classification.
# From the initial cost, should keep decreasing as model learns from mistakes - archived on list of cost
# records. Then finally, an experimental cost to see how well the model does on unseen/new data it hasn't
# encountered yet. The final computed solution should converge close to if not be the same value as the final
# cost in the cost record.
def binary_classification():
    case = 1
    data = load_iris()
    masking = data.target != 2
    x = data.data[masking]
    y = data.target[masking]
    [x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size = 0.2, random_state = 8)
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    learning_rate = float(input("Enter the learning rate: "))
    hidden_layer_size = int(input("Enter the number of neurons in the hidden layer: "))
    upper_boundary = float(input("Enter the upper boundary: "))
    num_iterations = int(input("Enter the number of iterations: "))

    # Suggested parameter test values.
    # learning_rate = 0.4
    # hidden_layer_size = 4
    # upper_boundary = 0.0001
    # num_iterations = 4500

    [cost_record, trained_params] = training_neural_network(x_train, y_train, learning_rate, hidden_layer_size, upper_boundary, num_iterations)
    experimental_forward_outputs = forward_propagation(x_test, trained_params)
    experimental_cost = loss_function_vector(experimental_forward_outputs, y_test)
    entry = 0
    for i in cost_record:
        entry += 1
        print(f"Current loss in record {entry}: ", i)
    print("Experimental loss: ", experimental_cost)
    output_text(cost_record, experimental_cost,case)
    plot_save_loss_curve(cost_record, learning_rate, hidden_layer_size, upper_boundary, num_iterations)
    plot_save_binary_classification_iris_dataset(experimental_forward_outputs, y_test)

# To count the number of classes or categories and set it to the output layer size.
def count_num_classes(y):
    set_of_classes = set()
    y = y.flatten()
    for i in y:
        set_of_classes.add(i)
    size = len(set_of_classes)
    return size

# Initializing weights and biases for the layers of the neural network (multi-class classification).
# Uses Xavier Initialization.
def initialize_parameters_multiclass_classification(x, number_of_classes, size_in_hidden):
    number_array.random.seed(8)
    input_layer_size = x.shape[0]
    output_layer_size = number_of_classes
    weight1 = (number_array.sqrt(1 / input_layer_size)) * (number_array.random.randn(size_in_hidden, input_layer_size))
    bias1 = number_array.zeros((size_in_hidden, 1))
    weight2 = (number_array.sqrt(1 / size_in_hidden)) * (number_array.random.randn(output_layer_size, size_in_hidden))
    bias2 = number_array.zeros((output_layer_size, 1))
    parameters_tuple = (weight1, bias1, weight2, bias2)
    return parameters_tuple


# The neural network takes the input and passes it through its layers which undergoes
# linear transformations followed by non-linear activations to output predictions (multi-class classification).
def forward_propagation_multi_class(x, parameters_tuple):
    result1 = number_array.dot(parameters_tuple[0], x) + parameters_tuple[1]
    result1_activated = number_array.tanh(result1)
    result2 = number_array.dot(parameters_tuple[2], result1_activated) + parameters_tuple[3]
    result2_activated = softmax(result2)
    forward_prop = {"R1" : result1, "R2" : result2, "R1Activated" : result1_activated, "R2Activated" : result2_activated}
    return forward_prop

# Uses optimized vector operation to measure the size of the error between the neural network's
# prediction value and the actual value (multi-class classification).
def loss_function_vector_multi_class(forward_prop, true_labels):
    n = true_labels.shape[1]
    numeric_stability = 1e-8
    log_holder_predictions = number_array.log(forward_prop["R2Activated"] + numeric_stability)
    cost = log_holder_predictions * true_labels
    total_cost = -(1/n) * number_array.sum(cost)
    return total_cost

# To train our neural network model, updates accordingly if predicted value is a far estimate from the
# true/actual label (i.e. learning from the mistakes it makes) (multi-class classification).
def training_neural_network_multi_class(x, y, learning_rate, hidden_size, upper_boundary, num_iterations, number_of_classes):
    param = initialize_parameters_multiclass_classification(x, number_of_classes, hidden_size)
    cost = []
    for i in range(num_iterations):
        updated_forward_prop = forward_propagation_multi_class(x, param)
        current_cost = loss_function_vector_multi_class(updated_forward_prop, y)
        cost.append(current_cost)
        if i > 0 :
            if upper_boundary > abs(cost[-1] - cost[-2]):
                break
        gradients = backward_propagation(x, y, param, updated_forward_prop)
        param = gradient_descent(param, gradients, learning_rate)
    return cost, param

# One Hot encoding method to convert categorical variables into binary format. This means, it synthesizes
# rows for each class, where 1 indicates that the class is present while the
# value of 0 signifies that it's not present. Data samples in this example are the columns.
def convert_one_hot_encoding(y, number_classes):
    samples = y.shape[1]
    container = number_array.zeros((number_classes, samples))
    for i in range(samples):
        label_holder = y[0, i]
        container[label_holder, i] = 1
    return container

# To plot and save the cost curve of the neural network through multi-class classification.
def plot_save_loss_curve_multi_class(cost_record_holder, learning_rate, hidden_layer, up, n_iterations):
    plt.plot(cost_record_holder)
    plt.title("Loss Curve for Iris Dataset - Multi-class Classification", color = "red")
    plt.xlabel("Number of Iterations", color = "blue")
    plt.ylabel("Loss Size", color = "green")
    plt.legend(["Training Loss"], loc = "upper left", bbox_to_anchor = (0.1, 0.95))
    container = (f"Learning Rate = {learning_rate}\n"
                 f"Hidden Size = {hidden_layer}\n"
                 f"Upper Boundary = {up}\n"
                 f"# of Iterations = {n_iterations}")
    plt.gca().text(0.95, 0.95, container, transform = plt.gca().transAxes, fontsize = 12, verticalalignment
    = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'square, pad = 0.4', facecolor = 'white',
                                                        alpha = 0.9))
    plt.savefig("[Iris Dataset - Multi-class] Loss.png")
    plt.show()

# To plot and save the multi-class classification task of the neural network regarding the implementation
# of the Iris Dataset.
def plot_save_multi_class_classification_iris_dataset(exp_forward_values, y_test_holder):
    predicted_outputs = exp_forward_values["R2Activated"]
    predicted_class = number_array.argmax(predicted_outputs, axis = 0)
    real_class = number_array.argmax(y_test_holder, axis = 0)
    counter = 0
    sample_size = len(real_class)
    for i in range(sample_size):
        if predicted_class[i] == real_class[i]:
            counter += 1
    accuracy = float(counter / sample_size)
    print("Accuracy: ", str(accuracy))
    plt.figure(figsize = (10,10))
    plt.xticks(range(y_test_holder.shape[1]))
    plt.scatter(range(y_test_holder.shape[1]), real_class.flatten(), label = "= Actual (0 - Iris setosa: 1 - Iris versicolor: 2 - Iris virginica)", color = "green", marker = "o")
    plt.scatter(range(y_test_holder.shape[1]), predicted_class.flatten(), label = "= Predicted", color = "red", marker = "x")
    plt.xlabel("Data Test Samples", color = "blue")
    plt.ylabel("Output - Types Of Iris Flower", color = "blue")
    plt.title(f"Actual vs. Predicted Values\n Iris Dataset - Multi-class Classification\n Accuracy: {accuracy:.4f}", color = "blue")
    plt.legend(loc='center', bbox_to_anchor = (0.5, .25))
    plt.savefig("[Iris Dataset - Multi-class] Actual vs. Predicted values.png")
    plt.show()

# {Test Multi-class classification using full Iris dataset}.
# Shows how well the neural network does on training with sample data sets based on multi-class classification.
# From the initial cost, should keep decreasing as model learns from mistakes - archived on list of cost
# records. Then finally, an experimental cost to see how well the model does on unseen/new data it hasn't
# encountered yet. The final computed solution should converge close to if not be the same value as the final
# cost in the cost record.
def multi_class_classification():
        case = 3
        data = load_iris()
        x = data.data
        y = data.target
        holder = count_num_classes(y)
        [x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size = 0.2, random_state = 8)
        x_train = x_train.T
        x_test = x_test.T
        y_train = y_train.reshape(1, -1)
        y_test = y_test.reshape(1, -1)
        y_train = convert_one_hot_encoding(y_train, holder)
        y_test = convert_one_hot_encoding(y_test, holder)

        learning_rate = float(input("Enter the learning rate: "))
        hidden_layer_size = int(input("Enter the number of neurons in the hidden layer: "))
        upper_boundary = float(input("Enter the upper boundary: "))
        num_iterations = int(input("Enter the number of iterations: "))

        # Suggested parameter test values.
        # learning_rate = 0.4
        # hidden_layer_size = 4
        # upper_boundary = 0.000001
        # num_iterations = 5000

        [cost_record, trained_params] = training_neural_network_multi_class(x_train, y_train, learning_rate, hidden_layer_size, upper_boundary, num_iterations, holder)
        experimental_forward_outputs = forward_propagation_multi_class(x_test, trained_params)
        experimental_cost = loss_function_vector_multi_class(experimental_forward_outputs, y_test)
        entry = 0
        for i in cost_record:
            entry += 1
            print(f"Current loss in record {entry}: ", i)
        print("Experimental loss: ", experimental_cost)
        output_text(cost_record, experimental_cost, case)
        plot_save_loss_curve_multi_class(cost_record, learning_rate, hidden_layer_size, upper_boundary, num_iterations)
        plot_save_multi_class_classification_iris_dataset(experimental_forward_outputs, y_test)

# Initializes the weights and biases of the layers of the Neural Network. The layers include 2
# convolutional layers, a dense layer, and an output layer. Convolutional Layer 1 has 32 neurons,
# Convolutional Layer 2 has 64 neurons, Dense Layer has 128 neurons, and Output Layer has
# 10 neurons. Uses He (Kaiming) Initialization for the weights and biases are initialized to 0.
# Returns the layers' weights and their corresponding biases.
def initialize_parameters_cnn(image, number_classes):
    channel, height, width, = image
    filter_container1 = 32
    filter_container2 = 64
    hidden_size = 128
    output_size = number_classes
    kernel1 = 3
    kernel2 = 3
    # convolutional layer 1
    fan_input1 = pow(kernel1, 2) * channel
    # convolutional layer 2
    fan_input2 = pow(kernel2, 2) * filter_container1
    # Dense Layer
    conv1_out = height - kernel1 + 1
    pool_1_out = conv1_out // 2
    conv2_out = pool_1_out - kernel2 + 1
    pool_2_out = conv2_out // 2
    size_flat = pool_2_out * filter_container2 * pool_2_out
    connect_fan_input = size_flat
    # Output Layer
    output_fan_i = hidden_size
    # Weights and Biases Conv 1
    conv1_shape = (filter_container1, channel, kernel1, kernel1)
    conv1_scaling = number_array.sqrt(2/ fan_input1)
    weight_conv1 = number_array.random.normal(size=conv1_shape) * conv1_scaling
    bias_conv1 = number_array.zeros((filter_container1, 1))
    # Weights and Biases Conv 2
    conv2_shape = (filter_container2, filter_container1, kernel2, kernel2)
    conv2_scaling = number_array.sqrt(2/ fan_input2)
    weight_conv2 = number_array.random.normal(size=conv2_shape) * conv2_scaling
    bias_conv2 = number_array.zeros((filter_container2, 1))
    # Dense Layer
    dense_shape = (hidden_size, size_flat)
    dense_scaling = number_array.sqrt(2 / connect_fan_input)
    weight_dense = number_array.random.normal(size=dense_shape) * dense_scaling
    bias_dense = number_array.zeros((hidden_size, 1))
    # Output Layer
    output_shape = (output_size, hidden_size)
    output_scaling = number_array.sqrt(2 / output_fan_i)
    weight_output = number_array.random.normal(size=output_shape) * output_scaling
    bias_output = number_array.zeros((output_size, 1))
    parameters_tuple = (weight_conv1, bias_conv1, weight_conv2, bias_conv2, weight_dense,
                        bias_dense,weight_output, bias_output)
    return parameters_tuple

# This method converts a given 4D input, containing batch size, height, width, and channel,
# with given filter weights and biases into a convolutional output. This method also returns the finalized
# feature maps.
def convert_2_d(inputs, weight, biases):
    batch_num = inputs.shape[0]
    height = inputs.shape[1]
    width_input = inputs.shape[2]
    channel = inputs.shape[3]
    filter_size = weight.shape[0]
    kernels = weight.shape[2]
    output_1 = 1 + height - kernels
    output_2 = 1 + width_input - kernels
    patch_holder1 = sliding_window_view(inputs, window_shape = (kernels, kernels), axis = (1, 2))
    patch_holder2 = number_array.reshape(patch_holder1, (batch_num * output_1 * output_2, channel * pow(kernels, 2)))
    flat_struc = number_array.reshape(weight, (filter_size, channel * pow(kernels, 2)))
    container_res = number_array.dot(patch_holder2, flat_struc.T)
    final = number_array.reshape(container_res, (batch_num, output_1, output_2, filter_size))
    final += biases.T
    return final

# This method performs a 2 by 2-max pooling to an input tensor that contains a shape of (batch size,
# height, width, and channel), and it also cuts the height and width in half. This returns the pooled tensor.
def max_pooling(temp):
    batch_num = temp.shape[0]
    height = temp.shape[1]
    width = temp.shape[2]
    channel = temp.shape[3]
    trim1 = height // 2
    trim2 = width // 2
    temp = temp[:, :trim1 * 2, :trim2 * 2, :]
    output_reshaped = number_array.reshape(temp, (batch_num, trim1, 2, trim2, 2, channel))
    output_pooled = number_array.max(output_reshaped, axis = (2, 4))
    return output_pooled

# Compute and return a constant value.
def area_size(x):
    holder = x * 2 + 2
    return holder

# This method flattens the input tensor with shape (batch size, height, width, channel) into a shaped
# structure of (batch size, channel * height * width).
def flatten_structure(temp):
    batch_num = temp.shape[0]
    height = temp.shape[1]
    width = temp.shape[2]
    channel = temp.shape[3]
    holder = channel * height * width
    result = temp.reshape(batch_num, holder)
    return result

# This performs a forward pass for the CNN model. This also performs a series of transformations for
# two blocks such as (Convolution 3 X 3, Relu (Non-linear activation function), Max Pooling) -> 2 Convoluted Layers.
# Then this flattens the resulting structure, the Dense Layers go through the Relu, and finally the Output Layer
# is pushed through the Softmax function to get the predicted output. It returns the following as a dictionary:
# feature maps and activations (converted_conv1, activation_1, pooling_1, converted_conv2, activation_2, pooling_2),
# flattened vector (struc_flat), the dense layer's pre-activation and activation result (converted_dense, activation_3),
# and finally the logits and probabilities generated by the Softmax function (converted_output, activation_4).
def forward_prop_cnn(x, parameters_tuple):
    converted_conv1 = convert_2_d(x, parameters_tuple[0], parameters_tuple[1])
    activation_1 = relu(converted_conv1)
    pooling_1 = max_pooling(activation_1)
    converted_conv2 = convert_2_d(pooling_1, parameters_tuple[2], parameters_tuple[3])
    activation_2 = relu(converted_conv2)
    pooling_2 = max_pooling(activation_2)
    struc_flat = flatten_structure(pooling_2)
    transpose_dense_weight = parameters_tuple[4].T
    converted_dense = parameters_tuple[5].T + number_array.dot(struc_flat, transpose_dense_weight)
    activation_3 = relu(converted_dense)
    transpose_output_weight = parameters_tuple[6].T
    converted_output = parameters_tuple[7].T + number_array.dot(activation_3, transpose_output_weight)
    activation_4 = softmax_cnn(converted_output)
    forward_prop = {"inputs" : x, "converted_conv1" : converted_conv1, "activation_1": activation_1,
               "pooling_1": pooling_1, "converted_conv2": converted_conv2, "activation_2": activation_2,
               "pooling_2": pooling_2, "struc_flat": struc_flat, "converted_dense": converted_dense,
               "activation_3": activation_3, "converted_output": converted_output, "activation_4": activation_4}
    return forward_prop

# To compute the average cross-entropy loss between Softmax outputs (predicted values) and one-hot labels (actual labels).
# The size of error or distance in between the predicted values and actual labels. This also returns that computed loss value.
def loss_cnn(activation, y):
    number_size = activation.shape[0]
    epsilon = 1e-8
    prob_container = number_array.log(epsilon + activation)
    cost_holder = (-1 * number_array.sum(prob_container * y)) / number_size
    return cost_holder

# This performs a backwards pass to the CNN model. This also computes all the gradients for the weights and biases of the
# layers such as Convolutional Layer 1, Convolutional Layer 2, Dense Layer, and the Output Layer. First
# at the Output Layer the Softmax function is used which is followed by the cross-entropy loss
# to be able to compute for d_weight_output and d_bias_output. Then, at the Dense Layer, the result
# from the forward Relu activation is used as a parameter for the derivative of the Relu function to compute
# for d_weight_dense and d_bias_dense which then reshapes using backwards pooling function.
# Then at the two Convolutional Layers the forward Relu activation and the pooling results are passed
# as parameters to the derivative of the Relu function which then propagates to compute for d_weight_conv2, d_bias_conv2,
# d_weight_conv1, and d_bias_conv1. Finally, this method returns all the gradients for the weights and biases
# of the layers of the Neural Network as a dictionary.
def backward_prop_cnn(forward_prop, parameters_tuple, y):
    number_size = forward_prop["inputs"].shape[0]
    d_holder4 = forward_prop["activation_4"] - y
    d_bias_output = (number_array.sum(d_holder4, axis = 0, keepdims = True) / number_size).T
    d_weight_output = number_array.dot(d_holder4.T, forward_prop["activation_3"]) / number_size
    d_output_layer = number_array.dot(d_holder4, parameters_tuple[6])
    d_holder3 = d_relu(forward_prop["converted_dense"], d_output_layer)
    d_weight_dense = number_array.dot(d_holder3.T, forward_prop["struc_flat"]) / number_size
    d_bias_dense = (number_array.sum(d_holder3, axis = 0, keepdims = True) / number_size).T
    d_dense_layer = number_array.dot(d_holder3, parameters_tuple[4])
    d_pooling_2 = number_array.reshape(d_dense_layer, forward_prop["pooling_2"].shape)
    d_a2 = back_pooling(forward_prop["activation_2"], d_pooling_2)
    d_holder2 = d_relu(forward_prop["converted_conv2"], d_a2)
    gradient_temp2 = backward_conv(parameters_tuple[2], forward_prop["pooling_1"], d_holder2)
    d_volume2 = gradient_temp2["volume_gradient"]
    d_bias_conv2 = gradient_temp2["sum_holder"].T
    d_weight_conv2 = gradient_temp2["collection_w_res"]
    d_a1 = back_pooling(forward_prop["activation_1"], d_volume2)
    d_holder1 = d_relu(forward_prop["converted_conv1"], d_a1)
    gradient_temp1 = backward_conv(parameters_tuple[0], forward_prop["inputs"], d_holder1)
    d_volume1 = gradient_temp1["volume_gradient"]
    d_bias_conv1 = gradient_temp1["sum_holder"].T
    d_weight_conv1 = gradient_temp1["collection_w_res"]
    result = {"d_volume1": d_volume1, "d_weight_conv1": d_weight_conv1, "d_bias_conv1": d_bias_conv1,
              "d_weight_conv2": d_weight_conv2, "d_bias_conv2": d_bias_conv2, "d_weight_dense": d_weight_dense,
              "d_bias_dense": d_bias_dense, "d_weight_output": d_weight_output, "d_bias_output": d_bias_output}
    return result

# This method backpropagates through a 2 X 2 max-pool, which directs collected pooled gradients back to their
# original locations in the input tensor.
def back_pooling(h_activation_2, d_pooling_2):
    batch_num_h_activation_2 = h_activation_2.shape[0]
    channel_h_activation_2 = h_activation_2.shape[3]
    batch_num_dp2 = d_pooling_2.shape[0]
    height_dp2 = d_pooling_2.shape[1]
    width_dp2 = d_pooling_2.shape[2]
    channel_dp2 = d_pooling_2.shape[3]
    crop_height = h_activation_2[:, :height_dp2 * 2, :width_dp2 * 2, :]
    scope_a2 = crop_height.reshape(batch_num_h_activation_2, height_dp2, 2, width_dp2, 2, channel_h_activation_2)
    maximum_holder = number_array.max(scope_a2, axis = (2, 4), keepdims = True)
    masking = (scope_a2 == maximum_holder)
    d_pooling = number_array.reshape(d_pooling_2, (batch_num_dp2, height_dp2, 1, width_dp2, 1, channel_dp2))
    d_a2_mask_result = d_pooling * masking
    cropped_da2 = d_a2_mask_result.reshape(batch_num_h_activation_2, height_dp2 * 2, width_dp2 * 2, channel_h_activation_2)
    d_final = number_array.zeros_like(h_activation_2)
    d_final[:, :height_dp2 * 2, :width_dp2 * 2, :] = cropped_da2
    return d_final

# This method backpropagates through the Convolutional Layer. This first computes the weight gradients by
# multiplying the gradient with the corresponding input patches. Then computes for the bias gradients
# by summing up the gradient values, and finally computes for gradient of the input volume. This also returns
# the computed weight gradients, bias gradients, and gradient for the input volume.
def backward_conv(weight, input_holder, gradient_holder):
    filters_weight = weight.shape[0]
    channel_weight = weight.shape[1]
    kernels_weight = weight.shape[2]
    batch_num_in = input_holder.shape[0]
    height_gr = gradient_holder.shape[1]
    width_gr = gradient_holder.shape[2]
    container_1 = batch_num_in * height_gr * width_gr
    container_2 = channel_weight * pow(kernels_weight, 2)
    collector_patch = sliding_window_view(input_holder, window_shape = (kernels_weight, kernels_weight), axis = (1, 2))
    reformed_patch = number_array.reshape(collector_patch, (container_1, container_2))
    flat_gradient = number_array.reshape(gradient_holder, (container_1, filters_weight))
    result1 = number_array.dot(flat_gradient.T, reformed_patch) / batch_num_in
    result2 = number_array.reshape(result1, (filters_weight, channel_weight, kernels_weight, kernels_weight))
    container_sum = number_array.sum(flat_gradient, axis = 0, keepdims = True) / batch_num_in
    reformed_w = number_array.reshape(weight, (filters_weight, container_2))
    w_result1 = number_array.dot(flat_gradient, reformed_w)
    w_result2 = number_array.reshape(w_result1, (batch_num_in, height_gr, width_gr, channel_weight, kernels_weight, kernels_weight))
    volume_gradient = number_array.zeros_like(input_holder)
    for i in range(kernels_weight):
        for z in range(kernels_weight):
            new_carrier = w_result2[:, :, :, :, i, z]
            volume_gradient[:, i: i + height_gr, z: z + width_gr, :] += new_carrier
    final = {"sum_holder": container_sum,"collection_w_res": result2, "volume_gradient": volume_gradient}
    return final

# Vectorized operation, this method backpropagates through the Relu, and it zeros out the gradients
# where the given input is non-positive i.e (x <= 0).
def d_relu(mat_size, holder):
    masking = mat_size > 0
    d_holder = holder * masking
    return d_holder

# This method is using the gradient descent formula to update the parameters for every mistake:
# Updates the weights and biases for all layers of the Neural Network -> [W_(x + 1) = W_x - learning_rate * gradient].
# This method returns the updated weights and biases for the 2 Convolutional Layers, Dense Layer, and Output Layer.
def gradient_update_cnn(params, gradients, learning_rate):
    u_weight_conv1 = params[0] - learning_rate * gradients["d_weight_conv1"]
    u_bias_conv1 =   params[1] - learning_rate * gradients["d_bias_conv1"]
    u_weight_conv2 = params[2] - learning_rate * gradients["d_weight_conv2"]
    u_bias_conv2 =   params[3] - learning_rate * gradients["d_bias_conv2"]
    u_weight_dense = params[4] - learning_rate * gradients["d_weight_dense"]
    u_bias_dense =   params[5] - learning_rate * gradients["d_bias_dense"]
    u_weight_output = params[6] - learning_rate * gradients["d_weight_output"]
    u_bias_output =   params[7] - learning_rate * gradients["d_bias_output"]
    result = (u_weight_conv1, u_bias_conv1, u_weight_conv2, u_bias_conv2, u_weight_dense, u_bias_dense,
              u_weight_output, u_bias_output)
    return result


# This method trains the Neural Network using a mini batch SGD (Stochastic Gradient Descent), this randomly shuffles data for each iteration,
# and also undergoes a forward pass, backward pass, parameter updates, and keeps an archive of the average loss per batch iteration.
# This method returns the newly trained parameters and the record of losses during training.
def train_cnn(x, y, parameters, learning_rate, num_iterations, batch_number, upper_boundary):
    length = len(x)
    cost = []
    for i in range(num_iterations):
        permutations_holder = number_array.random.permutation(x.shape[0])
        new_x = x[permutations_holder]
        new_y = y[permutations_holder]
        subset_cost = []
        for begin in range(0, length, batch_number):
            last = min(begin + batch_number, length)
            batches_x = new_x[begin : last]
            batches_y = new_y[begin : last]
            taker = forward_prop_cnn(batches_x, parameters)
            current_cost = loss_cnn(taker["activation_4"], batches_y)
            subset_cost.append(current_cost)
            gradient_holder = backward_prop_cnn(taker, parameters, batches_y)
            parameters = gradient_update_cnn(parameters, gradient_holder, learning_rate)
        average_cost = sum(subset_cost) / len(subset_cost)
        cost.append(average_cost)
        if i > 0 and upper_boundary > abs(cost[-1] - cost[-2]):
            break
    return cost, parameters

# This method plots the loss curve with respect to the number of iterations of training that the
# Neural Network underwent.
def plot_save_loss_curve_cnn(cost_record_holder, learning_rate, n_iter, batch_size, up):
    plt.plot(cost_record_holder)
    plt.title("Loss Curve for MNIST - Multi-class Classification", color = "red")
    plt.xlabel("Number of Iterations/Batch", color = "blue")
    plt.ylabel("Loss Size", color = "green")
    plt.legend(["Training Loss"], loc = "upper left", bbox_to_anchor = (0.1, 0.95))
    container = (f"Learning Rate = {learning_rate}\n"
                 f"# of Iterations = {n_iter}\n"
                 f"Batch Size = {batch_size}\n"
                 f"Upper Boundary = {up}")
    plt.gca().text(0.95, 0.95, container, transform = plt.gca().transAxes, fontsize = 12, verticalalignment
                   = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'square, pad = 0.4', facecolor = 'white',
                    alpha = 0.9))
    plt.savefig("[MNIST - Multi-class] Loss.png")
    plt.show()

# This method creates a scatter plot of all the given test samples along with the classification results
# from the Neural Network (its predicted results) vs. the corresponding actual labels for the associated sample space.
def cnn_scatter_plot(predicted, true):
    prediction_holder = number_array.argmax(predicted["activation_4"], axis = 1)
    actual = number_array.argmax(true, axis = 1)
    length = len(true)
    counter = 0
    for i in range (length):
        if prediction_holder[i] == actual[i]:
            counter += 1
    accuracy = counter / length
    print("Accuracy: ", str(accuracy))
    true_shape = number_array.arange(actual.shape[0])
    plt.figure(figsize = (10,10))
    plt.scatter(true_shape, actual, label = "= Actual", color = "green", marker = "o", alpha = 0.7)
    plt.scatter(true_shape, prediction_holder, label = "= Predicted", color = "red", marker = "x", alpha = 0.7)
    plt.xlabel("Data Test Samples", color = "blue")
    plt.ylabel("Output - Handwritten Digits (0 - 9)", color = "blue")
    plt.title(f"Actual vs. Predicted Values\n MNIST - Multi-class Classification\n Accuracy: {accuracy:.4f}", color = "blue")
    plt.legend()
    plt.savefig("[MNIST - Multi-class] Actual vs. Predicted values.png")
    plt.show()

# Fetches the MNIST dataset from the OpenML website and saves/caches in a local copy afterward.
# If a local copy is already cached in the same directory as the Neural Network source code, then it
# loads the dataset from there and skips the initial fetch.
def cached_mnist_784(file_path):
    if os.path.exists(file_path):
        with number_array.load(file_path) as loaded:
            x_holder = loaded["X"]
            y_holder = loaded["y"]
        print(f"The local file {file_path} that contains the dataset loaded successfully.")
    else:
        print("Retrieving the mnist_784 dataset from OpenML")
        dataset = fetch_openml(name = "mnist_784", version = 1, as_frame = False, parser = 'liac-arff')
        x_holder = dataset.data
        y_holder = dataset.target.astype(int)
        number_array.savez_compressed(file_path, X = x_holder, y = y_holder)
        print(f"The dataset has been saved locally as {file_path}")
    return x_holder, y_holder

# This method uses the CNN build of the Neural Network to do multi-class classification using the MNIST
# dataset.
def do_cnn():
    number_array.random.seed(6)
    case = 2
    file_name = "mnist_784.npz"
    x_begin, y_begin = cached_mnist_784(file_name)
    x = x_begin.reshape(-1, 28, 28, 1)
    x = x.astype(float) / 255.0
    y_res = y_begin.reshape(1, -1)
    y_holder = convert_one_hot_encoding(y_res, 10)
    y = y_holder.T
    [x_train, x_test, y_train, y_test] = train_test_split(x, y, test_size = 0.2, random_state = 24)
    x_train_subset = x_train[:1024]
    y_train_subset = y_train[:1024]
    x_test_subset = x_test[:512]
    y_test_subset = y_test[:512]
    channel = 1
    depth = 28
    width = 28
    learning_rate = float(input("Enter a learning rate: "))
    number_iterations = int(input("Enter the number of iterations: "))
    batch_size = int(input("Enter a batch size: "))
    upper_boundary = float(input("Enter an upper boundary: "))
    parameters = initialize_parameters_cnn((channel, depth, width),  10)
    [cost, trained] = train_cnn(x_train_subset, y_train_subset, parameters, learning_rate, number_iterations, batch_size, upper_boundary)
    experimental_test = forward_prop_cnn(x_test_subset, trained)
    experimental_cost = loss_cnn(experimental_test["activation_4"], y_test_subset)
    entry = 0
    for i in cost:
        entry += 1
        print(f"Current loss in record {entry}: ", i)
    print("Experimental loss: ", experimental_cost)
    output_text(cost, experimental_cost, case)
    plot_save_loss_curve_cnn(cost, learning_rate, number_iterations, batch_size, upper_boundary)
    cnn_scatter_plot(experimental_test, y_test_subset)

# To run the Neural Network program.
def main():
    while True :
        input_holder = input("For Binary Classification enter 'b' or for Multi-class Classification enter 'm': ")
        if input_holder.lower() == "b":
            print("Binary Classification:")
            print("Suggested Parameters: learning_rate = 0.4, hidden_layer_size = 4, upper_boundary = 0.0001, num_iterations = 200")
            binary_classification()
        else:
            print("Multi-class Classification:")
            input_container = input("For CNN enter 'c' or Feedforward enter 'f': ")
            if input_container.lower() == "c":
                print("Suggested Parameters: learning_rate = 0.33, number_iterations = 30, batch_size = 64, upper_boundary = 0.000001")
                do_cnn()
            else:
                print("Suggested Parameters: learning_rate = 0.01, hidden_layer_size = 4, upper_boundary = 0.000001, num_iterations = 5000")
                multi_class_classification()
        string_input = input("Do you want to keep testing the model? (y/n): ")
        if string_input.lower() != "y" :
            break
main()