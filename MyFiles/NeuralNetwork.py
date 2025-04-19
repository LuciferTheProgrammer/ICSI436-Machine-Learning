import numpy as number_array
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt



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
# prediction value and the actual value (binary classification).
def cost_function_vector(forward_prop, true_labels):
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
        current_cost = cost_function_vector(updated_forward_prop, y)
        cost.append(current_cost)
        if i > 0 :
            if upper_boundary > abs(cost[-1] - cost[-2]):
                break
        gradients = backward_propagation(x, y, param, updated_forward_prop)
        param = gradient_descent(param, gradients, learning_rate)
    return cost, param

# Method to write and save output of the model to a text file.
def output_text(list_collector, experimental_cost):
    entry = 0
    with open("Outputs.txt", "w") as output:
        for i in list_collector :
            entry += 1
            string_holder = "Current cost in record " + str(entry) + ": " + str(i) + "\n"
            output.write(string_holder)
        output.write("Experimental cost: " + str(experimental_cost) + "\n")

# To plot and save the cost curve of the neural network through binary classification.
def plot_save_cost_curve(cost_record_holder):
    plt.plot(cost_record_holder)
    plt.title("Cost for Iris Dataset - Binary Classification", color = "red")
    plt.xlabel("Number of Iterations Before Thresh Hold", color = "blue")
    plt.ylabel("Cost/Size of Error Over Time", color = "green")
    plt.savefig("[Iris Dataset - Binary] Cost.png")
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
    plt.ylabel("Output Values", color = "blue")
    plt.title("Actual vs. Predicted Values\n Iris Dataset - Binary Classification", color = "blue")
    plt.legend()
    plt.savefig("[Iris Dataset - Binary] Actual vs. Predicted values.png")
    plt.show()

# Shows how well the neural network does on training with sample data sets based on binary classification.
# From the initial cost, should keep decreasing as model learns from mistakes - archived on list of cost
# records. Then finally, an experimental cost to see how well the model does on unseen/new data it hasn't
# encountered yet. The final computed solution should converge close to if not be the same value as the final
# cost in the cost record.
def binary_classification():
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
        experimental_cost = cost_function_vector(experimental_forward_outputs, y_test)
        entry = 0
        for i in cost_record:
            entry += 1
            print(f"Current cost in record {entry}: ", i)
        print("Experimental cost: ", experimental_cost)
        output_text(cost_record, experimental_cost)
        plot_save_cost_curve(cost_record)
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
def cost_function_vector_multi_class(forward_prop, true_labels):
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
        current_cost = cost_function_vector_multi_class(updated_forward_prop, y)
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
def plot_save_cost_curve_multi_class(cost_record_holder):
    plt.plot(cost_record_holder)
    plt.title("Cost for Iris Dataset - Multi-class Classification", color = "red")
    plt.xlabel("Number of Iterations Before Thresh Hold", color = "blue")
    plt.ylabel("Cost/Size of Error Over Time", color = "green")
    plt.savefig("[Iris Dataset - Multi-class] Cost.png")
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
    plt.ylabel("Output Values", color = "blue")
    plt.title("Actual vs. Predicted Values\n Iris Dataset - Multi-class Classification", color = "blue")
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
        experimental_cost = cost_function_vector_multi_class(experimental_forward_outputs, y_test)
        entry = 0
        for i in cost_record:
            entry += 1
            print(f"Current cost in record {entry}: ", i)
        print("Experimental cost: ", experimental_cost)
        output_text(cost_record, experimental_cost)
        plot_save_cost_curve_multi_class(cost_record)
        plot_save_multi_class_classification_iris_dataset(experimental_forward_outputs, y_test)

# Start of CNN implementation for Neural Network.
def initialize_parameters_cnn(image, number_classes):
    container, depth, diameter, = image
    filter_container1 = 32
    filter_container2 = 64
    hidden_size = 128
    output_size = number_classes
    kernel1 = 3
    kernel2 = 3
    # convolutional layer 1
    fan_input1 = pow(kernel1, 2) * container
    fan_output1 = pow(kernel1, 2) * filter_container1
    # convolutional layer 2
    fan_input2 = pow(kernel2, 2) * filter_container1
    fan_output2 = pow(kernel2, 2) * filter_container2
    # Dense Layer
    comb_depth = depth // 4
    comb_diameter = diameter // 4
    size_flat = comb_diameter * filter_container2 * comb_depth
    connect_fan_output = hidden_size
    connect_fan_input = size_flat
    # Output Layer
    output_fan_i = hidden_size
    output_fan_o = output_size
    # Weights and Biases Conv 1
    conv1_shape = (filter_container1, container, kernel1, kernel1)
    conv1_scaling = number_array.sqrt(2/ fan_input1)
    weight_conv1 = number_array.random.normal(conv1_shape) * conv1_scaling
    bias_conv1 = number_array.zeros((filter_container1, 1))
    # Weights and Biases Conv 2
    conv2_shape = (filter_container2, filter_container1, kernel2, kernel2)
    conv2_scaling = number_array.sqrt(2/ fan_input2)
    weight_conv2 = number_array.random.normal(conv2_shape) * conv2_scaling
    bias_conv2 = number_array.zeros((filter_container2, 1))
    # Dense Layer
    dense_shape = (hidden_size, size_flat)
    dense_scaling = number_array.sqrt(2 / connect_fan_input)
    weight_dense = number_array.random.normal(dense_shape) * dense_scaling
    bias_dense = number_array.zeros((hidden_size, 1))
    # Output Layer
    output_shape = (output_size, hidden_size)
    output_scaling = number_array.sqrt(2 / output_fan_i)
    weight_output = number_array.random.normal(output_shape) * output_scaling
    bias_output = number_array.zeros((output_size, 1))
    parameters_tuple = (weight_conv1, bias_conv1, weight_conv2, bias_conv2, weight_dense,
                        bias_dense,weight_output, bias_output)
    return parameters_tuple

# inputs (holder1, holder2, holder3, holder4)
# weights (container1, container2, x, x)
# biases (bias filter)
def convert_2_d(inputs, weight, biases):
    batch_num = inputs.shape[0]
    height = inputs.shape[1]
    width_input = inputs.shape[2]
    channel = inputs.shape[3]
    filter_size = weight.shape[0]
    kernels = weight.shape[2]
    output_1 = 1 + height - kernels
    output_2 = 1 + width_input - kernels
    size_output = number_array.zeros((batch_num, output_1, output_2, filter_size))
    flatten_width = weight.reshape(filter_size, channel * pow(kernels, 2))
    flatten_d_t = flatten_width.T
    biases_transpose = biases.T
    for i in range(output_1):
        for z in range (output_2):
            holder_2 = inputs[:,  i : i + kernels, z : z + kernels, :]
            holder_1 = holder_2.reshape(batch_num, channel * pow(kernels, 2))
            size_output[:, i, z, :] = biases_transpose + holder_1.dot(flatten_d_t)
    return size_output

def max_pooling(temp):
    batch_num = temp.shape[0]
    height = temp.shape[1]
    width = temp.shape[2]
    channel = temp.shape[3]
    output_width = width // 2
    output_height = height // 2
    output_size = number_array.zeros((batch_num, output_height, output_width, channel))
    for i in range(output_height):
        for z in range(output_width):
            holder_1 = area_size(i)
            holder_2 = area_size(z)
            area = temp[:, i * 2: holder_1, z * 2: holder_2, :]
            output_size[:, i, z, :] = area.max(axis = (1, 2))
    return output_size

def area_size(x):
    holder = x * 2 + 2
    return holder

def flatten_structure(temp):
    batch_num = temp.shape[0]
    height = temp.shape[1]
    width = temp.shape[2]
    channel = temp.shape[3]
    holder = channel * height * width
    result = temp.reshape(batch_num, holder)
    return result


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
    activation_4 = softmax(converted_output)
    forward_prop = {"inputs" : x, "converted_conv1" : converted_conv1, "activation_1": activation_1,
               "pooling_1": pooling_1, "converted_conv2": converted_conv2, "activation_2": activation_2,
               "pooling_2": pooling_2, "struc_flat": struc_flat, "converted_dense": converted_dense,
               "activation_3": activation_3, "converted_output": converted_output, "activation_4": activation_4}

    return

def cost_cnn(activation, y):
    number_size = activation.shape[0]
    epsilon = 1e-8
    prob_container = number_array.log(epsilon + activation)
    cost_holder = (-1 * number_array.sum(prob_container * y)) / number_size
    return cost_holder

def back_prop_cnn(forward_prop, parameters_tuple, y):
    number_size = forward_prop["inputs"].shape[0]
    d_holder4 = forward_prop["activation_4"] - y

    # Output Layer
    transpose_activation_3 = forward_prop["activation_3"].T
    d_bias_output = number_array.sum(d_holder4) / number_size
    d_weight_output = number_array.dot(transpose_activation_3, d_holder4) / number_size
    d_output_layer = number_array.dot(d_holder4, parameters_tuple[6].T)

    # Dense Layer
    d_holder3 = d_relu(forward_prop["converted_dense"], d_output_layer)
    transpose_struct_flat = forward_prop["struc_flat"].T
    d_weight_dense = number_array.dot(transpose_struct_flat, d_holder3) / number_size
    d_bias_dense = number_array.sum(d_holder3) / number_size
    d_dense_layer = number_array.dot(d_holder3, parameters_tuple[4])

    d_pooling_2 = number_array.reshape(d_dense_layer, forward_prop["pooling_2"].shape)

def back_pooling(h_activation_2, d_pooling_2):
    batch_num_h_activation_2 = h_activation_2.shape[0]
    height_h_activation_2 = h_activation_2.shape[1]
    width_h_activation_2 = h_activation_2.shape[2]
    channel_h_activation_2 = h_activation_2.shape[3]
    batch_num_dp2 = d_pooling_2.shape[0]
    height_dp2 = d_pooling_2.shape[1]
    width_dp2 = d_pooling_2.shape[2]
    channel_dp2 = d_pooling_2.shape[3]
    scope_a2 = number_array.reshape(h_activation_2, (batch_num_h_activation_2, height_dp2, 2, width_dp2, 2, channel_h_activation_2))
    maximum_holder = number_array.max(scope_a2, axis = (2, 4), keepdims = True)
    masking = (scope_a2 == maximum_holder)
    d_pooling = number_array.reshape(d_pooling_2, (batch_num_dp2, height_dp2, 1, width_dp2, 1, channel_dp2))
    d_a2_mask_result = d_pooling * masking
    d_a2_final = number_array.reshape(d_a2_mask_result, (batch_num_h_activation_2, height_h_activation_2, width_h_activation_2, channel_h_activation_2))
    return d_a2_final


# Vectorized operation.
def d_relu(mat_size, holder):
    masking = mat_size > 0
    d_holder = holder * masking
    return d_holder

# Loop operation.
def d_relu1(mat_size, holder):
    batch_number = mat_size.shape[0]
    height = mat_size.shape[1]
    width = mat_size.shape[2]
    channel = mat_size.shape[3]
    d_holder = number_array.zeros(mat_size)

    for temp1 in range (batch_number):
        for temp2 in range(height):
            for temp3 in range(width):
                for temp4 in range (channel):
                    if mat_size[temp1, temp2, temp3, temp4] <= 0:
                        d_holder[temp1, temp2, temp3, temp4] = 0
                    else:
                        d_holder[temp1, temp2, temp3, temp4] = holder[temp1, temp2, temp3, temp4]
    return d_holder




def main():
    while True :
        input_holder = input("For Binary Classification enter b or for Multi-class Classification enter m: ")
        if input_holder.lower() == "b":
            print("Binary Classification:")
            binary_classification()
        else:
            print("Multi-class Classification:")
            multi_class_classification()
        string_input = input("Do you want to keep testing the model? (y/n): ")
        if string_input.lower() != "y" :
            break
main()