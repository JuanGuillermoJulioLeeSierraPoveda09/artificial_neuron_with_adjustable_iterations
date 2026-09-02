import numpy as np

class ArtificialNeuron:
    def __init__(self, input_quantity):
        self.weights = np.random.randn(input_quantity)
        self.bias = np.random.randn()

    def activation_function(self, z):
        return 1/(1+np.exp(-z))

    def forward(self, inputs):
        self.last_inputs = inputs
        #This is the Z = (x1*w1 + x2*w2 + ...) + b
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation_function(weighted_sum)

        return self.output

    def train_step(self, expected_output, learning_rate):
        error = expected_output - self.output
        derived_sigmoide = self.output*(1-self.output)
        delta = error*derived_sigmoide

        self.weights += learning_rate*delta*self.last_inputs
        self.bias += learning_rate*delta

        return error

if __name__ == "__main__":
    num_inputs = int(input("How many inputs do you need?: "))
    neuron = ArtificialNeuron(num_inputs)
    inputs_list = []

    for idx in range(num_inputs):
        val = float(input(f"Input #{idx+1}: "))
        inputs_list.append(val)

    X = np.array(inputs_list)
    expected_z = float(input("What is tthe expected result?:"))
    learning_rate = float(input("Learning rate (n): "))

    print("\n--- BEFORE TRAINING ---")
    print(f"Initial weights: {neuron.weights}")
    print(f"Initial bias: {neuron.bias:.4f}")
    initial_output = neuron.forward(X)
    print(f"Initial output: {initial_output:.4f}")
    error = neuron.train_step(expected_z, learning_rate)
    new_output = neuron.forward(X)

    print("\n--- AFTER ADJUSTMENT ---")
    print(f"Calculated Error: {error:.4f}")
    print(f"Updated weights: {neuron.weights}")
    print(f"Updated bias: {neuron.bias:.4f}")
    print(f"New output: {new_output:.4f}")