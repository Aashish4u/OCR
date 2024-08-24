import math
import numpy as np
import json

class OCRNeuralNetwork:
    LEARNING_RATE = 0.01  # Adjust this based on your needs
    NN_FILE_PATH = 'neural_network.json'

    def __init__(self, num_hidden_nodes, use_file=False):
        self.num_hidden_nodes = num_hidden_nodes
        self._use_file = use_file
        self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
        self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(1, 10)
        if use_file:
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        return np.random.rand(size_out, size_in) * 0.12 - 0.06

    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.exp(-z))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    def train(self, data):
        # Forward pass
        y1 = np.dot(self.theta1, data['y0'].T)
        sum1 = y1 + self.input_layer_bias.T
        y1 = self.sigmoid(sum1)

        y2 = np.dot(self.theta2, y1)
        y2 = y2 + self.hidden_layer_bias.T
        y2 = self.sigmoid(y2)

        # Compute the error
        actual_vals = np.zeros((10, 1))
        actual_vals[data['label']] = 1
        output_errors = actual_vals - y2
        hidden_errors = np.multiply(np.dot(self.theta2.T, output_errors), self.sigmoid_prime(sum1))

        # Update weights and biases
        self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, data['y0'])
        self.theta2 += self.LEARNING_RATE * np.dot(output_errors, y1.T)
        self.hidden_layer_bias += self.LEARNING_RATE * output_errors
        self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(self.theta1, test.T)
        y1 = y1 + self.input_layer_bias.T
        y1 = self.sigmoid(y1)

        y2 = np.dot(self.theta2, y1)
        y2 = y2 + self.hidden_layer_bias.T
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return

        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
            self.theta1 = np.array(nn['theta1'])
            self.theta2 = np.array(nn['theta2'])
            self.input_layer_bias = np.array(nn['b1'])
            self.hidden_layer_bias = np.array(nn['b2'])

# Example usage:
# nn = OCRNeuralNetwork(num_hidden_nodes=25, use_file=True)
# nn.train(training_data)
# prediction = nn.predict(test_data)
