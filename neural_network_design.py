from sklearn.neural_network import MLPClassifier as nn
import numpy as np

def OCRNeuralNetwork(data_matrix, data_labels, test_indices, train_indices):
    best_performance = 0
    best_nodes = 0

    for i in range(5, 50, 5):
        # Initialize the MLPClassifier with i hidden nodes
        nn_model = nn(hidden_layer_sizes=(i,), max_iter=1000)
        
        # Train the model
        nn_model.fit(data_matrix[train_indices], data_labels[train_indices])
        
        # Calculate the performance
        avg_sum = 0
        for j in range(100):
            correct_guess_count = 0
            for index in test_indices:
                test = data_matrix[index].reshape(1, -1)  # Ensure the test data is 2D
                prediction = nn_model.predict(test)
                if data_labels[index] == prediction:
                    correct_guess_count += 1

            avg_sum += (correct_guess_count / float(len(test_indices)))
        
        avg_performance = avg_sum / 100
        
        print(f"{i} Hidden Nodes: {avg_performance:.4f}")
        
        # Keep track of the best performance
        if avg_performance > best_performance:
            best_performance = avg_performance
            best_nodes = i

    print(f"Best performance was {best_performance:.4f} with {best_nodes} hidden nodes.")
    return best_nodes, best_performance

# Example usage (assuming data_matrix, data_labels, test_indices, train_indices are defined):
# best_nodes, best_performance = OCRNeuralNetwork(data_matrix, data_labels, test_indices, train_indices)
