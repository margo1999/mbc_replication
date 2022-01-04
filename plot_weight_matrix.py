import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    path = '/Users/Jette/GitRepo/clock_network/data/sequence_learning_performance/sequence_learning_and_prediction/2021-12-17 14:45:27.382192/ee_connections.npy'
    connections = np.load(path)
    max_neuron = int(max(connections[:,0].max(), connections[:,1].max()))
    weight_matrix = np.zeros((max_neuron + 1, max_neuron + 1))

    for pre, post, weight in connections:
        weight_matrix[int(pre), int(post)] = weight

    path_ = '/Users/Jette/GitRepo/clock_network/data/sequence_learning_performance/sequence_learning_and_prediction/2021-12-17 14:45:27.382192/ee_connections_before.npy'
    connections_ = np.load(path_)
    max_neuron_ = int(max(connections_[:,0].max(), connections_[:,1].max()))
    weight_matrix_ = np.zeros((max_neuron_ + 1, max_neuron_ + 1))

    for pre, post, weight in connections_:
        weight_matrix_[int(pre), int(post)] = weight

    plt.imshow(weight_matrix)

    plt.figure()

    plt.imshow(weight_matrix_)
    plt.show()