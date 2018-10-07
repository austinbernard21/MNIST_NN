import pandas as pd
import numpy as np
import random


class Network(object):
    def __init__(self, train_x, train_y, test_x, test_y, input_layer_size, hidden_layer_size, output_size):
        self.weight_1 = np.random.rand(hidden_layer_size, input_layer_size)
        self.bias_1 = np.random.rand(hidden_layer_size, 1)
        self.weight_2 = np.random.rand(output_size, hidden_layer_size)
        self.bias_2 = np.random.rand(output_size, 1)
        self.x_train = train_x
        self.y_train = train_y
        self.x_test = test_x
        self.y_test = test_y

    def feed_forward(self, x):
        z1 = np.dot(self.weight_1, x) + self.bias_1
        a1 = sigmoid(z1)
        z2 = np.dot(self.weight_2, a1) + self.bias_2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    def backprop(self, x, y, a2, z2, a1, z1):
        m = len(x)
        dz2 = a2 - y
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.weight_2.T, dz2) * sigmoid_p(z1)
        dw1 = (1 / m) * np.dot(dz1, x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        return dw2, dw1, db2, db1

    def predict(self, x):
        z1, a1, z2, a2 = self.feed_forward(x)
        return np.argmax(a2, axis=0)

    def update_parameters(self, x, y, mini_batch_size, alpha):
        z1, a1, z2, a2 = self.feed_forward(x)
        dw2, dw1, db2, db1 = self.backprop(x, y, a2, z2, a1, z1)
        self.weight_1 = self.weight_1 - alpha * dw1
        self.weight_2 = self.weight_2 - alpha * dw2
        self.bias_1 = self.bias_1 - alpha * db1
        self.bias_2 = self.bias_2 - alpha * db2

    def mini_batch_gradient_descent(self, mini_batch_size, epochs, alpha):
        m = len(self.x_train)
        for i in range(epochs):
            shuffle(self.x_train, self.y_train)
            mini_batches_x = [self.x_train[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]
            mini_batches_y = [self.y_train[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]
            for j in range(len(mini_batches_x)):
                self.update_parameters(mini_batches_x[j].T, mini_batches_y[j].T, mini_batch_size, alpha)

    def check_accuracy(self):
        prediction = self.predict(self.x_test.T)
        y_test_outputs = [np.argmax(y, axis=0) for y in self.y_test]
        print('accuracy for test set')
        print(np.sum(y_test_outputs == prediction) / len(y_test_outputs))


def read_data():
    test_data = pd.read_csv('mnist_test.csv')
    train_data = pd.read_csv('mnist_train.csv')

    x_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    return x_train, y_train, x_test, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) - (1 - sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def shuffle(X,Y):
    n_elem = X.shape[0]
    indeces = np.random.permutation(n_elem)
    return X[indeces], Y[indeces]

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    # create one hot encoded labels for y_training set and y_test set
    n_values = np.max(y_train) + 1
    y_train_encoded = np.eye(n_values)[y_train]

    n_values = np.max(y_test) + 1
    y_test_encoded = np.eye(n_values)[y_test]

    n1 = Network(x_train, y_train_encoded, x_test, y_test_encoded, 784, 275, 10)
    n1.mini_batch_gradient_descent(64, 10, 0.45)
    n1.check_accuracy()