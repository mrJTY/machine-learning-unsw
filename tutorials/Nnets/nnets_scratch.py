#https://www.youtube.com/watch?v=Py4xvZx-A1E

import numpy as np
import pdb

class Nnet():

    def __init__(self):
        np.random.seed(1)

        self.weights = 2 * np.random.random((3,1)) - 1


    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_inputs, training_outputs, training_iters):

        for i in range(training_iters):
            output = self.think(training_inputs)
            err = training_outputs - output
            adjustments = np.dot(training_inputs.T, err * self.sigmoid_derivative(output))
            self.weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))

        return output

if __name__ == '__main__':
    nnet = Nnet()

    print("Random weights:")
    print(nnet.weights)


    train_X = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])

    train_y = np.array([[0, 1, 1, 0]]).T

    nnet.train(train_X, train_y, 1000)

    print("X")
    print(train_X)
    print("y")
    print(train_y)

    print("Output")
    print(nnet.think(train_X))
