"""
neuralnet.py

What you need to do:
- Complete random_init
- Implement SoftMaxCrossEntropy methods
- Implement Sigmoid methods
- Implement Linear methods
- Implement NN methods

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming 
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
"""

import numpy as np
import argparse
from typing import Callable, List, Tuple

from matplotlib import pyplot as plt

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')


def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
str, str, str, int, int, int, float]:
    """
    DO NOT modify this function.

    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in
        (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in*
        (numpy array)
    y_te: test label (numpy array)
    out_tr: file for predicted output for train data (file)
    out_te: file for predicted output for test data (file)
    out_metrics: file for output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """
    DO NOT modify this function.

    ZERO Initialization: All weights are initialized to 0.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape)


def random_init(shape):
    """

    RANDOM Initialization: The weights are initialized randomly from a uniform
        distribution from -0.1 to 0.1.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)  # Don't change this line!

    # TODO: create the random matrix here!
    # Hint: numpy might have some useful function for this
    # Hint: make sure you have the right distribution
    weights = np.random.uniform(low=-0.1, high=0.1, size=(M, D))
    return weights


class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """
        # TODO: implement
        z = z - np.max(z)
        exp_z = np.exp(z)
        softmax_output = exp_z / np.sum(exp_z)
        return softmax_output

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        # TODO: implement
        return -np.log(y_hat[y])

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        # TODO: Call your implementations of _softmax and _cross_entropy here
        y_hat = self._softmax(z)
        loss = self._cross_entropy(y, y_hat)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        probabilities, we are directly computing gradient w.r.t. the softmax
        input.

        Try deriving the gradient yourself (see Question 1.2(b) on the written),
        and you'll see why we want to calculate this in a single step.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        # TODO: implement using the formula you derived in the written
        y_true = np.zeros_like(y_hat)
        y_true[y] = 1
        grad = y_hat - y_true
        return grad


class Sigmoid:
    def __init__(self):
        """
        Initialize state for sigmoid activation layer
        """
        # TODO Initialize any additional values you may need to store for the
        #  backward pass here
        self.sigmoid_output = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass
        x = np.clip(x, -500, 500)
        self.sigmoid_output = 1 / (1 + np.exp(-x))
        return self.sigmoid_output

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
        :return: partial derivative of loss with respect to input of
            sigmoid activation
        """
        # TODO: implement
        sigmoid_de = self.sigmoid_output * (1 - self.sigmoid_output)
        return dz * sigmoid_de


# This refers to a function type that takes in a tuple of 2 integers (row, col)
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]


class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in the input of the layer
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight
                               matrices for layer. This function takes in a
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate

        # TODO: Initialize weight matrix for this layer - since we are
        #  folding the bias into the weight matrix, be careful about the
        #  shape you pass in.
        #  To be consistent with the formulas you derived in the written and
        #  in order for the unit tests to work correctly,
        #  the first dimension should be the output size
        self.w = weight_init_fn((output_size, input_size + 1))

        # TODO: set the bias terms to zero
        self.w[:, 0] = 0

        # TODO: Initialize matrix to store gradient with respect to weights
        self.dw = np.zeros_like(self.w)

        # TODO: Initialize any additional values you may need to store for the
        self.x_store = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.
        :return: output z of linear layer with shape (output_size,

        HINT: You may want to cache some of the values you compute in this
        function. Inspect your expressions for backprop to see which values
        should be cached.
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass

        x_with_bias = np.insert(x, 0, 1)  # add the bias column
        self.x_store = x_with_bias
        z = np.dot(self.w, x_with_bias)
        return z

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear

        Note that this function should set self.dw
            (gradient of loss with respect to weights)
            but not directly modify self.w; NN.step() is responsible for
            updating the weights.

        HINT: You may want to use some of the values you previously cached in
        your forward() method.
        """

        self.dw = np.outer(dz, self.x_store)
        dx = np.dot(dz, self.w[:, 1:])  # remove bias
        return dx

    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been
        set in NN.backward().
        """
        # TODO: implement
        self.w -= self.lr * self.dw


class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float, num_hidden_layers: int = 1):
        """
        Initialize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.
        :param input_size: number of units in input to network
        :param hidden_size: number of units in each hidden layer
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight
                               matrices for layer. This function takes in a
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        :param num_hidden_layers: number of hidden layers (1 or 2)
        """
        self.num_hidden_layers = num_hidden_layers
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize layers based on the number of hidden layers
        self.linear1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.sigmoid1 = Sigmoid()

        if self.num_hidden_layers == 2:
            self.linear2 = Linear(hidden_size, hidden_size, weight_init_fn, learning_rate)
            self.sigmoid2 = Sigmoid()

        # Output layer
        self.output_layer = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.softmax_cross_entropy = SoftMaxCrossEntropy()

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        # Forward pass with an additional hidden layer if num_hidden_layers == 2
        z1 = self.linear1.forward(x)
        a1 = self.sigmoid1.forward(z1)

        if self.num_hidden_layers == 2:
            z2 = self.linear2.forward(a1)
            a2 = self.sigmoid2.forward(z2)
            z3 = self.output_layer.forward(a2)
        else:
            z3 = self.output_layer.forward(a1)

        y_hat, loss = self.softmax_cross_entropy.forward(z3, y)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        # Backward pass with additional layer if num_hidden_layers == 2
        dz = self.softmax_cross_entropy.backward(y, y_hat)

        if self.num_hidden_layers == 2:
            da2 = self.output_layer.backward(dz)
            dz2 = self.sigmoid2.backward(da2)
            da1 = self.linear2.backward(dz2)
        else:
            da1 = self.output_layer.backward(dz)

        dz1 = self.sigmoid1.backward(da1)
        self.linear1.backward(dz1)

    def step(self):
        # Update weights
        self.linear1.step()
        if self.num_hidden_layers == 2:
            self.linear2.step()
        self.output_layer.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # TODO: compute loss over the entire dataset
        #  Hint: reuse your forward function
        ttl_loss = 0
        for x, label in zip(X, y):
            _, loss = self.forward(x, label)
            ttl_loss += loss
        return ttl_loss / len(X)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        train_losses = []
        test_losses = []

        # Keep a copy of the original training data
        original_X_tr = np.copy(X_tr)
        original_y_tr = np.copy(y_tr)

        for epoch in range(n_epochs):
            # Reshuffle the original training data
            X_tr, y_tr = shuffle(original_X_tr, original_y_tr, epoch)

            for x_sample, y_sample in zip(X_tr, y_tr):
                y_hat, loss = self.forward(x_sample, y_sample)
                self.backward(y_sample, y_hat)
                self.step()

            train_loss = self.compute_loss(X_tr, y_tr)
            test_loss = self.compute_loss(X_test, y_test)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        return train_losses, test_losses

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        predictions = []
        error = 0
        for x, label in zip(X, y):
            y_hat, _ = self.forward(x, label)
            predict = np.argmax(y_hat)

            predictions.append(predict)

            if predict != label:
                error += 1

        error_rate = error / len(y)
        return np.array(predictions), error_rate

# Define our labels
labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]


def plot_losses_comparison(train_losses_1: List[float], val_losses_1: List[float],
                           train_losses_2: List[float], val_losses_2: List[float],
                           learning_rate: float) -> None:
    """
    Plot and save the average training and validation cross-entropy loss
    for both 1-hidden-layer and 2-hidden-layer models.

    :param train_losses_1: Training losses per epoch for 1 hidden layer model.
    :param val_losses_1: Validation losses per epoch for 1 hidden layer model.
    :param train_losses_2: Training losses per epoch for 2 hidden layer model.
    :param val_losses_2: Validation losses per epoch for 2 hidden layer model.
    :param learning_rate: Learning rate used during training.
    """
    epochs = range(1, len(train_losses_1) + 1)
    plt.figure(figsize=(10, 6))

    # Plotting training and validation losses for 1 hidden layer model
    plt.plot(epochs, train_losses_1, label='1 Hidden Layer - Training Loss', linestyle='-')
    plt.plot(epochs, val_losses_1, label='1 Hidden Layer - Validation Loss', linestyle='--')

    # Plotting training and validation losses for 2 hidden layer model
    plt.plot(epochs, train_losses_2, label='2 Hidden Layers - Training Loss', linestyle='-.')
    plt.plot(epochs, val_losses_2, label='2 Hidden Layers - Validation Loss', linestyle=':')

    plt.xlabel("Epochs")
    plt.ylabel("Average Cross-Entropy Loss")
    plt.legend()
    plt.title(f"Training and Validation Losses for 1 and 2 Hidden Layer Models (LR: {learning_rate})")
    plt.grid(True)
    plt.savefig("loss_comparison_plot.png")
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()

    # Define hyperparameters
    hidden_size = 50
    learning_rate = 0.003
    n_epochs = 100

    # Load data
    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)

    # Train 1-hidden-layer model
    nn_1_layer = NN(
        input_size=X_tr.shape[-1],
        hidden_size=hidden_size,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=learning_rate,
        num_hidden_layers=1
    )
    train_losses_1, test_losses_1 = nn_1_layer.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # Train 2-hidden-layer model
    nn_2_layers = NN(
        input_size=X_tr.shape[-1],
        hidden_size=hidden_size,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=learning_rate,
        num_hidden_layers=2
    )
    train_losses_2, test_losses_2 = nn_2_layers.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # Plot losses for comparison
    plot_losses_comparison(train_losses_1, test_losses_1, train_losses_2, test_losses_2, learning_rate)

