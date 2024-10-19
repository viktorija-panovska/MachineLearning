#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def ReLU(x):
    return np.maximum(x, 0)


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(x):
    return np.argmax(x, axis=1)

def accuracy(predictions, targets):
    return np.sum([p == t for p, t in zip(predictions, targets)]) / len(targets)


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    onehot = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
    onehot_train_target = onehot.fit_transform(train_target.reshape(-1, 1))

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
        # The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
        hidden = ReLU(inputs @ weights[0] + biases[0])
        output = softmax(hidden @ weights[1] + biases[1])
        return hidden, output

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: The gradient used in SGD has now four parts, gradient of `weights[0]` and `weights[1]`
        # and gradient of `biases[0]` and `biases[1]`.

        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to `weights[1]` and `biases[1]`
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to `weights[0]` and `biases[0]`
        permutation = permutation.reshape([permutation.size // args.batch_size, args.batch_size])
        
        for batch in permutation:
            x = np.array([train_data[i] for i in batch])
            y = np.array([onehot_train_target[i] for i in batch])
            hidden, output = forward(x)

            dL_dy_in = np.array(output - y)
            dL_dh_in = np.dot(dL_dy_in, weights[1].T) * (hidden > 0)

            biases[1] -= (args.learning_rate / args.batch_size) * np.sum(dL_dy_in, axis=0)
            weights[1] -= (args.learning_rate / args.batch_size) * np.dot(hidden.T, dL_dy_in)
            biases[0] -= (args.learning_rate / args.batch_size) * np.sum(dL_dh_in, axis=0)
            weights[0] -= (args.learning_rate / args.batch_size) * np.dot(x.T, dL_dh_in)


        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.
        train_accuracy = accuracy(predict(forward(train_data)[1]), train_target)
        test_accuracy = accuracy(predict(forward(test_data)[1]), test_target)

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")