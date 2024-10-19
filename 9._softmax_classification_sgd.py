#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def predict_at(x, weights):
    return np.argmax(softmax(np.dot(x, weights)))

def accuracy(data, target, weights):
    predictions = [predict_at(x, weights) for x in data]
    return np.sum([y_true == y_pred for y_true, y_pred in zip(target, predictions)]) / len(target)

def loss(data, target, weights):
    predictions = [softmax(np.dot(x, weights)) for x in data]
    return np.sum( [ -target[i] * np.log(predictions[i]) for i in range(len(target)) ] ) / len(target)



def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    onehot = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
    onehot_train_target = onehot.fit_transform(train_target.reshape(-1, 1))

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        permutation = permutation.reshape([permutation.size // args.batch_size, args.batch_size])
        for batch in permutation:
            gradient = 0
            for i in batch:
                x = train_data[i].reshape(1, len(train_data[i]))
                t = onehot_train_target[i].reshape(1, len(onehot_train_target[i]))
                gradient += (1 / args.batch_size) * ((softmax(x @ weights) - t).T @ x)
            weights -= (args.learning_rate * gradient).T


        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or cross-entropy loss, or KL loss) per example.
        train_accuracy = accuracy(train_data, train_target, weights)
        train_loss = loss(train_data, onehot_train_target, weights)
        test_accuracy = accuracy(test_data, test_target, weights)
        test_loss = loss(test_data, onehot.fit_transform(test_target.reshape(-1, 1)), weights)

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")