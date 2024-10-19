

#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--epochs", default=200, type=int, help="Number of SGD training epochs")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def K(x, z, kernel, degree, gamma):
    if (kernel == "poly"):
        return (gamma * np.dot(x.T, z) + 1)**degree
    elif (kernel == "rbf"):
        return np.e**(-gamma * np.linalg.norm(x - z)**2)


def main(args: argparse.Namespace) -> tuple[np.ndarray, float, list[float], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    # Initialize the parameters: the betas and the bias.
    betas = np.zeros(args.data_size)
    bias = 0

    kernels = np.array([ [K(train_data[i], train_data[j], args.kernel, args.kernel_degree, args.kernel_gamma) for j in range(train_data.shape[0])]
                         for i in range(train_data.shape[0])])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        
        for i in range(0, len(permutation), args.batch_size):
            batch = permutation[i:i + args.batch_size]
            update = np.zeros(len(betas))

            for i in batch:
                update[i] = (1 / args.batch_size) * (np.sum([betas[j] * kernels[i, j] for j in range(len(betas))]) + bias - train_target[i])

            bias -= (args.learning_rate / args.batch_size) * np.sum([np.sum([betas[j] * kernels[i, j] for j in range(len(betas))]) + bias - train_target[i] for i in batch])
            betas -= args.learning_rate * (update + args.l2 * betas)
            
        def train_predict(i):
            return np.sum([betas[j] * kernels[i, j] for j in range(len(betas))]) + bias

        def test_predict(i):
            return np.sum([betas[j] * K(test_data[i], train_data[j], args.kernel, args.kernel_degree, args.kernel_gamma) for j in range(len(betas))]) + bias

        # TODO: Append current RMSE on train/test data to `train_rmses`/`test_rmses`.

        train_rmses.append(sklearn.metrics.mean_squared_error(train_target, np.array([train_predict(z) for z in range(len(train_data))]), squared=False))
        test_rmses.append(sklearn.metrics.mean_squared_error(test_target, np.array([test_predict(z) for z in range(len(test_data))]), squared=False))

        if (epoch + 1) % 10 == 0:
            print("After epoch {}: train RMSE {:.2f}, test RMSE {:.2f}".format(
                epoch + 1, train_rmses[-1], test_rmses[-1]))

    return betas, bias, train_rmses, test_rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    betas, bias, train_rmses, test_rmses = main(args)
    print("Learned betas", *("{:.2f}".format(beta) for beta in betas[:15]), "...")
    print("Learned bias", bias)