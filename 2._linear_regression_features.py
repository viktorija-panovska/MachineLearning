#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=40, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--range", default=3, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> list[float]:
    # Create the data
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    # ones = np.ones([xs.size])
    # a = np.concatenate([xs.reshape([xs.size, 1]), ones.reshape([ones.size, 1])], axis=1)

    # print(a[:, 0])

    rmses = []
    for order in range(1, args.range + 1):
        # TODO: Create features `(x^1, x^2, ..., x^order)`, preferably in this ordering.
        # Note that you can just append `x^order` to the features from the previous iteration.
        if (order == 1):
            xs = xs.reshape([xs.size, 1])
        else:
            new_feature = xs[:, 0] * xs [:, -1]
            xs = np.concatenate([xs, new_feature.reshape([new_feature.size, 1])], axis=1)

        # TODO: Split the data into a train set and a test set.
        # Use `sklearn.model_selection.train_test_split` method call, passing
        # arguments `test_size=args.test_size, random_state=args.seed`.
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(xs, ys, test_size=args.test_size, random_state=args.seed)

        # TODO: Fit a linear regression model using `sklearn.linear_model.LinearRegression`;
        # consult the documentation and see especially the `fit` method.
        model = sklearn.linear_model.LinearRegression().fit(train_data, train_target)

        # TODO: Predict targets on the test set using the `predict` method of the trained model.
        test_predicted = model.predict(test_data)

        # TODO: Compute root mean square error on the test set predictions.
        # You can either do it manually or look at `sklearn.metrics.mean_squared_error` method
        # and its `squared` parameter.
        rmse = sklearn.metrics.mean_squared_error(test_target, test_predicted, squared=False)

        rmses.append(rmse)

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.plot(train_data[:, 0], train_target, "go")
            plt.plot(test_data[:, 0], test_target, "ro")
            plt.plot(np.linspace(xs[0], xs[-1], num=100),
                     model.predict(np.power.outer(np.linspace(xs[0], xs[-1], num=100), np.arange(1, order + 1))), "b")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))