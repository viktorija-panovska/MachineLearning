 #!/usr/bin/env python3
import argparse
from cmath import sqrt

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    input = dataset.data
    target = dataset.target

    # TODO: Append a new feature to all input data, with value "1"
    rows = input.shape[0]
    ones = np.ones([rows, 1])

    input = np.concatenate([input, ones], axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    input_train, input_test, target_train, target_test = sklearn.model_selection.train_test_split(input, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    weights = np.linalg.inv(input_train.T @ input_train) @ input_train.T @ target_train

    # TODO: Predict target values on the test set.
    predicted_target = np.array(input_test @ weights)

    # TODO: Manually compute root mean square error on the test set predictions.
    target_test = np.array(target_test)
    mse = np.square(np.subtract(predicted_target, target_test)).mean()
    rmse = np.sqrt(mse)

    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))