#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

from scipy import stats

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


class Node():
    def __init__(self):
        pass

class InnerNode(Node):
    def __init__(self, feature, threshold, criterion_value, left_child, right_child):
        self.feature = feature
        self.threshold = threshold
        self.criterion_value = criterion_value
        self.left_child = left_child
        self.right_child = right_child

class Leaf(Node):
    def __init__(self, value):
        self.value = value


class DT():

    def __init__(self, max_depth = None, subsample_feature_picker = None):
        self.root = None
        self.max_depth = max_depth
        self.subsample_feature_picker = subsample_feature_picker

    def build_tree(self, X, t, depth = 0):

        if self.max_depth != None and depth >= self.max_depth:
            labels = np.unique(t, return_counts=True)
            return Leaf(labels[0][np.argmax(labels[1])])

        if len(np.unique(t)) == 1:
            return Leaf(t[0])

        feature, threshold, criterion_value, left_indices, right_indices = self.find_best_split(X, t)

        if criterion_value != 0:
            left_child = self.build_tree(X[left_indices], t[left_indices], depth + 1)
            right_child = self.build_tree(X[right_indices], t[right_indices], depth + 1)
            return InnerNode(feature, threshold, criterion_value, left_child, right_child)

    def find_best_split(self, X, t):

        subsample_best_feature, best_threshold, left_indices, right_indices = None, None, None, None
        best_criterion = float('inf')
        
        picker = self.subsample_feature_picker(X.shape[1])
        subsample_features = X[:, picker]

        for feature in range(subsample_features.shape[1]):

            features = subsample_features[:, feature]
            unique_features = np.unique(features)
            thresholds = [ (i + j) / 2 for i, j in zip(unique_features, unique_features[1:]) ]

            for threshold in thresholds:

                left, right = self.split(features, threshold)

                if (len(left) != 0 and len(right) != 0):

                    criterion = self.compute_criterion(t, left, right)

                    if criterion < best_criterion:
                        subsample_best_feature = feature
                        best_threshold = threshold
                        best_criterion = criterion
                        left_indices = left
                        right_indices = right

        best_feature = np.where(picker == True)[0][subsample_best_feature]

        return best_feature, best_threshold, best_criterion, left_indices, right_indices


    def split(self, X, threshold):
        left = np.argwhere(X <= threshold).flatten()
        right = np.argwhere(X > threshold).flatten()
        return left, right

    def compute_criterion(self, t, left, right):
        return self.compute_entropy(t[left]) + self.compute_entropy(t[right]) - self.compute_entropy(t)

    def compute_entropy(self, t):
        class_count = np.bincount(t)
        p = class_count[class_count != 0] / len(t)
        return -len(t) * np.sum(p * np.log(p))

    def fit(self, X, t):
        X = np.array(X)
        self.root = self.build_tree(X, t)

    def predict(self, X):
        return [ self.predict_for(x, self.root) for x in X ]

    def predict_for(self, x, node):

        if isinstance(node, Leaf):
            return node.value

        if isinstance(node, InnerNode):
            if x[node.feature] <= node.threshold:
                return self.predict_for(x, node.left_child)
            else:
                return self.predict_for(x, node.right_child)



def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return generator_feature_subsampling.uniform(size=number_of_features) <= args.feature_subsampling

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    trees = [None] * args.trees

    for t in range(args.trees):
        trees[t] = DT(args.max_depth, subsample_features)

        if args.bagging:
            dataset_indices = bootstrap_dataset(train_data)
            trees[t].fit(train_data[dataset_indices], train_target[dataset_indices])
        else:
            trees[t].fit(train_data, train_target)

    train_predictions = np.zeros((args.trees, len(train_data)))
    test_predictions = np.zeros((args.trees, len(test_data)))

    for t in range(args.trees):
        train_predictions[t] = trees[t].predict(train_data)
        test_predictions[t] = trees[t].predict(test_data)

    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with the smallest class number in case of a tie.

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_target, stats.mode(train_predictions, axis=0, keepdims=False)[0])
    test_accuracy = sklearn.metrics.accuracy_score(test_target, stats.mode(test_predictions, axis=0, keepdims=False)[0])

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))