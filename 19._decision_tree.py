#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

from queue import PriorityQueue

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


class Node():
    def __init__(self,
                 feature = None, threshold = None, criterion_value = None, 
                 left_child = None, right_child = None, 
                 value = None):
        self.feature = feature
        self.threshold = threshold
        self.criterion_value = criterion_value
        self.left_child = left_child
        self.right_child = right_child
        self.value = value

    def __lt__(self, other):
        return self.criterion_value < other.criterion_value


class DT():

    def __init__(self, criterion = 'gini', max_depth = None, max_leaves = None, min_to_split = None):
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_to_split = min_to_split


    def build_tree(self, X, t):
        q = PriorityQueue()

        feature, threshold, criterion, left_indices, right_indices = self.find_best_split(X, t)
        root = Node(feature, threshold, criterion)
        q.put((root, 0, X, t, left_indices, right_indices))
        
        on_queue = 1
        leaves = 0

        while on_queue + leaves < self.max_leaves:
            node, depth, data, target, left_i, right_i = q.get()
            on_queue -= 1

            if (self.min_to_split != None and len(data) < self.min_to_split) or \
               (self.max_depth != None and depth >= self.max_depth) or \
                node.criterion_value == 0:
                node.value = np.argmax(np.bincount(target))
                leaves += 1
                continue

            left_X = data[left_i]
            right_X = data[right_i]
            left_t = target[left_i]
            right_t = target[right_i]

            left_feature, left_threshold, left_criterion, left_left_i, left_right_i = self.find_best_split(left_X, left_t)
            right_feature, right_threshold, right_criterion, right_left_i, right_right_i = self.find_best_split(right_X, right_t)

            left = Node(left_feature, left_threshold, left_criterion)
            right = Node(right_feature, right_threshold, right_criterion)

            node.left_child = left
            node.right_child = right

            q.put((left, depth + 1, left_X, left_t, left_left_i, left_right_i))
            q.put((right, depth + 1, right_X, right_t, right_left_i, right_right_i))

            on_queue += 2

        while on_queue > 0:
            node, _, _, target, _, _ = q.get()
            node.value = np.argmax(np.bincount(target))
            on_queue -= 1

        return root


    def build_tree_recursive(self, X, t, depth = 0):

        if (self.min_to_split != None and len(X) < self.min_to_split) or (self.max_depth != None and depth >= self.max_depth) or len(np.unique(t)) == 1:
            return Node(value = np.argmax(np.bincount(t)))

        feature, threshold, criterion_value, left_indices, right_indices = self.find_best_split(X, t)

        if criterion_value != 0:
            left_child = self.build_tree_recursive(X[left_indices], t[left_indices], depth + 1)
            right_child = self.build_tree_recursive(X[right_indices], t[right_indices], depth + 1)
            return Node(feature, threshold, criterion_value, left_child, right_child)


    def find_best_split(self, X, t):

        best_feature, best_threshold, left_indices, right_indices = None, None, None, None
        best_criterion = float('inf')

        for feature in range(X.shape[1]):

            features = X[:, feature]
            unique_features = np.unique(features)
            thresholds = [ (i + j) / 2 for i, j in zip(unique_features, unique_features[1:]) ]

            for threshold in thresholds:

                left, right = self.split(features, threshold)

                if (len(left) != 0 and len(right) != 0):

                    criterion = self.compute_criterion(t, left, right)

                    if criterion < best_criterion:
                        best_feature = feature
                        best_threshold = threshold
                        best_criterion = criterion
                        left_indices = left
                        right_indices = right
                        
        return best_feature, best_threshold, best_criterion, left_indices, right_indices

    def split(self, X, threshold):
        left = np.argwhere(X <= threshold).flatten()
        right = np.argwhere(X > threshold).flatten()
        return left, right

    def compute_criterion(self, t, left, right):

        if self.criterion == "gini":
            return self.compute_gini(t[left]) + self.compute_gini(t[right]) - self.compute_gini(t)
        
        if self.criterion == "entropy":
            return self.compute_entropy(t[left]) + self.compute_entropy(t[right]) - self.compute_entropy(t)

    def compute_gini(self, t):
        class_count = np.bincount(t)
        p = class_count[class_count != 0] / len(t)
        return len(t) * np.sum(p * (1 - p))

    def compute_entropy(self, t):
        class_count = np.bincount(t)
        p = class_count[class_count != 0] / len(t)
        return -len(t) * np.sum(p * np.log(p))

    def fit(self, X, t):
        X = np.array(X)

        if self.max_leaves == None:
            self.root = self.build_tree_recursive(X, t)
        else:
            self.root = self.build_tree(X, t)

    def predict(self, X):
        return [ self.predict_for(x, self.root) for x in X ]

    def predict_for(self, x, node):

        if node.value != None:
            return node.value

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

    dt = DT(args.criterion, args.max_depth, args.max_leaves, args.min_to_split)
    dt.fit(train_data, train_target)

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_target, dt.predict(train_data))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, dt.predict(test_data))

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))