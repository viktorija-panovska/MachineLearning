 #!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics

from collections import Counter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type gaussian/multinomial/bernoulli")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.

    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    def gaussian():

        #train
        mean = np.zeros((args.classes, train_data.shape[1]))
        variance = np.zeros((args.classes, train_data.shape[1]))
        priors = np.zeros(args.classes)

        for c in range(args.classes):
            x = train_data[train_target == c]
            mean[c] = np.sum(x, axis=0) / len(x)
            variance[c] = np.sqrt(np.sum((x - mean[c])**2, axis=0) / len(x) + args.alpha)
            priors[c] = len(x) / len(train_data)

        #predict
        predictions = np.zeros(len(test_data))

        for i in range(len(test_data)):
            posteriors = np.zeros(args.classes)

            for c in range(args.classes):
                posteriors[c] = np.log(priors[c]) + np.sum(scipy.stats.norm.logpdf(test_data[i], loc=mean[c], scale=variance[c]))
            
            predictions[i] = np.argmax(posteriors)

        return predictions


    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    def multinomial():
        
        # train
        priors = np.zeros(args.classes)
        p = np.zeros((args.classes, train_data.shape[1]))

        for c in range(args.classes):
            x = train_data[train_target == c]
            priors[c] = len(x) / len(train_data)
            feature_sum = np.sum(x, axis=0)
            p[c] = (feature_sum + args.alpha) / (np.sum(feature_sum) + args.alpha * train_data.shape[1])

        #predict
        predictions = np.zeros(len(test_data))

        for i in range(len(test_data)):
            posteriors = np.zeros(args.classes)

            for c in range(args.classes):
                posteriors[c] = np.log(priors[c]) + np.sum(np.array(test_data[i]) * np.log(p[c]))

            predictions[i] = np.argmax(posteriors)

        return predictions

    
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    def bernoulli():

        # train
        binarized_train_data = (train_data >= 8)
        priors = np.zeros(args.classes)
        p = np.zeros((args.classes, binarized_train_data.shape[1]))

        for c in range(args.classes):
            x = binarized_train_data[train_target == c]
            priors[c] = len(x) / len(binarized_train_data)
            p[c] = (np.sum(x, axis=0) + args.alpha) / (len(x) + 2 * args.alpha)

        # predict
        binarized_test_data = (test_data >= 8)
        predictions = np.zeros(len(binarized_test_data))

        for i in range(len(binarized_test_data)):
            posteriors = np.zeros(args.classes)

            for c in range(args.classes):
                posteriors[c] = np.log(priors[c]) + np.sum(binarized_test_data[i] * np.log(p[c] / (1 - p[c])) + np.log(1 - p[c]))

            predictions[i] = np.argmax(posteriors)

        return predictions


    predictions = []

    if args.naive_bayes_type == "gaussian":
        predictions = gaussian()
    elif args.naive_bayes_type == "multinomial":
        predictions = multinomial()
    elif args.naive_bayes_type == "bernoulli":
        predictions = bernoulli()

    # TODO: Predict the test data classes and compute the test accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(test_target, predictions)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))