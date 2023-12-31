from audioop import mul
from dataclasses import dataclass
import enum
from sklearn.naive_bayes import CategoricalNB
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from collections import Counter
from typing import List, Tuple


##################################################################
# 1. Generate random samples from d-dimension Gaussian distribution.
def rand_gaussian(mean, cov, size=1):
    """
    Generate random samples from d-dimension Gaussian distribution.
    :param mean: mean vector, size [d,]
    :param cov: covariance matrix, size: [d, d]
    :param size: sampling size
    :return: a scalar or an array with length `size`
    """
    mean = np.array(mean)
    cov = np.atleast_2d(cov)
    return np.random.multivariate_normal(mean, cov, size)



##################################################################
# 2. A discriminant procedure: calculate logrithm of post probability
def log_post_prob(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, class_prob: float):
    """
    Post probability estimation based on multivariant Gaussain distribution.
    """
    d = x.shape[0]
    return -0.5 * (x - mean) @ (np.linalg.inv(cov) @ (x - mean)) - d / 2 * np.log(2 * np.pi) - 0.5 * np.log(
        np.linalg.det(cov)) + np.log(class_prob)


def bayes_gaussian_predict(data: np.ndarray, paras: List[Tuple[np.ndarray, np.ndarray, float]], classes: List[int]):
    """
    ERM prediction
    """
    prob_est = []
    for paras_c in paras:
        prob_est.append([log_post_prob(x, *paras_c) for x in data])
    prob_est = np.vstack(prob_est)  # size [C, N]
    idx = prob_est.argmax(0)  # colum-wise maximum indices
    return classes[idx]


##################################################################
# 3. parameter estimation and test dataset evaluation
def est_gaussian_paras(data, labels):
    """
    Naive
    :param data: 2-D array like tensor
    :param labels: labels from C classes
    :return: a list including C two-element pairs, [..., (mean_i, cov_i, p_i), ...]
    """
    data = np.atleast_2d(data)[np.argsort(labels)]  # sort firstly
    labels = np.sort(labels)
    N = len(labels)
    classes = np.unique(labels)
    freqs = list(Counter(labels).values())
    probs = np.array(freqs) / N  # empirical class probabilities
    mu_list = []
    cov_list = []
    for c in classes:
        data_c = data[labels == c]
        mu = np.mean(data_c, axis=0)
        cov = np.cov(data_c, rowvar=False)
        mu_list.append(mu)
        cov_list.append(cov)

    return list(zip(mu_list, cov_list, probs))

##################################################################
# 4. Write a procedure for computing the Mahalanobis distance
# between a point x and some mean vector mu,
# given a covariance matrix.
def mahalanobis_dist(x, mean, cov):
    """
    Calculate Mahalanobis distance between a point and the mean vector.
    """
    x = np.array(x)
    mean = np.array(mean)
    cov = np.atleast_2d(cov)
    cov_inv = np.linalg.inv(cov)
    return np.sqrt(np.dot(x - mean, cov_inv @ (x - mean)))


##################################################################
# 5. Implement the naïve Bayes classifier from scratch and then compare your results to that of
# Python’s built-in implementation. Use different means, covariance matrices, prior probabilities
# (indicated by relative data size for each class) to demonstrate that your implementations are correct.


# 连续特征 or 离散特征：https://blog.csdn.net/jinruoyanxu/article/details/79237285
class NaiveBayesClassifier(BaseEstimator):
    """
    Self-implemented Naive Bayes Classifier, based on independent Gaussian distribution assumption
    """

    def __init__(self):
        self.classes = None
        self.paras = None
        self.probs = None

    def fit(self, X, y):
        X = np.atleast_2d(X)[np.argsort(y)]  # sort firstly
        y = np.sort(y)

        N = len(y)
        freqs = list(Counter(y).values())
        self.classes = np.unique(y)
        self.probs = np.array(freqs) / N  # empirical class probabilities
        mu_list = []  # mean vector list
        cov_list = []  # covariance vector list (each cov is a diagonal matrix)
        for c in self.classes:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            cov = np.diag(Xc.std(axis=0))
            mu_list.append(mu)
            cov_list.append(cov)
        self.paras = list(zip(mu_list, cov_list))

    def _log_post_prob(self, x, mean, cov, prob):
        """
        Logrithm post probability calculating
        """
        d = x.shape[0]
        return -0.5 * (x - mean) @ (np.linalg.inv(cov) @ (x - mean)) - d / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prob)

    def predict(self, X):
        X = np.atleast_2d(X)
        prob_est = []  # size: [C, N]
        for i, c in enumerate(self.classes):
            prob_est.append([self._log_post_prob(
                x, self.paras[i][0], self.paras[i][1], self.probs[i]) for x in X])
        prob_est = np.vstack(prob_est)
        idx = prob_est.argmax(0)
        return self.classes[idx]

