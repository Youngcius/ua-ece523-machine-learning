from cmath import sin
import numpy as np
from collections import Counter


def gen_cb(N, a, alpha):
    """
    Generate random checkerboard data
    :param N: numbe of points on the checkerboard
    :param a: width of the checkerboard
    :param alpha: rotation of the checkerboard in ranians
    """
    d = np.random.rand(N, 2).T
    d_transformed = np.array([
        d[0] * np.cos(alpha) - d[1]*np.sin(alpha),
        d[0]*np.sin(alpha) + d[1]*np.cos(alpha)
    ]).T
    s = np.ceil(d_transformed[:, 0]/a + np.floor(d_transformed[:, 1]/a))
    lab = 2 - (s % 2)
    data = d.T
    return data, lab


def prob_prior(y):
    y = np.sort(y).tolist()
    return dict(Counter(y).items())


class DensityEstimator:
    def __init__(self):
        self.classes = None
        self.paras = None
        self.probs = None  # piror probabilties, P(y)
        self.classes = None
        self.n_class = 0
        self.n_feature = 0

    def fit(self, X, y):
        """
        :param X: size [N, n_feature]
        :param y: labels, 1-D, array-like
        """
        X = np.atleast_2d(X)[np.argsort(y)]  # sort firstly
        y = np.sort(y)
        self.classes = np.unique(y)
        self.n_class = len(self.classes)
        self.n_feature = X.shape[1]

        N = len(y)
        freqs = list(Counter(y).values())
        self.probs = np.array(freqs) / N  # empirical class probabilities
        mu_list = []  # mean vector list
        cov_list = []  # covariance matrix list
        for c in self.classes:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            cov = np.cov(Xc.T)
            mu_list.append(mu)
            cov_list.append(cov)
        self.paras = list(zip(mu_list, cov_list))

    # def density(self, X):
        

    def _log_post_prob(self, x, mean, cov, prob):
        """
        Logrithm post probability calculating
        """
        d = x.shape[0]
        return -0.5 * (x - mean) @ (np.linalg.inv(cov) @ (x - mean)) - d / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prob)

    def predict(self, X, label=False):
        """
        :param X: testing data
        :param label: if True, return discrete values; return probabilities in default
        """
        X = np.atleast_2d(X)
        prob_est = []  # size: [C, N]
        for i, c in enumerate(self.classes):
            prob_est.append([self._log_post_prob(
                x, self.paras[i][0], self.paras[i][1], self.probs[i]) for x in X])
        prob_est = np.vstack(prob_est).T # size [N, n_class]
        if label:
            idx = prob_est.argmax(1)
            return self.classes[idx]
        else:
            return np.exp(prob_est)
