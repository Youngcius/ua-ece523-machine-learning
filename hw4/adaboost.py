"""
Self-implemented Adaboost algorithm
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn import tree
from sample import sample


class MyAdaBoostClassifier(BaseEstimator):
    """
    My self-implemented Adaboost classifier, whose base estimator is shallow decision tree
    """

    def __init__(self, n_estimators=50, max_depth=10, sampling=False, rand_seed=None):
        """
        Initialize the instance with predefined super-parameters
        :param n_estimators: number of weak estimators
        :param max_depth: maximal depth for the series of week shallow decision trees
        :param sampling: if True, directly use weights vector to training; if False, sample the trainset each round: TODO, delete it
        :param rand_seed: random seed for sampling
        """
        super(MyAdaBoostClassifier, self).__init__()
        self.rand_seed = rand_seed
        self.n_estimators = n_estimators
        self.sampling = sampling
        self.estimators = [tree.DecisionTreeClassifier(max_depth=max_depth) for _ in range(n_estimators)]
        self.coefficients = np.ones(n_estimators)
        # weights distribution of training data, dynamically updated, size: [N, ]
        self.weights = None
        self.errors = np.ones(n_estimators)

    def fit(self, X, y):
        """
        Fitting the model
        :param X: 2-D training data
        :param y: 1-D labels
        """
        X = np.atleast_2d(X)
        y = np.ravel(y)
        N = len(y)

        self.weights = np.ones_like(y) / N

        # train M estimators in loop
        # np.random.seed(self.rand_seed)
        for m in range(self.n_estimators):
            # 1) resampling training dataset
            idx = sample(N, self.weights, rand_seed=self.rand_seed)
            X_, y_ = X[idx], y[idx]

            # 2) m-th weak classifier
            self.estimators[m].fit(X_, y_)
            yhat_ = self.estimators[m].predict(X_)
            
            # print('{}-th weak estimator, accu: {:.4f}'.format(m, metrics.accuracy_score(yhat_, y_)))

            # 3) calculate error and coefficient
            self.errors[m] = np.sum((yhat_ != y_) * self.weights) + 1e-4
            self.coefficients[m] = 0.5 * np.log((1-self.errors[m])/(self.errors[m]))

            # 4) update weights
            self.weights *= np.exp(-self.coefficients[m] * y_ * yhat_)
            self.weights /= sum(self.weights)

    def predict(self, X):
        """
        Predicting method
        :param X: 2-D data
        :return: 1-D predicts
        """
        res = np.array([self.estimators[m].predict(X) for m in range(self.n_estimators)]) # size: [M, N]
        coeff = self.coefficients.reshape(-1,1) # size: [M, 1]
        pred = (coeff * res).sum(0) # size: [N,]
        return np.sign(pred)
        