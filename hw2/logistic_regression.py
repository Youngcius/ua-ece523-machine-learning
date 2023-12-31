import numpy as np
from sklearn.base import BaseEstimator


class LogisticRegressor(BaseEstimator):
    """
    Classifier based on Logistic regression algorithm
    """

    def __init__(self, lr: float = 1e-3, eps=1e-4, max_iter=10000, seed=123) -> None:
        super(LogisticRegressor, self).__init__()
        self.classes = []
        self.n_class = 0
        self.n_feature = 0
        self.coef = np.array([])
        self.intercept = np.array([])
        self.lr = lr
        self.eps = eps
        self.max_iter = max_iter
        self.loss = np.inf
        self.seed = seed

    def fit(self, X, y):
        """
        Training process on training dataset
        :param X: the sample dataset, with size [n_sample, n_feature]
        :param y: 2-class or multi-class label vector
        """
        X = np.atleast_2d(X)[np.argsort(y)]
        y = np.sort(y)
        self.classes = np.unique(y)
        self.n_class = len(self.classes)
        self.n_feature = X.shape[1]

        C = self.n_class
        if C == 2:
            self.coef, self.intercept = self._binary_fit(X, y)
        else:
            raise NotImplementedError('not implemented multi-classification mdoel')

    def _binary_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Return the weight vector and intercept scalar for binary classification
        """
        N = len(X)

        # initialize weight and intercept parameters
        self.coef = np.random.randn(self.n_feature)
        self.intercept = 0

        np.random.seed(self.seed)
        for k in range(self.max_iter):
            # calculate corss-entropy loss
            p = self.predict(X)
            self.loss = - np.mean(y*np.log(p) + (1-y)*np.log(1-p))

            if self.loss < self.eps:
                break

            # update parameters by SGD
            i = np.random.choice(N)
            w_grad = (p[i] - y[i]) * X[i]  # gradient on i-th sample
            b_grad = p[i] - y[i]
            
            # update parameters by GD
            # w_grad = np.mean((p - y)[:, np.newaxis] * X, 0)
            # b_grad = np.mean((p - y)[:, np.newaxis], 0)

            self.coef -= self.lr * w_grad
            self.intercept -= self.lr * b_grad
        return self.coef, self.intercept

    def _multiple_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Multiple calssification fitting
        """
        C = len(self.classes)
        self.coef = np.random.randn(self.n_feature, int(C * (C-1)/2))
        self.intercept = np.zeros(int(C*(C-1)/2))
        col_idx = 0

        # C * (C - 1) / 2 loops
        for i in range(C):
            for j in range(i+1, C):
                self.coef[:, col_idx], self.intercept[col_idx] = self._binary_fit()
                col_idx += 1
                pass

    def predict(self, X, prob=False):
        """
        Predict process on unkown data
        :param X: the sample dataset, with size [n_sample, n_feature]
        :param prob: if designed True, return probabilities rather than labels
        """
        X = np.atleast_2d(X)  # size [n, m]
        p = 1 / (1 + np.exp(- (self.coef @ X.T + self.intercept)))
        if prob:
            return p
        else:
            pred = np.zeros(len(X))
            pred[p >= 0.5] = 1
            return pred
