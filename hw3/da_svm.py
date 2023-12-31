"""
Linear soft-margin SVM and Domain-adaptation SVM implementation by Convex Quadratic Programming
"""
import numpy as np
import cvxpy as cp


def da_svm_hyperplane(X, y, w_s, B, C, dual=False, return_bias=False):
    """
    Calculate the hyperplane of Domain Adaptation (Linear soft-margin) SVM
    :param X: dataset features, 2-D array-like input
    :param y: dataset labels, 1-D array-like input
    :param w_s: known normal vector of source domain dataset
    :param B: regularization factor in terms of known hyperplane of source domain
    :param C: penalty factor of mis-classification
    :param dual: via solving the dual problem or not
    :param return_bias: whether return the bias coefficient or not
    :return: hyperplane normal vector, with bias coefficient optionally
    """
    X = np.atleast_2d(X)
    y = np.ravel(y)
    N, d = X.shape
    assert len(w_s) == d, "size of w_s is not applicable"
    if dual:
        a = cp.Variable(N)  # alpha vector in the familiar formula
        G = (X * y.reshape(-1, 1)) @ (X * y.reshape(-1, 1)).T  # Gram matrix
        G = G + 0.001 * np.eye(*G.shape)  # in order to satisfy the DCP condition
        obj = cp.Minimize(0.5 * cp.quad_form(a, G) + B * cp.sum(cp.multiply(cp.multiply(a, y), X @ w_s)) - cp.sum(a))
        cons = [cp.sum(cp.multiply(a, y)) == 0, a >= 0, a <= C]

        problem = cp.Problem(obj, cons)
        problem.solve()

        a = a.value
        w_t = np.sum((a * y).reshape(-1, 1) * X, 0)
        idx = ((a > 0.01 * C) & (a < 0.99 * C)).tolist().index(True)
        b = y[idx] - w_t @ X[idx]
    else:
        w_t = cp.Variable(d)
        ksi = cp.Variable(N)
        b = cp.Variable()

        obj = cp.Minimize(0.5 * cp.norm(w_t) ** 2 + C * cp.sum(ksi) - B * w_t @ w_s)
        cons = [cp.multiply(y, X @ w_t + b) >= 1 - ksi, ksi >= 0]

        problem = cp.Problem(obj, cons)
        problem.solve()

        w_t, b = w_t.value, b.value

    if return_bias:
        return w_t, b
    else:
        return w_t


def linear_svm_hyperplane(X, y, C, dual=False, return_bias=False):
    """
    Calculate the hyperplane of Linear soft-margin SVM
    :param X: dataset features, 2-D array-like input
    :param y: dataset labels, 1-D array-like input
    :param C: penalty factor of mis-classification
    :param dual: via solving the dual problem or not
    :param return_bias: whether return the bias coefficient or not
    :return: hyperplane normal vector, with bias coefficient optionally
    """
    X = np.atleast_2d(X)
    y = np.ravel(y)
    N, d = X.shape
    if dual:
        a = cp.Variable(N)  # alpha vector in the familiar formula
        G = (X * y.reshape(-1, 1)) @ (X * y.reshape(-1, 1)).T  # Gram matrix
        G = G + 0.001 * np.eye(*G.shape)
        obj = cp.Minimize(0.5 * cp.quad_form(a, G) - cp.sum(a))
        cons = [cp.sum(cp.multiply(a, y)) == 0, a >= 0, a <= C]

        problem = cp.Problem(obj, cons)
        problem.solve()

        a = a.value
        w = np.sum((a * y).reshape(-1, 1) * X, 0)
        idx = ((a > 0.01 * C) & (a < 0.99 * C)).tolist().index(True)
        b = y[idx] - w @ X[idx]
    else:
        w = cp.Variable(d)
        b = cp.Variable()
        ksi = cp.Variable(N)

        obj = cp.Minimize(0.5 * cp.norm(w) ** 2 + C * cp.sum(ksi))
        cons = [cp.multiply(y, X @ w + b) >= 1 - ksi, ksi >= 0]

        problem = cp.Problem(obj, cons)
        problem.solve()

        w, b = w.value, b.value

    if return_bias:
        return w, b
    else:
        return w
