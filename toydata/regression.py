import numpy as np


def create_regression_data(num_samples=100, num_features=3, return_true_weights=False):
    X = np.random.normal(size=(num_samples, num_features))
    true_beta = np.random.normal(size=num_features)
    eps = np.random.normal(loc=0, scale=0.1, size=num_samples)
    y = true_beta.T.dot(X.T) + eps
    if return_true_weights:
        return X, y, true_beta
    else:
        return X, y

