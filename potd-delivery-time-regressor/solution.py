import numpy as np


def delivery_time_regressor(
    x: np.ndarray, y: np.ndarray, queries: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    x: n x d training feature matrix.
    y: length-n training targets.
    queries: m x d feature matrix to predict for.

    Fit w = pinv(X_aug) @ y, where X_aug prepends a column of ones to x
    (intercept term). Use an SVD-based pseudoinverse or lstsq, never an
    explicit inverse of X^T X, so collinear / rank-deficient features
    don't blow up.

    Return (intercept, coefficients, predictions), where coefficients has
    shape (d,) and predictions has shape (m,).
    """
    # TODO: augment with an intercept column, solve via pseudoinverse/lstsq,
    # then apply the fitted model to each query row.
    X_aug = np.column_stack((np.ones(x.shape[0]), x))

    w = np.linalg.pinv(X_aug) @ y

    intercept = float(w[0])
    coefficients  = w[1:]

    predictions = intercept + queries @ coefficients
    return intercept, coefficients, predictions
