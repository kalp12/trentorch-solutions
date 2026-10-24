import numpy as np


def causal_mask_misfire(scores: np.ndarray) -> np.ndarray:
    """
    Row-wise, numerically stable softmax under a causal mask: for query
    row i, only key columns j <= i are visible. Forbidden entries (j > i)
    must be exactly 0.0 in the output.

    scores: shape (n, n).
    """
    # TODO: mask with -inf BEFORE taking the row max, so forbidden entries
    # never influence the max-subtraction or the normalizer.
    n = scores.shape[0]

    masked = scores.copy()
    masked[np.triu_indices(n, k = 1)] = -np.inf

    row_max = np.max(masked, axis = 1, keepdims = True)
    exp_scores = np.exp(masked - row_max)
    probabilities = exp_scores / np.sum(exp_scores, axis = 1, keepdims =True)
    return probabilities
