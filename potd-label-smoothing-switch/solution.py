import numpy as np


def label_smoothing(logits: np.ndarray, y: int, eps: float) -> tuple[np.ndarray, float]:
    """
    Build the label-smoothed target distribution over C = len(logits)
    classes and compute its cross-entropy loss against logits.

    q[c] = 1 - eps + eps/C for c == y, else eps/C.
    loss = -sum(q * log_softmax(logits)).

    Return (q, loss).
    """
    # TODO: smooth EVERY class by eps/C first, then add the remaining
    # mass (1 - eps) onto the true class y.
    C = len(logits)
    q = np.full(C, eps / C)

    q[y] += 1 - eps

    shifted = logits - np.max(logits)
    log_probs = shifted - np.log(np.sum(np.exp(shifted)))

    loss = -np.sum(q * log_probs)
    return q, float(loss)
