def gradient_accumulator(
    theta: float, eta: float, microbatches: list[tuple[int, float]]
) -> tuple[float, float]:
    """
    microbatches: list of (n_j, g_j) pairs, n_j examples with mean
    gradient g_j.

    g = sum(n_j * g_j) / sum(n_j)   (example-weighted mean, NOT the mean
    of the microbatch means).
    theta' = theta - eta * g.

    Return (g, theta_prime).
    """
    # TODO: weight by example count, divide exactly once.
    weighted_g = 0.0
    total_example = 0
    for n_j, g_j in microbatches:
        weighted_g += n_j * g_j
        total_example += n_j
        
    g = weighted_g  / total_example
    theta_prime = theta - eta * g
    return (g, theta_prime)
