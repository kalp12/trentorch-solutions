def cache_hit_rate(keys: list[int]) -> tuple[int, int, float]:
    """
    keys: a sequence of request keys in arrival order.

    A request is a hit iff its key has appeared earlier in the sequence
    (the first occurrence of any key is always a miss).

    Return (hits, misses, rate) where rate = hits / len(keys).
    """
    # TODO: track which keys have been seen so far, in order.
    hits = 0
    misses = 0
    seen = set()
    for k in keys:
        if k in seen:
            hits += 1
        else:
            misses += 1
            seen.add(k)

    rate = hits / len(keys)
    return (hits, misses, rate)
