import math
def conv_output_shape(
    h: int, w: int, k: int, p: int, s: int
) -> tuple[int, int]:
    """
    h, w: input height and width.
    k: kernel size (applied to both axes).
    p: symmetric padding (applied to both axes).
    s: stride (applied to both axes).

    O = floor((I + 2*p - k) / s) + 1, applied independently to h and w.

    Return (out_h, out_w).
    """
    out_h = math.floor((h + 2 * p - k) / s) + 1
    out_w = math.floor((w + 2 * p - k) / s) + 1
    return (out_h, out_w)
