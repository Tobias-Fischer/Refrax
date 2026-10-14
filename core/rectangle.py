"""Largest axis-aligned rectangle of valid pixels (crop of the corrected image)."""
import numpy as np


def _largest_rectangle_numpy(valid):
    """Row-wise maximal-rectangle DP (height / left / right per column), vectorised over
    columns. Ties are broken as in the stack-based reference: first row, then smallest
    right edge, then tallest."""
    H, W = valid.shape
    cols = np.arange(W)
    height = np.zeros(W, dtype=np.int64)
    left = np.zeros(W, dtype=np.int64)
    right = np.full(W, W, dtype=np.int64)
    best_area, best_box = 0, (0, H, 0, W)
    for y in range(H):
        row = valid[y]
        height = np.where(row, height + 1, 0)
        run_left = np.maximum.accumulate(np.where(row, 0, cols + 1))
        run_right = np.minimum.accumulate(np.where(row, W, cols)[::-1])[::-1]
        left = np.where(row, np.maximum(left, run_left), 0)
        right = np.where(row, np.minimum(right, run_right), W)
        area = height * (right - left)
        a = int(area.max())
        if a > best_area:
            cand = np.flatnonzero(area == a)
            r_min = right[cand].min()
            cand = cand[right[cand] == r_min]
            k = cand[np.argmax(height[cand])]
            best_area = a
            best_box = (y - int(height[k]) + 1, y + 1, int(left[k]), int(right[k]))
    return best_box


try:
    import numba as nb

    @nb.njit(cache=True, nogil=True)
    def _largest_rectangle_numba(valid):
        """The reference stack algorithm (remove_refraction.largest_valid_rectangle), compiled."""
        H, W = valid.shape
        height = np.zeros(W + 1, dtype=np.int64)
        st_x = np.empty(W + 1, dtype=np.int64)
        st_h = np.empty(W + 1, dtype=np.int64)
        best_area = 0
        b0, b1, b2, b3 = 0, H, 0, W
        for y in range(H):
            for x in range(W):
                height[x] = height[x] + 1 if valid[y, x] else 0
            n = 0
            for x in range(W + 1):
                h = height[x] if x < W else 0
                start = x
                while n > 0 and st_h[n - 1] > h:
                    n -= 1
                    s_x, s_h = st_x[n], st_h[n]
                    area = s_h * (x - s_x)
                    if area > best_area:
                        best_area = area
                        b0, b1, b2, b3 = y - s_h + 1, y + 1, s_x, x
                    start = s_x
                st_x[n] = start
                st_h[n] = h
                n += 1
        return b0, b1, b2, b3
except ImportError:  # pragma: no cover
    _largest_rectangle_numba = None


def largest_valid_rectangle(mask, backend="auto"):
    """Largest axis-aligned rectangle of non-zero mask pixels, as (y0, y1, x0, x1).

    Falls back to the whole image if the mask has no valid pixel.
    """
    valid = np.ascontiguousarray(np.asarray(mask) > 0)
    if backend != "numpy" and _largest_rectangle_numba is not None:
        return tuple(int(v) for v in _largest_rectangle_numba(valid))
    return _largest_rectangle_numpy(valid)
