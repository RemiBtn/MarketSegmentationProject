import numpy as np


def as_barycenters(X, criteria_mins, criteria_maxs, L):
    P, n = X.shape
    X_bary = np.zeros((P, n, L + 1), dtype=np.float32)
    X = X.clip(criteria_mins, criteria_maxs)
    float_indices = np.clip(
        L * (X - criteria_mins) / (criteria_maxs - criteria_mins), 0, L
    )
    right_bary, indices = np.modf(float_indices)
    indices = indices.astype(np.int32)
    mask = indices != L
    X_bary[np.logical_not(mask), L] = 1
    X_bary[mask, indices[mask]] = 1 - right_bary[mask]
    X_bary[mask, indices[mask] + 1] = right_bary[mask]
    return X_bary


def compute_scores(utility_functions, X, criteria_mins, criteria_maxs):
    K, n, Lp1 = utility_functions.shape
    P = X.shape[0]
    X_bar = as_barycenters(X, criteria_mins, criteria_maxs, Lp1 - 1)
    utility_functions.shape = K, n * Lp1
    X_bar.shape = P, n * Lp1
    scores = np.dot(X_bar, utility_functions.T)
    utility_functions.shape = K, n, Lp1
    return scores


def prepare_XY(X, Y, criteria_mins, criteria_maxs):
    X_bar = as_barycenters(X, criteria_mins, criteria_maxs)
    Y_bar = as_barycenters(Y, criteria_mins, criteria_maxs)
    diff = X_bar - Y_bar
    diff_swapped_axes = np.moveaxis(diff, 0, -1)
    diff_swapped_axes = np.ascontiguousarray(diff_swapped_axes, dtype=np.float32)
    return diff_swapped_axes


def fast_explained_pairs(utility_functions, X_minus_Y_transp):
    *batch_shape, K, n, Lp1 = utility_functions.shape
    _n, _Lp1, P = X_minus_Y_transp.shape

    utility_functions.shape = *batch_shape, K, n * Lp1
    diff.shape = n * Lp1, P

    scores_diffs = np.dot(utility_functions, X_minus_Y_transp)
    explained = np.any(scores_diffs > 0, axis=-2)
    n_explained = np.count_nonzero(explained, axis=-1)

    utility_functions.shape = *batch_shape, K, n, Lp1
    diff.shape = n, Lp1, P

    return n_explained
