"""Utility functions and classes"""

import numpy as np
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix


def _uniquify(seq, sep='-'):
    """Uniquify a list of strings.

    Adding unique numbers to duplicate values.

    Parameters
    ----------
    seq : `list` or `array-like`
        A list of values
    sep : `str`
        Separator

    Returns
    -------
    seq: `list` or `array-like`
        A list of updated values
    """

    dups = {}

    for i, val in enumerate(seq):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            seq[i] += (sep+str(dups[val][1]))

    return(seq)


def _gini(array):
    """Calculate the Gini coefficient of a numpy array.
    """

    array = array.flatten().astype(float)
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def _knn(X,
         k=20,
         leaf_size=40,
         metric='euclidean'):
    """Calculate K nearest neigbors for each row.
    """
    kdt = KDTree(X, leaf_size=leaf_size, metric=metric)
    kdt_d, kdt_i = kdt.query(X, k=k+1, return_distance=True)
    kdt_i = kdt_i[:, 1:]  # exclude the point itself
    kdt_d = kdt_d[:, 1:]  # exclude the point itself
    sp_row = np.repeat(np.arange(kdt_i.shape[0]), kdt_i.shape[1])
    sp_col = kdt_i.flatten()
    sp_conn = np.repeat(1, len(sp_row))
    sp_dist = kdt_d.flatten()
    mat_knn_conn = csr_matrix(
        (sp_conn, (sp_row, sp_col)),
        shape=(X.shape[0], X.shape[0]))
    mat_knn_dist = csr_matrix(
        (sp_dist, (sp_row, sp_col)),
        shape=(X.shape[0], X.shape[0]))
    return mat_knn_conn, mat_knn_dist
