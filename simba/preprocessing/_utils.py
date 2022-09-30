"""Utility functions and classes"""

import numpy as np
from kneed import KneeLocator
from scipy.sparse import csr_matrix, diags


def locate_elbow(x, y, S=10, min_elbow=0,
                 curve='convex', direction='decreasing', online=False,
                 **kwargs):
    """Detect knee points

    Parameters
    ----------
    x : `array_like`
        x values
    y : `array_like`
        y values
    S : `float`, optional (default: 10)
        Sensitivity
    min_elbow: `int`, optional (default: 0)
        The minimum elbow location
    curve: `str`, optional (default: 'convex')
        Choose from {'convex','concave'}
        If 'concave', algorithm will detect knees,
        If 'convex', algorithm will detect elbows.
    direction: `str`, optional (default: 'decreasing')
        Choose from {'decreasing','increasing'}
    online: `bool`, optional (default: False)
        kneed will correct old knee points if True,
        kneed will return first knee if False.
    **kwargs: `dict`, optional
        Extra arguments to KneeLocator.

    Returns
    -------
    elbow: `int`
        elbow point
    """
    kneedle = KneeLocator(x[int(min_elbow):], y[int(min_elbow):],
                          S=S, curve=curve,
                          direction=direction,
                          online=online,
                          **kwargs,
                          )
    if kneedle.elbow is None:
        elbow = len(y)
    else:
        elbow = int(kneedle.elbow)
    return elbow


def cal_tf_idf(mat):
    """Transform a count matrix to a tf-idf representation
    """
    mat = csr_matrix(mat)
    tf = csr_matrix(mat/(mat.sum(axis=0)))
    idf = np.array(np.log(1 + mat.shape[1] / mat.sum(axis=1))).flatten()
    tf_idf = csr_matrix(np.dot(diags(idf), tf))
    return tf_idf
