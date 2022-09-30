"""General preprocessing functions"""

import numpy as np
from sklearn.utils import sparsefuncs
from sklearn import preprocessing
from ._utils import (
    cal_tf_idf
)
from scipy.sparse import (
    issparse,
    csr_matrix,
)


def log_transform(adata):
    """Return the natural logarithm of one plus the input array, element-wise.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes logarithmized data matrix.
    """
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    adata.X = np.log1p(adata.X)
    return None


def binarize(adata,
             threshold=1e-5):
    """Binarize an array.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    threshold: `float`, optional (default: 1e-5)
        Values below or equal to this are replaced by 0, above it by 1.

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes binarized data matrix.
    """
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    adata.X = preprocessing.binarize(adata.X,
                                     threshold=threshold,
                                     copy=True)


def normalize(adata,
              method='lib_size',
              scale_factor=1e4,
              save_raw=True):
    """Normalize count matrix.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    method: `str`, optional (default: 'lib_size')
        Choose from {{'lib_size','tf_idf'}}
        Method used for dimension reduction.
        'lib_size': Total-count normalize (library-size correct)
        'tf_idf': TF-IDF (term frequency–inverse document frequency)
        transformation

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes normalized data matrix.
    """
    if method not in ['lib_size', 'tf_idf']:
        raise ValueError("unrecognized method '%s'" % method)
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    if save_raw:
        adata.layers['raw'] = adata.X.copy()
    if method == 'lib_size':
        sparsefuncs.inplace_row_scale(adata.X, 1/adata.X.sum(axis=1).A)
        adata.X = adata.X*scale_factor
    if method == 'tf_idf':
        adata.X = cal_tf_idf(adata.X)
