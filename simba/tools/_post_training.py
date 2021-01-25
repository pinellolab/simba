"""Functions and classes for the analysis after PBG training"""

import numpy as np


def softmax_transform(adata_ref,
                      list_adata_query,
                      T=1):
    """Softmax-based transformation

    This will transform query data to reference-comparable data

    Parameters
    ----------
    adata_ref: `AnnData`
        Reference anndata.
    adata_query: `AnnData`
        Query anndata
    T: `float`
        Temperature parameter to control the output probability distribution.
        when T goes to inf, it becomes a discrete uniform distribution,
        each query becomes the average of reference;
        when T goes to zero, softargmax converges to arg max,
        each query is approximately the best of reference.

    Returns
    -------
    updates `adata_query` with the following field.
    softmax: `array_like` (`.layers['softmax']`)
        Store #observations × #dimensions softmax transformed data matrix.
    """

    assert isinstance(list_adata_query, list), \
        "`list_adata_query` must be list"
    for adata_query in list_adata_query:
        scores_ref_query = np.matmul(adata_ref.X, adata_query.X.T)
        scores_softmax = np.exp(scores_ref_query/T) / \
            (np.exp(scores_ref_query/T).sum(axis=0))[None, :]
        X_query = np.dot(scores_softmax.T, adata_ref.X)
        adata_query.layers['softmax'] = X_query
