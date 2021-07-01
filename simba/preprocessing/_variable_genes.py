"""Preprocess"""

import numpy as np
from scipy.sparse import (
    csr_matrix,
)
from sklearn.utils import sparsefuncs
from skmisc.loess import loess


def select_variable_genes(adata,
                          layer='raw',
                          span=0.3,
                          n_top_genes=2000,
                          ):
    """Select highly variable genes.

    This function implenments the method 'vst' in Seurat v3.
    Inspired by Scanpy.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    layer: `str`, optional (default: 'raw')
        The layer to use for calculating variable genes.
    span: `float`, optional (default: 0.3)
        Loess smoothing factor
    n_top_genes: `int`, optional (default: 2000)
        The number of genes to keep

    Returns
    -------
    updates `adata` with the following fields.

    variances_norm: `float`, (`adata.var['variances_norm']`)
        Normalized variance per gene
    variances: `float`, (`adata.var['variances']`)
        Variance per gene.
    means: `float`, (`adata.var['means']`)
        Means per gene
    highly_variable: `bool` (`adata.var['highly_variable']`)
        Indicator of variable genes
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer].astype(np.float64).copy()
    mean, variance = sparsefuncs.mean_variance_axis(X, axis=0)
    variance_expected = np.zeros(adata.shape[1], dtype=np.float64)
    not_const = variance > 0

    model = loess(np.log10(mean[not_const]),
                  np.log10(variance[not_const]),
                  span=span,
                  degree=2)
    model.fit()
    variance_expected[not_const] = 10**model.outputs.fitted_values
    N = adata.shape[0]
    clip_max = np.sqrt(N)
    clip_val = np.sqrt(variance_expected) * clip_max + mean

    X = csr_matrix(X)
    mask = X.data > clip_val[X.indices]
    X.data[mask] = clip_val[X.indices[mask]]

    squared_X_sum = np.array(X.power(2).sum(axis=0))
    X_sum = np.array(X.sum(axis=0))

    norm_gene_var = (1 / ((N - 1) * variance_expected)) \
        * ((N * np.square(mean))
            + squared_X_sum
            - 2 * X_sum * mean
           )
    norm_gene_var = norm_gene_var.flatten()

    adata.var['variances_norm'] = norm_gene_var
    adata.var['variances'] = variance
    adata.var['means'] = mean
    ids_top = norm_gene_var.argsort()[-n_top_genes:][::-1]
    adata.var['highly_variable'] = np.isin(range(adata.shape[1]), ids_top)
    print(f'{n_top_genes} variable genes are selected.')
