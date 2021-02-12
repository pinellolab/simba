"""Integration across experimental conditions or single cell modalities"""

import numpy as np
import anndata as ad
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix


def node_dist(adata_ref,
              adata_query,
              feature='highly_variable',
              n_components=20,
              random_state=42,
              layer=None,
              metric='euclidean',
              **kwargs,
              ):
    """node distances across experimental conditions or single cell modalities

    Parameters
    ----------
    adata_ref: `AnnData`
        Annotated reference data matrix.
    adata_query: `AnnData`
        Annotated query data matrix.
    feature: `str`, optional (default: None)
        Feature used for edges inference.
        The data type of `.var[feature]` needs to be `bool`
    n_components: `int`, optional (default: 20)
        The number of components used in `randomized_svd`
        for comparing reference and query observations
    random_state: `int`, optional (default: 42)
        The seed used for truncated randomized SVD
    n_top_edges: `int`, optional (default: None)
        The number of edges to keep
        If specified, `percentile` will be ignored
    percentile: `float`, optional (default: 0.01)
        The percentile of edges to keep
    cutoff: `float`, optional (default: None)
        The distance cutoff.
        If None, it will be decided by `n_top_edges` or `percentile`
        If specified, `n_top_edges` and `percentile` will be ignored
    metric: `str`, optional (default: 'euclidean')
        The metric to use when calculating distance between
        reference and query observations
    layer: `str`, optional (default: None)
        The layer used to perform edge inference
        If None, `.X` will be used.
    kwargs:
        Other keyword arguments are passed down to `randomized_svd()`

    Returns
    -------
    adata_ref_query: `AnnData`
        Annotated relation matrix betwewn reference and query observations
        Store reference entity as observations and query entity as variables
    """
    mask_ref = adata_ref.var[feature]
    feature_ref = adata_ref.var_names[mask_ref]
    feature_query = adata_query.var_names
    feature_shared = list(set(feature_ref).intersection(set(feature_query)))
    if layer is None:
        X_ref = adata_ref[:, feature_shared].X
        X_query = adata_query[:, feature_shared].X
    else:
        X_ref = adata_ref[:, feature_shared].layers[layer]
        X_query = adata_query[:, feature_shared].layers[layer]

    if any(X_ref.sum(axis=1) == 0) or any(X_query.sum(axis=1) == 0):
        raise ValueError(
            f'Some nodes contain zero expressed {feature} features.\n'
            f'Please try to include more {feature} features.')
    mat = X_ref * X_query.T

    U, Sigma, VT = randomized_svd(mat,
                                  n_components=n_components,
                                  random_state=random_state,
                                  **kwargs)
    cca_data = np.vstack((U, VT.T))
    X_cca_ref = cca_data[:U.shape[0], :]
    X_cca_query = cca_data[-VT.shape[1]:, :]
    X_cca_ref = X_cca_ref / (X_cca_ref**2).sum(-1, keepdims=True)**0.5
    X_cca_query = X_cca_query / (X_cca_query**2).sum(-1, keepdims=True)**0.5

    dist_ref_query = pairwise_distances(X_cca_ref,
                                        X_cca_query,
                                        metric=metric)
    conn_ref_query = csr_matrix(dist_ref_query.shape)
    adata_ref_query = ad.AnnData(X=conn_ref_query,
                                 obs=adata_ref.obs,
                                 var=adata_query.obs)
    adata_ref_query.layers['dist'] = dist_ref_query
    adata_ref_query.obsm['cca'] = X_cca_ref
    adata_ref_query.varm['cca'] = X_cca_query
    return adata_ref_query


def infer_edges(adata_ref_query,
                n_top_edges=None,
                percentile=0.015,
                cutoff=None,
                ):
    """Infer edges from pairwise distance between reference and query observations

    Parameters
    ----------
    adata_ref_query: `AnnData`
         Annotated relation matrix betwewn reference and query observations.
    n_top_edges: `int`, optional (default: None)
        The number of edges to keep
        If specified, `percentile` will be ignored
    percentile: `float`, optional (default: 0.01)
        The percentile of edges to keep
    cutoff: `float`, optional (default: None)
        The distance cutoff.
        If None, it will be decided by `n_top_edges` or `percentile`
        If specified, `n_top_edges` and `percentile` will be ignored

    Returns
    -------
    updates `adata_ref_query` with the following field.
    X: `array_like`
        relation matrix betwewn reference and query observations
    """
    dist_ref_query = adata_ref_query.layers['dist']

    if cutoff is None:
        if n_top_edges is None:
            cutoff = np.percentile(dist_ref_query.flatten(), percentile)
        else:
            cutoff = np.partition(dist_ref_query.flatten(),
                                  n_top_edges-1)[n_top_edges-1]
    id_x, id_y = np.where(dist_ref_query <= cutoff)
    print(f'{len(id_x)} edges are selected')
    conn_ref_query = csr_matrix((dist_ref_query[id_x, id_y],
                                 (id_x, id_y)),
                                shape=dist_ref_query.shape)
    adata_ref_query.X = conn_ref_query
