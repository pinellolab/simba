"""Integration across experimental conditions or single cell modalities"""

import numpy as np
import anndata as ad
# from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix, find

from ._utils import _knn


def infer_edges(adata_ref,
                adata_query,
                feature='highly_variable',
                n_components=20,
                random_state=42,
                layer=None,
                k=20,
                metric='euclidean',
                leaf_size=40,
                **kwargs):
    """Infer edges between reference and query observations

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
    k: `int`, optional (default: 5)
        The number of nearest neighbors to consider within each dataset
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
    print(f'#shared features: {len(feature_shared)}')
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

    print('Performing randomized SVD ...')
    mat = X_ref * X_query.T
    U, Sigma, VT = randomized_svd(mat,
                                  n_components=n_components,
                                  random_state=random_state,
                                  **kwargs)
    svd_data = np.vstack((U, VT.T))
    X_svd_ref = svd_data[:U.shape[0], :]
    X_svd_query = svd_data[-VT.shape[1]:, :]
    X_svd_ref = X_svd_ref / (X_svd_ref**2).sum(-1, keepdims=True)**0.5
    X_svd_query = X_svd_query / (X_svd_query**2).sum(-1, keepdims=True)**0.5

    # print('Searching for neighbors within each dataset ...')
    # knn_conn_ref, knn_dist_ref = _knn(
    #     X_ref=X_svd_ref,
    #     k=k,
    #     leaf_size=leaf_size,
    #     metric=metric)
    # knn_conn_query, knn_dist_query = _knn(
    #     X_ref=X_svd_query,
    #     k=k,
    #     leaf_size=leaf_size,
    #     metric=metric)

    print('Searching for mutual nearest neighbors ...')
    knn_conn_ref_query, knn_dist_ref_query = _knn(
        X_ref=X_svd_ref,
        X_query=X_svd_query,
        k=k,
        leaf_size=leaf_size,
        metric=metric)
    knn_conn_query_ref, knn_dist_query_ref = _knn(
        X_ref=X_svd_query,
        X_query=X_svd_ref,
        k=k,
        leaf_size=leaf_size,
        metric=metric)

    sum_conn_ref_query = knn_conn_ref_query + knn_conn_query_ref.T
    id_x, id_y, values = find(sum_conn_ref_query > 1)
    print(f'{len(id_x)} edges are selected')
    conn_ref_query = csr_matrix(
        (values*1, (id_x, id_y)),
        shape=(knn_conn_ref_query.shape))
    dist_ref_query = csr_matrix(
        (knn_dist_ref_query[id_x, id_y].A.flatten(), (id_x, id_y)),
        shape=(knn_conn_ref_query.shape))
    # it's easier to distinguish zeros (no connection vs zero distance)
    # using similarity scores
    sim_ref_query = csr_matrix(
        (1/(dist_ref_query.data+1), dist_ref_query.nonzero()),
        shape=(dist_ref_query.shape))  # similarity scores

    # print('Computing similarity scores ...')
    # dist_ref_query = pairwise_distances(X_svd_ref,
    #                                     X_svd_query,
    #                                     metric=metric)
    # sim_ref_query = 1/(1+dist_ref_query)
    # # remove low similarity entries to save memory
    # sim_ref_query = np.where(
    #     sim_ref_query < np.percentile(sim_ref_query, pct_keep*100),
    #     0, sim_ref_query)
    # sim_ref_query = csr_matrix(sim_ref_query)

    adata_ref_query = ad.AnnData(X=sim_ref_query,
                                 obs=adata_ref.obs,
                                 var=adata_query.obs)
    adata_ref_query.layers['conn'] = conn_ref_query
    adata_ref_query.obsm['svd'] = X_svd_ref
    # adata_ref_query.obsp['conn'] = knn_conn_ref
    # adata_ref_query.obsp['dist'] = knn_dist_ref
    adata_ref_query.varm['svd'] = X_svd_query
    # adata_ref_query.varp['conn'] = knn_conn_query
    # adata_ref_query.varp['dist'] = knn_dist_query
    return adata_ref_query


def trim_edges(adata_ref_query,
               cutoff=None,
               n_edges=None):
    """Trim edges based on the similarity scores

    Parameters
    ----------
    adata_ref_query: `AnnData`
         Annotated relation matrix betwewn reference and query observations.
    n_edges: `int`, optional (default: None)
        The number of edges to keep
        If specified, `percentile` will be ignored
    cutoff: `float`, optional (default: None)
        The distance cutoff.
        If None, it will be decided by `n_top_edges`
        If specified, `n_top_edges` will be ignored

    Returns
    -------
    updates `adata_ref_query` with the following field.
    `.layers['conn']` : `array_like`
        relation matrix betwewn reference and query observations
    """
    sim_ref_query = adata_ref_query.X
    if cutoff is None:
        if n_edges is None:
            raise ValueError('"cutoff" or "n_edges" has to be specified')
        else:
            cutoff = \
                np.partition(sim_ref_query.data,
                             (sim_ref_query.size-n_edges))[
                                 sim_ref_query.size-n_edges]
            # cutoff = \
            #     np.partition(sim_ref_query.flatten(),
            #                  (len(sim_ref_query.flatten())-n_edges))[
            #                      len(sim_ref_query.flatten())-n_edges]
    id_x, id_y, values = find(sim_ref_query > cutoff)

    print(f'{len(id_x)} edges are selected')
    conn_ref_query = csr_matrix(
        (values*1, (id_x, id_y)),
        shape=(sim_ref_query.shape))
    adata_ref_query.layers['conn'] = conn_ref_query
