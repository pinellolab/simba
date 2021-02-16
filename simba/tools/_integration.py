"""Integration across experimental conditions or single cell modalities"""

import numpy as np
import anndata as ad
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix, find


def node_similarity(adata_ref,
                    adata_query,
                    feature='highly_variable',
                    n_components=20,
                    random_state=42,
                    layer=None,
                    metric='euclidean',
                    pct_keep=0.5,
                    **kwargs,
                    ):
    """Compute similarity between nodes

    These nodes of different experimental conditions or single cell modalities

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

    print('Performing CCA ...')
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

    # print('Searching for neighbors ...')
    # kdt = KDTree(X_cca_ref, leaf_size=leaf_size, metric=metric)
    # kdt_d, kdt_i = kdt.query(X_cca_query, k=k, return_distance=True)
    # sp_row = kdt_i.flatten()
    # sp_col = np.repeat(np.arange(kdt_i.shape[0]), kdt_i.shape[1])
    # sp_data = 1/(1+kdt_d.flatten())  # convert distance to similarity
    # sim_ref_query = csr_matrix(
    #     (sp_data, (sp_row, sp_col)),
    #     shape=(X_cca_ref.shape[0], X_cca_query.shape[0]))

    print('Computing similarity scores ...')
    dist_ref_query = pairwise_distances(X_cca_ref,
                                        X_cca_query,
                                        metric=metric)
    sim_ref_query = 1/(1+dist_ref_query)
    # remove low similarity entries to save memory
    sim_ref_query = np.where(
        sim_ref_query < np.percentile(sim_ref_query, pct_keep*100),
        0, sim_ref_query)
    sim_ref_query = csr_matrix(sim_ref_query)
    adata_ref_query = ad.AnnData(X=sim_ref_query,
                                 obs=adata_ref.obs,
                                 var=adata_query.obs)
    adata_ref_query.obsm['cca'] = X_cca_ref
    adata_ref_query.varm['cca'] = X_cca_query
    return adata_ref_query


def infer_edges(adata_ref_query,
                cutoff=None,
                n_edges=None,
                ):
    """Infer edges from pairwise distance between reference and query observations

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
    id_x, id_y, values = find(sim_ref_query >= cutoff)

    print(f'{len(id_x)} edges are selected')
    conn_ref_query = csr_matrix(
        (values, (id_x, id_y)),
        shape=(sim_ref_query.shape))
    adata_ref_query.layers['conn'] = conn_ref_query
