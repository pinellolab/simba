"""Integration across experimental conditions or single cell modalities"""

import numpy as np
import anndata as ad
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix


def infer_edges(adata_ref,
                adata_query,
                feature='highly_variable',
                n_components=20,
                random_state=42,
                n_top_edges=6000,
                percentile=0.01,
                layer=None,
                metric='euclidean',
                **kwargs,
                ):
    """Identify edges across experimental conditions or single cell modalities
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields:
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
    if n_top_edges is None:
        cutoff = np.percentile(dist_ref_query.flatten(), percentile)
    else:
        cutoff = np.partition(dist_ref_query.flatten(),
                              n_top_edges-1)[n_top_edges-1]
    id_x, id_y = np.where(dist_ref_query <= cutoff)
    conn_ref_query = csr_matrix((dist_ref_query[id_x, id_y],
                                 (id_x, id_y)),
                                shape=dist_ref_query.shape)
    adata_ref_query = ad.AnnData(X=conn_ref_query,
                                 obs=adata_ref.obs,
                                 var=adata_query.obs)
    return adata_ref_query
