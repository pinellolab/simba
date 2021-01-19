"""General-purpose tools"""

from sklearn.preprocessing import KBinsDiscretizer


def discretize(adata,
               layer=None,
               n_bins=3,
               encode='ordinal',
               strategy='kmeans',
               dtype=None):
    """Discretize continous features
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
    if layer is None:
        X = adata.X.copy()
    else:
        X = adata.layers[layer].copy()
    est = KBinsDiscretizer(n_bins=n_bins,
                           encode=encode,
                           strategy=strategy,
                           dtype=dtype)
    nonzero_cont = X.data.copy()
    nonzero_id = est.fit_transform(nonzero_cont.reshape(-1, 1))
    nonzero_disc = est.inverse_transform(nonzero_id).reshape(-1, )

    adata.layers['disc'] = adata.layers['raw'].copy()
    adata.layers['disc'].data = (nonzero_id+1).reshape(-1,)

    # discretized data transformed back to original feature space
    adata.uns['disc'] = dict()
    adata.uns['disc']['disc_ori'] = adata.layers['raw'].copy()
    adata.uns['disc']['disc_ori'].data = nonzero_disc.reshape(-1,)
    adata.uns['disc']['bin_edges'] = est.bin_edges_
