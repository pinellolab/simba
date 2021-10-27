"""General-purpose tools"""

import numpy as np
from sklearn.cluster import KMeans


def discretize(adata,
               layer=None,
               n_bins=5,
               max_bins=100):
    """Discretize continous values

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    layer: `str`, optional (default: None)
        The layer used to perform discretization
    n_bins: `int`, optional (default: 5)
        The number of bins to produce.
        It must be smaller than `max_bins`.
    max_bins: `int`, optional (default: 100)
        The number of bins used in the initial approximation.
        i.e. the number of bins to cluster.

    Returns
    -------
    updates `adata` with the following fields

    `.layer['disc']` : `array_like`
        Discretized values.
    `.uns['disc']` : `dict`
        `bin_edges`: The edges of each bin.
        `bin_count`: The number of values in each bin.
        `hist_edges`: The edges of each bin \
                      in the initial approximation.
        `hist_count`: The number of values in each bin \
                      for the initial approximation.
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    nonzero_cont = X.data

    hist_count, hist_edges = np.histogram(
        nonzero_cont,
        bins=max_bins,
        density=False)
    hist_centroids = (hist_edges[0:-1] + hist_edges[1:])/2

    kmeans = KMeans(n_clusters=n_bins, random_state=2021).fit(
        hist_centroids.reshape(-1, 1),
        sample_weight=hist_count)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

    padding = (hist_edges[-1] - hist_edges[0])/(max_bins*10)
    bin_edges = np.array(
        [hist_edges[0]-padding] +
        list((cluster_centers[0:-1] + cluster_centers[1:])/2) +
        [hist_edges[-1]+padding])
    nonzero_disc = np.digitize(nonzero_cont, bin_edges).reshape(-1,)
    bin_count = np.unique(nonzero_disc, return_counts=True)[1]

    adata.layers['disc'] = X.copy()
    adata.layers['disc'].data = nonzero_disc
    adata.uns['disc'] = dict()
    adata.uns['disc']['bin_edges'] = bin_edges
    adata.uns['disc']['bin_count'] = bin_count
    adata.uns['disc']['hist_edges'] = hist_edges
    adata.uns['disc']['hist_count'] = hist_count
