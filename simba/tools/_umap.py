"""UMAP (Uniform Manifold Approximation and Projection)"""

import umap as umap_learn


def umap(adata,
         n_neighbors=15,
         n_components=2,
         random_state=2020,
         layer=None,
         obsm=None,
         n_dim=None,
         **kwargs,
         ):
    """perform UMAP
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    n_neighbors: `int`, optional (default: 15)
        The size of local neighborhood for UMAP
    n_components: `int`, optional (default: None)
        The dimension of the space to embed into for UMAP
    random_state: `int`, optional (default: None)
        The seed used by the random number generator for UMAP
    layer: `str`, optional (default: None)
        The layer used to perform UMAP
    obsm: `str`, optional (default: None)
        The multi-dimensional annotation of observations used to perform UMAP
    n_dim: `str`, optional (default: None)
        The number of dimensions used in `layer` or `obsm`
    kwargs:
        Other keyword arguments are passed down to `umap_learn.UMAP`

    Returns
    -------
    updates `adata` with the following fields:
    `.obsm['X_umap']` : `array`
        UMAP coordinates of samples.
    """

    if sum(list(map(lambda x: x is not None,
                    [layer, obsm]))) == 2:
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif obsm is not None:
        X = adata.obsm[obsm]
    elif layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    if n_dim is not None:
        X = X[:, :n_dim]
    reducer = umap_learn.UMAP(n_neighbors=n_neighbors,
                              n_components=n_components,
                              random_state=random_state,
                              **kwargs)
    reducer.fit(X)
    adata.obsm['X_umap'] = reducer.embedding_
