"""Functions and classes for the analysis after PBG training"""

import numpy as np
import anndata as ad
from scipy.stats import entropy
from sklearn.neighbors import KDTree

from ._utils import _gini


def softmax(adata_ref,
            adata_query,
            T=0.5,
            n_top=None,
            percentile=0):
    """Softmax-based transformation

    This will transform query data to reference-comparable data

    Parameters
    ----------
    adata_ref: `AnnData`
        Reference anndata.
    adata_query: `list`
        Query anndata objects
    T: `float`
        Temperature parameter.
        It controls the output probability distribution.
        When T goes to inf, it becomes a discrete uniform distribution,
        each query becomes the average of reference;
        When T goes to zero, softargmax converges to arg max,
        each query is approximately the best of reference.
    cutoff: `float`
        The cutoff used to filter out low-probability reference entities
    Returns
    -------
    updates `adata_query` with the following field.
    softmax: `array_like` (`.layers['softmax']`)
        Store #observations × #dimensions softmax transformed data matrix.
    """

    scores_ref_query = np.matmul(adata_ref.X, adata_query.X.T)
    # avoid overflow encountered
    scores_ref_query = scores_ref_query - scores_ref_query.max()
    scores_softmax = np.exp(scores_ref_query/T) / \
        (np.exp(scores_ref_query/T).sum(axis=0))[None, :]
    if n_top is None:
        thresh = np.percentile(scores_softmax, q=percentile, axis=0)
    else:
        thresh = (np.sort(scores_softmax, axis=0)[::-1, :])[n_top-1, ]
    mask = scores_softmax < thresh[None, :]
    scores_softmax[mask] = 0
    # rescale to make scores add up to 1
    scores_softmax = scores_softmax/scores_softmax.sum(axis=0, keepdims=1)
    X_query = np.dot(scores_softmax.T, adata_ref.X)
    adata_query.layers['softmax'] = X_query


class SimbaEmbed:
    """A class used to represent post-training embedding analyis

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self,
                 adata_ref,
                 list_adata_query,
                 T=0.5,
                 list_T=None,
                 percentile=50,
                 n_top=None,
                 list_percentile=None,
                 use_precomputed=True,
                 ):
        """
        Parameters
        ----------
        adata_ref: `AnnData`
            Reference anndata.
        list_adata_query: `list`
            A list query anndata objects
        T: `float`
            Temperature parameter shared by all query adata objects.
            It controls the output probability distribution.
            when T goes to inf, it becomes a discrete uniform distribution,
            each query becomes the average of reference;
            when T goes to zero, softargmax converges to arg max,
            each query is approximately the best of reference.
        list_T: `list`, (default: None)
            A list of temperature parameters.
            It should correspond to each of query data.
            Once it's specified, it will override `T`.
        cutoff: `float`, (default: None)
            The cutoff used to filter out low-probability reference entities
        list_cutoff: `list`, (default: None)
            A list of cutoff values.
            It should correspond to each of query data.
            Once it's specified, it will override `cutoff`.
        """
        assert isinstance(list_adata_query, list), \
            "`list_adata_query` must be list"
        if list_T is not None:
            assert isinstance(list_T, list), \
                "`list_T` must be list"
        self.adata_ref = adata_ref
        self.list_adata_query = list_adata_query
        self.T = T
        self.list_T = list_T
        self.percentile = percentile
        self.n_top = n_top
        self.list_percentile = list_percentile
        self.use_precomputed = use_precomputed

    def embed(self):
        """Embed a list of query datasets along with reference dataset
        into the same space

        Returns
        -------
        adata_all: `AnnData`
            Store #entities × #dimensions.
        """
        adata_ref = self.adata_ref
        list_adata_query = self.list_adata_query
        use_precomputed = self.use_precomputed
        T = self.T
        list_T = self.list_T
        n_top = self.n_top
        percentile = self.percentile
        list_percentile = self.list_percentile
        X_all = adata_ref.X.copy()
        # obs_all = pd.DataFrame(
        #     data=['ref']*adata_ref.shape[0],
        #     index=adata_ref.obs.index,
        #     columns=['id_dataset'])
        obs_all = adata_ref.obs.copy()
        obs_all['id_dataset'] = ['ref']*adata_ref.shape[0]
        for i, adata_query in enumerate(list_adata_query):
            if list_T is not None:
                param_T = list_T[i]
            else:
                param_T = T
            if list_percentile is not None:
                param_percentile = list_percentile[i]
            else:
                param_percentile = percentile
            if use_precomputed:
                if 'softmax' in adata_query.layers.keys():
                    print(f'Reading in precomputed softmax-transformed matrix '
                          f'for query data {i};')
                else:
                    print(f'No softmax-transformed matrix exists '
                          f'for query data {i}')
                    print("Performing softmax transformation;")
                    softmax(
                        adata_ref,
                        adata_query,
                        T=param_T,
                        percentile=param_percentile,
                        n_top=n_top,
                    )
            else:
                print(f'Performing softmax transformation '
                      f'for query data {i};')
                softmax(
                    adata_ref,
                    adata_query,
                    T=param_T,
                    percentile=param_percentile,
                    n_top=n_top,
                    )
            X_all = np.vstack((X_all, adata_query.layers['softmax']))
            # obs_all = obs_all.append(
            #     pd.DataFrame(
            #         data=[f'query_{i}']*adata_query.shape[0],
            #         index=adata_query.obs.index,
            #         columns=['id_dataset'])
            #         )
            obs_query = adata_query.obs.copy()
            obs_query['id_dataset'] = [f'query_{i}']*adata_query.shape[0]
            obs_all = obs_all.append(obs_query, ignore_index=False)
        adata_all = ad.AnnData(X=X_all,
                               obs=obs_all)
        return adata_all


def embed(adata_ref,
          list_adata_query,
          T=0.5,
          list_T=None,
          percentile=0,
          n_top=None,
          list_percentile=None,
          use_precomputed=False):
    """Embed a list of query datasets along with reference dataset
    into the same space

    Parameters
    ----------
        adata_ref: `AnnData`
            Reference anndata.
        list_adata_query: `list`
            A list query anndata objects
        T: `float`
            Temperature parameter shared by all query adata objects.
            It controls the output probability distribution.
            when T goes to inf, it becomes a discrete uniform distribution,
            each query becomes the average of reference;
            when T goes to zero, softargmax converges to arg max,
            each query is approximately the best of reference.
        list_T: `list`, (default: None)
            A list of temperature parameters.
            It should correspond to each of query data.
            Once it's specified, it will override `T`.

    Returns
    -------
    adata_all: `AnnData`
        Store #entities × #dimensions.
    updates `adata_query` with the following field.
    softmax: `array_like` (`.layers['softmax']`)
        Store #observations × #dimensions softmax transformed data matrix.
    """
    SE = SimbaEmbed(adata_ref,
                    list_adata_query,
                    T=T,
                    list_T=list_T,
                    percentile=percentile,
                    n_top=n_top,
                    list_percentile=list_percentile,
                    use_precomputed=use_precomputed)
    adata_all = SE.embed()
    return adata_all


def compare_entities(adata_ref,
                     adata_query,
                     n_top_cells=50):
    """Compare the embeddings of two entities by calculating
    the following values between reference and query entities:
    - dot product
    - normalized dot product
    - softmax probability

    and the following metrics for each query entity
    - max (The average maximum dot product of top-rank reference entities,
      based on normalized dot product)
    - std (standard deviation of reference entities,
      based on dot product)
    - gini (Gini coefficients of reference entities,
      based on softmax probability)
    - entropy (The entropy of reference entities,
      based on softmax probability)

    Parameters
    ----------
        adata_ref: `AnnData`
            Reference entity anndata.
        adata_query: `list`
            Query entity anndata.
        n_top_cells: `int`, optional (default: 50)
            The number of cells to consider when calculating the metric 'max'

    Returns
    -------
    adata_cmp: `AnnData`
        Store reference entity as observations and query entity as variables
    """

    X_ref = adata_ref.X
    X_query = adata_query.X
    X_cmp = np.matmul(X_ref, X_query.T)
    adata_cmp = ad.AnnData(X=X_cmp,
                           obs=adata_ref.obs,
                           var=adata_query.obs)
    adata_cmp.layers['norm'] = X_cmp \
        - np.log(np.exp(X_cmp).mean(axis=0)).reshape(1, -1)
    adata_cmp.layers['softmax'] = np.exp(X_cmp) \
        / np.exp(X_cmp).sum(axis=0).reshape(1, -1)
    adata_cmp.var['max'] = \
        np.clip(np.sort(adata_cmp.layers['norm'], axis=0)[-n_top_cells:, ],
                a_min=0,
                a_max=None).mean(axis=0)
    adata_cmp.var['std'] = np.std(X_cmp, axis=0, ddof=1)
    adata_cmp.var['gini'] = np.array([_gini(adata_cmp.layers['softmax'][:, i])
                                      for i in np.arange(X_cmp.shape[1])])
    adata_cmp.var['entropy'] = entropy(adata_cmp.layers['softmax'])
    return adata_cmp


def query(adata,
          obsm='X_umap',
          layer=None,
          metric='euclidean',
          filters=None,
          anno_filters=None,
          entity=None,
          pin=None,
          k=20,
          use_radius=False,
          r=None,
          **kwargs
          ):
    """Query the "database" of entites
    """
    if(sum(list(map(lambda x: x is None,
                    [entity, pin]))) == 2):
        raise ValueError("One of `entity` and `pin` must be specified")
    if(sum(list(map(lambda x: x is not None,
                    [entity, pin]))) == 2):
        print("`entity` will be ignored.")

    if(sum(list(map(lambda x: x is not None,
                    [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif(obsm is not None):
        X = adata.obsm[obsm]
        if pin is None:
            pin = adata[entity, :].obsm[obsm].copy()
    elif(layer is not None):
        X = adata.layers[layer]
        if pin is None:
            pin = adata[entity, :].layers[layer].copy()
    else:
        X = adata.X
        if pin is None:
            pin = adata[entity, :].X.copy()
    pin = np.reshape(np.array(pin), [-1, 2])
    kdt = KDTree(X, metric=metric, **kwargs)
    if use_radius:
        if r is None:
            r = np.mean(X.max(axis=0) - X.min(axis=0))/5
        ind, dist = kdt.query_radius(pin,
                                     r=r,
                                     sort_results=True,
                                     return_distance=True)
        ind = ind[0].flatten()
        dist = dist[0].flatten()
    else:
        dist, ind = kdt.query(pin,
                              k=k+1,
                              sort_results=True,
                              return_distance=True)
        ind = ind.flatten()
        dist = dist.flatten()
    df_output = adata.obs.iloc[ind, ].copy()
    df_output['distance'] = dist
    if anno_filters is not None:
        if anno_filters in adata.obs_keys():
            if filters is None:
                filters = df_output[anno_filters].unique().tolist()
            df_output.query(f'{anno_filters} == @filters', inplace=True)
        else:
            raise ValueError(f'could not find {anno_filters}')
    adata.uns['query'] = dict()
    adata.uns['query']['params'] = {'obsm': obsm,
                                    'layer': layer,
                                    'entity': entity,
                                    'pin': pin,
                                    'k': k,
                                    'use_radius': use_radius,
                                    'r': r}
    adata.uns['query']['output'] = df_output.copy()
    return df_output
