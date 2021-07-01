"""Functions and classes for the analysis after PBG training"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import entropy
from sklearn.neighbors import KDTree
from scipy.spatial import distance
# import faiss

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
                     n_top_cells=50,
                     T=1):
    """Compare the embeddings of two entities by calculating

    the following values between reference and query entities:

    - dot product
    - normalized dot product
    - softmax probability

    and the following metrics for each query entity:

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
    T: `float`
        Temperature parameter for softmax.
        It controls the output probability distribution.
        When T goes to inf, it becomes a discrete uniform distribution,
        each query becomes the average of reference;
        When T goes to zero, softargmax converges to arg max,
        each query is approximately the best of reference.

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
    adata_cmp.layers['softmax'] = np.exp(X_cmp/T) \
        / np.exp(X_cmp/T).sum(axis=0).reshape(1, -1)
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
          anno_filter=None,
          filters=None,
          entity=None,
          pin=None,
          k=20,
          use_radius=False,
          r=None,
          **kwargs
          ):
    """Query the "database" of entites

    Parameters
    ----------
    adata : `AnnData`
        Anndata object to query.
    obsm : `str`, optional (default: "X_umap")
        The multi-dimensional annotation to use for calculating the distance.
    layer : `str`, optional (default: None)
        The layer to use for calculating the distance.
    metric : `str`, optional (default: "euclidean")
        The distance metric to use.
        More metrics can be found at "`DistanceMetric class
        <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html>`__"
    anno_filter : `str`, optional (default: None)
        The annotation of filter to use.
        It should be one of ``adata.obs_keys()``
    filters : `list`, optional (default: None)
        The filters to use.
        It should be a list of values in ``adata.obs[anno_filter]``
    entity : `list`, optional (default: None)
        Query entity. It needs to be in ``adata.obs_names()``
    k : `int`, optional (default: 20)
        The number of nearest neighbors to return.
        Only valid if ``use_radius`` is False
    use_radius : `bool`, optional (default: False)
        If True, query for neighbors within a given radius
    r: `float`, optional (default: None)
        Distance within which neighbors are returned.
        If None, it will be estimated based the range of the space.
    **kwargs: `dict`, optional
        Extra arguments to ``sklearn.neighbors.KDTree``.

    Returns
    -------
    updates `adata` with the following fields.

    params: `dict`, (`adata.uns['query']['params']`)
        Parameters used for the query
    output: `pandas.DataFrame`, (`adata.uns['query']['output']`)
        Query result.
    """
    if(sum(list(map(lambda x: x is None,
                    [entity, pin]))) == 2):
        raise ValueError("One of `entity` and `pin` must be specified")
    if(sum(list(map(lambda x: x is not None,
                    [entity, pin]))) == 2):
        print("`entity` will be ignored.")
    if entity is not None:
        entity = np.array(entity).flatten()

    if(sum(list(map(lambda x: x is not None,
                    [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    elif(obsm is not None):
        X = adata.obsm[obsm].copy()
        if pin is None:
            pin = adata[entity, :].obsm[obsm].copy()
    elif(layer is not None):
        X = adata.layers[layer].copy()
        if pin is None:
            pin = adata[entity, :].layers[layer].copy()
    else:
        X = adata.X.copy()
        if pin is None:
            pin = adata[entity, :].X.copy()
    pin = np.reshape(np.array(pin), [-1, X.shape[1]])

    if use_radius:
        kdt = KDTree(X, metric=metric, **kwargs)
        if r is None:
            r = np.mean(X.max(axis=0) - X.min(axis=0))/5
        ind, dist = kdt.query_radius(pin,
                                     r=r,
                                     sort_results=True,
                                     return_distance=True)
        df_output = pd.DataFrame()
        for ii in np.arange(pin.shape[0]):
            df_output_ii = adata.obs.iloc[ind[ii], ].copy()
            df_output_ii['distance'] = dist[ii]
            if entity is not None:
                df_output_ii['query'] = entity[ii]
            else:
                df_output_ii['query'] = ii
            df_output = df_output.append(df_output_ii)
        if anno_filter is not None:
            if anno_filter in adata.obs_keys():
                if filters is None:
                    filters = df_output[anno_filter].unique().tolist()
                df_output.query(f'{anno_filter} == @filters', inplace=True)
            else:
                raise ValueError(f'could not find {anno_filter}')
        df_output = df_output.sort_values(by='distance')
    else:
        # assert (metric in ['euclidean', 'dot_product']),\
        #             "`metric` must be one of ['euclidean','dot_product']"
        if anno_filter is not None:
            if anno_filter in adata.obs_keys():
                if filters is None:
                    filters = adata.obs[anno_filter].unique().tolist()
                ids_filters = \
                    np.where(np.isin(adata.obs[anno_filter], filters))[0]
            else:
                raise ValueError(f'could not find {anno_filter}')
        else:
            ids_filters = np.arange(X.shape[0])
        kdt = KDTree(X[ids_filters, :], metric=metric, **kwargs)
        dist, ind = kdt.query(pin,
                              k=k,
                              sort_results=True,
                              return_distance=True)

        # use faiss
        # X = X.astype('float32')
        # if metric == 'euclidean':
        #     faiss_index = faiss.IndexFlatL2(X.shape[1])   # build the index
        #     faiss_index.add(X[ids_filters, :])
        #     dist, ind = faiss_index.search(pin, k)
        # if metric == 'dot_product':
        #     faiss_index = faiss.IndexFlatIP(X.shape[1])   # build the index
        #     faiss_index.add(X[ids_filters, :])
        #     sim, ind = faiss_index.search(pin, k)
        #     dist = -sim

        df_output = pd.DataFrame()
        for ii in np.arange(pin.shape[0]):
            df_output_ii = \
                adata.obs.iloc[ids_filters, ].iloc[ind[ii, ], ].copy()
            df_output_ii['distance'] = dist[ii, ]
            if entity is not None:
                df_output_ii['query'] = entity[ii]
            else:
                df_output_ii['query'] = ii
            df_output = df_output.append(df_output_ii)
        df_output = df_output.sort_values(by='distance')

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


def find_master_regulators(adata_all,
                           list_tf_motif=None,
                           list_tf_gene=None,
                           metric='euclidean',
                           anno_filter='entity_anno',
                           filter_gene='gene',
                           metrics_gene=None,
                           metrics_motif=None,
                           cutoff_gene_max=1.5,
                           cutoff_gene_gini=0.3,
                           cutoff_gene_std=None,
                           cutoff_gene_entropy=None,
                           cutoff_motif_max=1.5,
                           cutoff_motif_gini=0.3,
                           cutoff_motif_std=None,
                           cutoff_motif_entropy=None,
                           ):
    """Find all the master regulators

    Parameters
    ----------
    adata_all : `AnnData`
        Anndata object storing SIMBA embedding of all entities.
    list_tf_motif : `list`
        A list of TF motifs. They should match TF motifs in `list_tf_gene`.
    list_tf_gene : `list`
        A list TF genes. They should match TF motifs in `list_tf_motif`.
    metric : `str`, optional (default: "euclidean")
        The distance metric to use. It can be ‘braycurtis’, ‘canberra’,
        ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
        ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
        ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    anno_filter : `str`, optional (default: None)
        The annotation of filter to use.
        It should be one of ``adata.obs_keys()``
    filter_gene : `str`, optional (default: None)
        The filter for gene.
        It should be in ``adata.obs[anno_filter]``
    metrics_gene : `pandas.DataFrame`, optional (default: None)
        SIMBA metrics for genes.
    metrics_motif : `pandas.DataFrame`, optional (default: None)
        SIMBA metrics for motifs.
    cutoff_gene_max, cutoff_motif_max: `float`
        cutoff of SIMBA metric `max value` for genes and motifs
    cutoff_gene_gini,  cutoff_motif_gini: `float`
        cutoff of SIMBA metric `Gini index` for genes and motifs
    cutoff_gene_gini,  cutoff_motif_gini: `float`
        cutoff of SIMBA metric `Gini index` for genes and motifs
    cutoff_gene_std,  cutoff_motif_std: `float`
        cutoff of SIMBA metric `standard deviation` for genes and motifs
    cutoff_gene_entropy,  cutoff_motif_entropy: `float`
        cutoff of SIMBA metric `entropy` for genes and motifs

    Returns
    -------
    df_MR: `pandas.DataFrame`
        Dataframe of master regulators
    """
    if(sum(list(map(lambda x: x is None,
                    [list_tf_motif, list_tf_gene]))) > 0):
        return("Please specify both `list_tf_motif` and `list_tf_gene`")

    assert isinstance(list_tf_motif, list), \
        "`list_tf_motif` must be list"
    assert isinstance(list_tf_gene, list), \
        "`list_tf_gene` must be list"
    assert len(list_tf_motif) == len(list_tf_gene), \
        "`list_tf_motif` and `list_tf_gene` must have the same length"
    assert len(list_tf_motif) == len(set(list_tf_motif)), \
        "Duplicates are found in `list_tf_motif`"

    genes = adata_all[adata_all.obs[anno_filter] == filter_gene].\
        obs_names.tolist().copy()
    # Master Regulator
    df_MR = pd.DataFrame(list(zip(list_tf_motif, list_tf_gene)),
                         columns=['motif', 'gene'])

    if metrics_motif is not None:
        print('Adding motif metrics ...')
        assert isinstance(metrics_motif, pd.DataFrame), \
            "`metrics_motif` must be pd.DataFrame"
        df_metrics_motif = metrics_motif.loc[list_tf_motif, ].copy()
        df_metrics_motif.columns = df_metrics_motif.columns + '_motif'
        df_MR = df_MR.merge(df_metrics_motif,
                            how='left',
                            left_on='motif',
                            right_index=True)

    if metrics_gene is not None:
        print('Adding gene metrics ...')
        assert isinstance(metrics_gene, pd.DataFrame), \
            "`metrics_gene` must be pd.DataFrame"
        df_metrics_gene = metrics_gene.loc[list_tf_gene, ].copy()
        df_metrics_gene.index = list_tf_motif  # avoid duplicate genes
        df_metrics_gene.columns = df_metrics_gene.columns + '_gene'
        df_MR = df_MR.merge(df_metrics_gene,
                            how='left',
                            left_on='motif',
                            right_index=True)
    print('Computing distances between TF motifs and genes ...')
    dist_MG = distance.cdist(adata_all[df_MR['motif'], ].X,
                             adata_all[genes, ].X,
                             metric=metric)
    dist_MG = pd.DataFrame(dist_MG,
                           index=df_MR['motif'].tolist(),
                           columns=genes)
    df_MR.insert(2, 'rank', -1)
    df_MR.insert(3, 'dist', -1)
    for i in np.arange(df_MR.shape[0]):
        x_motif = df_MR['motif'].iloc[i]
        x_gene = df_MR['gene'].iloc[i]
        df_MR.loc[i, 'rank'] = dist_MG.loc[x_motif, ].rank()[x_gene]
        df_MR.loc[i, 'dist'] = dist_MG.loc[x_motif, x_gene]

    if metrics_gene is not None:
        print('filtering master regulators based on gene metrics:')
        if cutoff_gene_entropy is not None:
            print('entropy')
            df_MR = df_MR[df_MR['entropy_gene'] > cutoff_gene_entropy]
        if cutoff_gene_gini is not None:
            print('Gini index')
            df_MR = df_MR[df_MR['gini_gene'] > cutoff_gene_gini]
        if cutoff_gene_max is not None:
            print('max')
            df_MR = df_MR[df_MR['max_gene'] > cutoff_gene_max]
        if cutoff_gene_std is not None:
            print('standard deviation')
            df_MR = df_MR[df_MR['std_gene'] > cutoff_gene_std]
    if metrics_motif is not None:
        print('filtering master regulators based on motif metrics:')
        if cutoff_motif_entropy is not None:
            print('entropy')
            df_MR = df_MR[df_MR['entropy_motif'] > cutoff_motif_entropy]
        if cutoff_motif_gini is not None:
            print('Gini index')
            df_MR = df_MR[df_MR['gini_motif'] > cutoff_motif_gini]
        if cutoff_motif_max is not None:
            print('max')
            df_MR = df_MR[df_MR['max_motif'] > cutoff_motif_max]
        if cutoff_motif_std is not None:
            print('standard deviation')
            df_MR = df_MR[df_MR['std_motif'] > cutoff_motif_std]
    df_MR = df_MR.sort_values(by='rank', ignore_index=True)
    return df_MR


def find_target_genes(adata_all,
                      adata_PM,
                      list_tf_motif=None,
                      list_tf_gene=None,
                      adata_CP=None,
                      metric='euclidean',
                      anno_filter='entity_anno',
                      filter_peak='peak',
                      filter_gene='gene',
                      n_genes=200,
                      cutoff_gene=None,
                      cutoff_peak=1000,
                      use_precomputed=True,
                      ):
    """For a given TF, infer its target genes

    Parameters
    ----------
    adata_all : `AnnData`
        Anndata object storing SIMBA embedding of all entities.
    adata_PM : `AnnData`
        Peaks-by-motifs anndata object.
    list_tf_motif : `list`
        A list of TF motifs. They should match TF motifs in `list_tf_gene`.
    list_tf_gene : `list`
        A list TF genes. They should match TF motifs in `list_tf_motif`.
    adata_CP : `AnnData`, optional (default: None)
        When ``use_precomputed`` is True, it can be set None
    metric : `str`, optional (default: "euclidean")
        The distance metric to use. It can be ‘braycurtis’, ‘canberra’,
        ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
        ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
        ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    anno_filter : `str`, optional (default: None)
        The annotation of filter to use.
        It should be one of ``adata.obs_keys()``
    filter_gene : `str`, optional (default: None)
        The filter for gene.
        It should be in ``adata.obs[anno_filter]``
    filter_peak : `str`, optional (default: None)
        The filter for peak.
        It should be in ``adata.obs[anno_filter]``
    n_genes : `int`, optional (default: 200)
        The number of neighbor genes to consider initially
        around TF gene or TF motif
    cutoff_gene : `float`, optional (default: None)
        Cutoff of "average_rank"
    cutoff_peak : `int`, optional (default: 1000)
        Cutoff for peaks-associated ranks, including
        "rank_peak_to_gene" and "rank_peak_to_TFmotif".
    use_precomputed : `bool`, optional (default: True)
        Distances calculated between genes, peaks, and motifs
        (stored in `adata.uns['tf_targets']`) will be imported

    Returns
    -------
    dict_tf_targets : `dict`
        Target genes for each TF.

    updates `adata` with the following fields.

    tf_targets: `dict`, (`adata.uns['tf_targets']`)
        Distances calculated between genes, peaks, and motifs
    """
    if(sum(list(map(lambda x: x is None,
                    [list_tf_motif, list_tf_gene]))) > 0):
        return("Please specify both `list_tf_motif` and `list_tf_gene`")

    assert isinstance(list_tf_motif, list), \
        "`list_tf_motif` must be list"
    assert isinstance(list_tf_gene, list), \
        "`list_tf_gene` must be list"
    assert len(list_tf_motif) == len(list_tf_gene), \
        "`list_tf_motif` and `list_tf_gene` must have the same length"

    def isin(a, b):
        return np.array([item in b for item in a])

    print('Preprocessing ...')
    if use_precomputed and 'tf_targets' in adata_all.uns_keys():
        print('importing precomputed variables ...')
        genes = adata_all.uns['tf_targets']['genes']
        peaks = adata_all.uns['tf_targets']['peaks']
        peaks_in_genes = adata_all.uns['tf_targets']['peaks_in_genes']
        dist_PG = adata_all.uns['tf_targets']['dist_PG']
        overlap_PG = adata_all.uns['tf_targets']['overlap']
    else:
        assert (adata_CP is not None), \
            '`adata_CP` needs to be specified '\
            'when no precomputed variable is stored'
        if 'gene_scores' not in adata_CP.uns_keys():
            print('Please run "si.tl.gene_scores(adata_CP)" first.')
        else:
            overlap_PG = adata_CP.uns['gene_scores']['overlap'].copy()
            overlap_PG['peak'] = \
                overlap_PG[['chr_p', 'start_p', 'end_p']].apply(
                    lambda row: '_'.join(row.values.astype(str)), axis=1)
            tuples = list(zip(overlap_PG['symbol_g'], overlap_PG['peak']))
            multi_indices = pd.MultiIndex.from_tuples(
                tuples, names=["gene", "peak"])
            overlap_PG.index = multi_indices

        genes = adata_all[adata_all.obs[anno_filter] == filter_gene].\
            obs_names.tolist().copy()
        peaks = adata_all[adata_all.obs[anno_filter] == filter_peak].\
            obs_names.tolist().copy()
        peaks_in_genes = list(set(overlap_PG['peak']))

        print(f'#genes: {len(genes)}')
        print(f'#peaks: {len(peaks)}')
        print(f'#genes-associated peaks: {len(peaks_in_genes)}')
        print('computing distances between genes '
              'and genes-associated peaks ...')
        dist_PG = distance.cdist(
            adata_all[peaks_in_genes, ].X,
            adata_all[genes, ].X,
            metric=metric,
            )
        dist_PG = pd.DataFrame(dist_PG, index=peaks_in_genes, columns=[genes])
        print("Saving variables into `.uns['tf_targets']` ...")
        adata_all.uns['tf_targets'] = dict()
        adata_all.uns['tf_targets']['overlap'] = overlap_PG
        adata_all.uns['tf_targets']['dist_PG'] = dist_PG
        adata_all.uns['tf_targets']['peaks_in_genes'] = peaks_in_genes
        adata_all.uns['tf_targets']['genes'] = genes
        adata_all.uns['tf_targets']['peaks'] = peaks
        adata_all.uns['tf_targets']['peaks_in_genes'] = peaks_in_genes

    dict_tf_targets = dict()
    for tf_motif, tf_gene in zip(list_tf_motif, list_tf_gene):

        print(f'searching for target genes of {tf_motif}')
        motif_peaks = adata_PM.obs_names[adata_PM[:, tf_motif].X.nonzero()[0]]
        motif_genes = list(
            set(overlap_PG[isin(overlap_PG['peak'], motif_peaks)]['symbol_g'])
            .intersection(genes))

        # rank of the distances from genes to tf_motif
        dist_GM_motif = distance.cdist(adata_all[genes, ].X,
                                       adata_all[tf_motif, ].X,
                                       metric=metric)
        dist_GM_motif = pd.DataFrame(dist_GM_motif,
                                     index=genes,
                                     columns=[tf_motif])
        rank_GM_motif = dist_GM_motif.rank(axis=0)

        # rank of the distances from genes to tf_gene
        dist_GG_motif = distance.cdist(adata_all[genes, ].X,
                                       adata_all[tf_gene, ].X,
                                       metric=metric)
        dist_GG_motif = pd.DataFrame(dist_GG_motif,
                                     index=genes,
                                     columns=[tf_gene])
        rank_GG_motif = dist_GG_motif.rank(axis=0)

        # rank of the distances from peaks to tf_motif
        dist_PM_motif = distance.cdist(
            adata_all[peaks_in_genes, ].X,
            adata_all[tf_motif, ].X,
            metric=metric)
        dist_PM_motif = pd.DataFrame(dist_PM_motif,
                                     index=peaks_in_genes,
                                     columns=[tf_motif])
        rank_PM_motif = dist_PM_motif.rank(axis=0)

        # rank of the distances from peaks to candidate genes
        cand_genes = \
            dist_GG_motif[tf_gene].nsmallest(n_genes).index.tolist()\
            + dist_GM_motif[tf_motif].nsmallest(n_genes).index.tolist()
        print(f'#candinate genes is {len(cand_genes)}')
        print('removing duplicate genes ...')
        print('removing genes that do not contain TF motif ...')
        cand_genes = list(set(cand_genes).intersection(set(motif_genes)))
        print(f'#candinate genes is {len(cand_genes)}')
        dist_PG_motif = distance.cdist(
            adata_all[peaks_in_genes, ].X,
            adata_all[cand_genes, ].X,
            metric=metric
            )
        dist_PG_motif = pd.DataFrame(dist_PG_motif,
                                     index=peaks_in_genes,
                                     columns=cand_genes)
        rank_PG_motif = dist_PG_motif.rank(axis=0)

        df_tf_targets = pd.DataFrame(index=cand_genes)
        df_tf_targets['average_rank'] = -1
        df_tf_targets['has_motif'] = 'no'
        df_tf_targets['rank_gene_to_TFmotif'] = -1
        df_tf_targets['rank_gene_to_TFgene'] = -1
        df_tf_targets['rank_peak_to_TFmotif'] = -1
        df_tf_targets['rank_peak2_to_TFmotif'] = -1
        df_tf_targets['rank_peak_to_gene'] = -1
        df_tf_targets['rank_peak2_to_gene'] = -1
        for i, g in enumerate(cand_genes):
            g_peaks = list(set(overlap_PG.loc[[g]]['peak']))
            g_motif_peaks = list(set(g_peaks).intersection(motif_peaks))
            if len(g_motif_peaks) > 0:
                df_tf_targets.loc[g, 'has_motif'] = 'yes'
                df_tf_targets.loc[g, 'rank_gene_to_TFmotif'] = \
                    rank_GM_motif[tf_motif][g]
                df_tf_targets.loc[g, 'rank_gene_to_TFgene'] = \
                    rank_GG_motif[tf_gene][g]
                df_tf_targets.loc[g, 'rank_peak_to_TFmotif'] = \
                    rank_PM_motif.loc[g_peaks, tf_motif].min()
                df_tf_targets.loc[g, 'rank_peak2_to_TFmotif'] = \
                    rank_PM_motif.loc[g_motif_peaks, tf_motif].min()
                df_tf_targets.loc[g, 'rank_peak_to_gene'] = \
                    rank_PG_motif.loc[g_peaks, g].min()
                df_tf_targets.loc[g, 'rank_peak2_to_gene'] = \
                    rank_PG_motif.loc[g_peaks, g].min()
            if i % int(len(cand_genes)/5) == 0:
                print(f'completed: {i/len(cand_genes):.1%}')
        df_tf_targets['average_rank'] = \
            df_tf_targets[['rank_gene_to_TFmotif',
                           'rank_gene_to_TFgene']].mean(axis=1)
        if cutoff_peak is not None:
            print('Pruning candidate genes based on nearby peaks ...')
            df_tf_targets = df_tf_targets[
                (df_tf_targets[[
                    'rank_peak_to_TFmotif',
                    'rank_peak_to_gene']]
                    < cutoff_peak).sum(axis=1) > 0]
        if cutoff_gene is not None:
            print('Pruning candidate genes based on average rank ...')
            df_tf_targets = df_tf_targets[
                df_tf_targets['average_rank'] < cutoff_gene]
        dict_tf_targets[tf_motif] = \
            df_tf_targets.sort_values(by='average_rank').copy()
    return dict_tf_targets
