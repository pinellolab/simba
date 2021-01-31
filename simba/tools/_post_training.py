"""Functions and classes for the analysis after PBG training"""

import numpy as np
import anndata as ad


def softmax(adata_ref,
            adata_query,
            T=0.3):
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
    Returns
    -------
    updates `adata_query` with the following field.
    softmax: `array_like` (`.layers['softmax']`)
        Store #observations × #dimensions softmax transformed data matrix.
    """

    scores_ref_query = np.matmul(adata_ref.X, adata_query.X.T)
    scores_softmax = np.exp(scores_ref_query/T) / \
        (np.exp(scores_ref_query/T).sum(axis=0))[None, :]
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
                 T=0.3,
                 list_T=None,
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
        X_all = adata_ref.X.copy()
        # obs_all = pd.DataFrame(
        #     data=['ref']*adata_ref.shape[0],
        #     index=adata_ref.obs.index,
        #     columns=['id_dataset'])
        obs_all = adata_ref.obs.copy()
        obs_all['id_dataset'] = ['ref']*adata_ref.shape[0]
        for i, adata_query in enumerate(list_adata_query):
            if list_T is not None:
                T = list_T[i]
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
                        T=T
                    )
            else:
                print(f'Performing softmax transformation '
                      f'for query data {i};')
                softmax(
                    adata_ref,
                    adata_query,
                    T=T
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
          T=0.3,
          list_T=None,
          use_precomputed=True):
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
                    use_precomputed=use_precomputed)
    adata_all = SE.embed()
    return adata_all


# def compare_entities(adata_ref,
#                      adata_query):
#     """Compare the embeddings of two entities

#     Parameters
#     ----------
#         adata_ref: `AnnData`
#             Reference entity anndata.
#         adata_query: `list`
#             Query entity anndata.

#     Returns
#     -------
#     adata_cmp: `AnnData`
#         Store reference entity as observations and query entity as variables
#     """
#     X_ref = adata_ref.X
#     X_query = adata_query.X
#     X_cmp = np.matmul(X_ref, X_query.T)
#     adata_cmp = ad.AnnData(X=X_cmp,
#                            obs=adata_ref.obs,
#                            var=adata_query.obs)
#     return adata_cmp
#     # adata_cmp.layers['norm'] = df_scores_MC.subtract(np.log(np.exp(X_cmp).mean(axis=0)),axis=1)
