"""reading and writing"""

import os
import pandas as pd
import json
from anndata import (
    AnnData,
    read_h5ad,
    read_csv,
    read_excel,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
    read_umi_tools,
    read_zarr,
)
from pathlib import Path

from ._settings import settings


def read_embedding(path_emb=None,
                   path_entity=None,
                   convert_alias=True,
                   path_entity_alias=None,
                   prefix=None,
                   num_epochs=None,
                   **kwargs):
    """Read in entity embeddings from pbg training

    Parameters
    ----------
    path_emb: `str`, optional (default: None)
        Path to directory for pbg embedding model
    path_entity: `str`, optional (default: None)
        Path to entity name file
    prefix: `list`, optional (default: None)
        A list of entity type prefixes to include.
        By default, it reads in the embeddings of all entities.
    convert_alias: `bool`, optional (default: True)
        If True, it will convert entity aliases to the original indices
    path_entity: `str`, optional (default: None)
        Path to entity alias file
    num_epochs: `int`, optional (default: None)
        The embedding result associated with num_epochs to read in

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects of shape
        (#entities x #dimensions)
    """
    pbg_params = settings.pbg_params
    if path_emb is None:
        path_emb = pbg_params['checkpoint_path']
    if path_entity is None:
        path_entity = pbg_params['entity_path']
    if num_epochs is None:
        num_epochs = pbg_params["num_epochs"]
    if prefix is None:
        prefix = []
    assert isinstance(prefix, list), \
        "`prefix` must be list"
    if convert_alias:
        if path_entity_alias is None:
            path_entity_alias = Path(path_emb).parent.as_posix()
        df_entity_alias = pd.read_csv(
            os.path.join(path_entity_alias, 'entity_alias.txt'),
            header=0,
            index_col=0,
            sep='\t')
        df_entity_alias['id'] = df_entity_alias.index
        df_entity_alias.index = df_entity_alias['alias'].values

    dict_adata = dict()
    for x in os.listdir(path_emb):
        if x.startswith('embeddings'):
            entity_type = x.split('_')[1]
            if (len(prefix) == 0) or (entity_type in prefix):
                adata = \
                    read_hdf(os.path.join(path_emb,
                                          f'embeddings_{entity_type}_0.'
                                          f'v{num_epochs}.h5'),
                             key="embeddings")
                with open(
                    os.path.join(path_entity,
                                 f'entity_names_{entity_type}_0.json'), "rt")\
                        as tf:
                    names_entity = json.load(tf)
                if convert_alias:
                    names_entity = \
                        df_entity_alias.loc[names_entity, 'id'].tolist()
                adata.obs.index = names_entity
                dict_adata[entity_type] = adata
    return dict_adata
