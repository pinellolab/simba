"""reading and writing"""

import os
import json
import h5py
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

from ._settings import settings


def read_embedding(path_emb=None,
                   path_entity=None,
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

    dict_adata = dict()
    for x in os.listdir(path_emb):
        if x.startswith('embeddings'):
            entity_type = x.split('_')[1]
            if (len(prefix) == 0) or (entity_type is prefix):
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
                adata.obs.index = names_entity
                dict_adata[entity_type] = adata
    return dict_adata
