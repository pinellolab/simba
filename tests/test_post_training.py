import simba as si
import pytest


@pytest.fixture
def dict_adata():

    return si.read_embedding(
        path_emb='tests/data/pbg_training/model/',
        path_entity='tests/data/pbg_training/input/entity/',
        path_entity_alias='tests/data/pbg_training')


def test_embeddding_rna(dict_adata):
    adata_C = dict_adata['C']
    adata_G = dict_adata['G']
    adata_all_CG = si.tl.embed(
        adata_ref=adata_C,
        list_adata_query=[adata_G])
    si.tl.umap(adata_all_CG,
               n_neighbors=15,
               n_components=2)
