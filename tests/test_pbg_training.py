import simba as si
import pytest


@pytest.fixture
def adata_CG():
    return si.read_h5ad(
        "tests/data/preprocessed/rna_preprocessed.h5ad")


@pytest.fixture
def adata_CP():
    return si.read_h5ad(
        "tests/data/preprocessed/atac_preprocessed.h5ad")


def test_pbg_training_rna(adata_CG, tmp_path):
    si.settings.set_workdir(tmp_path / "simba_rna")
    si.tl.gen_graph(list_CG=[adata_CG],
                    copy=False,
                    dirname='graph0')
    si.tl.pbg_train(auto_wd=True,
                    output='model')


def test_pbg_training_atac(adata_CP, tmp_path):
    si.settings.set_workdir(tmp_path / "simba_atac")
    si.tl.gen_graph(list_CP=[adata_CP],
                    copy=False,
                    dirname='graph0')
    si.tl.pbg_train(auto_wd=True,
                    output='model')
