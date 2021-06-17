import simba as si


def test_rna():
    adata_CG = si.read_h5ad("./data/10xpbmc_rna_subset.h5ad")
    si.pp.filter_genes(adata_CG, min_n_cells=3)
    si.pp.cal_qc_rna(adata_CG)
    si.pp.normalize(adata_CG, method='lib_size')
    si.pp.log_transform(adata_CG)
    si.pp.select_variable_genes(adata_CG, n_top_genes=2000)
    si.tl.discretize(adata_CG, n_bins=5)


def test_atac():
    adata_CP = si.read_h5ad("./data/10xpbmc_atac_subset.h5ad")
    si.pp.filter_peaks(adata_CP, min_n_cells=3)
    si.pp.cal_qc_atac(adata_CP)
    si.pp.pca(adata_CP, n_components=30)
    si.pp.select_pcs(adata_CP, n_pcs=10)
    si.pp.select_pcs_features(adata_CP)
