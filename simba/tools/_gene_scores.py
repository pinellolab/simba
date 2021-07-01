"""Predict gene scores based on chromatin accessibility"""

import numpy as np
import pandas as pd
import anndata as ad
import io
import pybedtools
from scipy.sparse import (
    coo_matrix,
    csr_matrix
)
import pkgutil

from ._utils import _uniquify


class GeneScores:
    """A class used to represent gene scores

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self,
                 adata,
                 genome,
                 gene_anno=None,
                 tss_upstream=1e5,
                 tss_downsteam=1e5,
                 gb_upstream=5000,
                 cutoff_weight=1,
                 use_top_pcs=True,
                 use_precomputed=True,
                 use_gene_weigt=True,
                 min_w=1,
                 max_w=5):
        """
        Parameters
        ----------
        adata: `Anndata`
            Input anndata
        genome : `str`
            The genome name
        """
        self.adata = adata
        self.genome = genome
        self.gene_anno = gene_anno
        self.tss_upstream = tss_upstream
        self.tss_downsteam = tss_downsteam
        self.gb_upstream = gb_upstream
        self.cutoff_weight = cutoff_weight
        self.use_top_pcs = use_top_pcs
        self.use_precomputed = use_precomputed
        self.use_gene_weigt = use_gene_weigt
        self.min_w = min_w
        self.max_w = max_w

    def _read_gene_anno(self):
        """Read in gene annotation

        Parameters
        ----------

        Returns
        -------

        """
        assert (self.genome in ['hg19', 'hg38', 'mm9', 'mm10']),\
            "`genome` must be one of ['hg19','hg38','mm9','mm10']"

        bin_str = pkgutil.get_data('simba',
                                   f'data/gene_anno/{self.genome}_genes.bed')
        gene_anno = pd.read_csv(io.BytesIO(bin_str),
                                encoding='utf8',
                                sep='\t',
                                header=None,
                                names=['chr', 'start', 'end',
                                       'symbol', 'strand'])
        self.gene_anno = gene_anno
        return self.gene_anno

    def _extend_tss(self, pbt_gene):
        """Extend transcription start site in both directions

        Parameters
        ----------

        Returns
        -------

        """
        ext_tss = pbt_gene
        if(ext_tss['strand'] == '+'):
            ext_tss.start = max(0, ext_tss.start - self.tss_upstream)
            ext_tss.end = max(ext_tss.end, ext_tss.start + self.tss_downsteam)
        else:
            ext_tss.start = max(0, min(ext_tss.start,
                                       ext_tss.end - self.tss_downsteam))
            ext_tss.end = ext_tss.end + self.tss_upstream
        return ext_tss

    def _extend_genebody(self, pbt_gene):
        """Extend gene body upstream

        Parameters
        ----------

        Returns
        -------

        """
        ext_gb = pbt_gene
        if(ext_gb['strand'] == '+'):
            ext_gb.start = max(0, ext_gb.start - self.gb_upstream)
        else:
            ext_gb.end = ext_gb.end + self.gb_upstream
        return ext_gb

    def _weight_genes(self):
        """Weight genes

        Parameters
        ----------

        Returns
        -------

        """
        gene_anno = self.gene_anno
        gene_size = gene_anno['end'] - gene_anno['start']
        w = 1/gene_size
        w_scaled = (self.max_w-self.min_w) * (w-min(w)) / (max(w)-min(w)) \
            + self.min_w
        return w_scaled

    def cal_gene_scores(self):
        """Calculate gene scores

        Parameters
        ----------

        Returns
        -------

        """
        adata = self.adata
        if self.gene_anno is None:
            gene_ann = self._read_gene_anno()
        else:
            gene_ann = self.gene_anno

        df_gene_ann = gene_ann.copy()
        df_gene_ann.index = _uniquify(df_gene_ann['symbol'].values)
        if self.use_top_pcs:
            mask_p = adata.var['top_pcs']
        else:
            mask_p = pd.Series(True, index=adata.var_names)
        df_peaks = adata.var[mask_p][['chr', 'start', 'end']].copy()

        if('gene_scores' not in adata.uns_keys()):
            print('Gene scores are being calculated for the first time')
            print('`use_precomputed` has been ignored')
            self.use_precomputed = False

        if(self.use_precomputed):
            print('Using precomputed overlap')
            df_overlap_updated = adata.uns['gene_scores']['overlap'].copy()
        else:
            # add the fifth column
            # so that pybedtool can recognize the sixth column as the strand
            df_gene_ann_for_pbt = df_gene_ann.copy()
            df_gene_ann_for_pbt['score'] = 0
            df_gene_ann_for_pbt = df_gene_ann_for_pbt[['chr', 'start', 'end',
                                                       'symbol', 'score',
                                                       'strand']]
            df_gene_ann_for_pbt['id'] = range(df_gene_ann_for_pbt.shape[0])

            df_peaks_for_pbt = df_peaks.copy()
            df_peaks_for_pbt['id'] = range(df_peaks_for_pbt.shape[0])

            pbt_gene_ann = pybedtools.BedTool.from_dataframe(
                df_gene_ann_for_pbt
                )
            pbt_gene_ann_ext = pbt_gene_ann.each(self._extend_tss)
            pbt_gene_gb_ext = pbt_gene_ann.each(self._extend_genebody)

            pbt_peaks = pybedtools.BedTool.from_dataframe(df_peaks_for_pbt)

            # peaks overlapping with extended TSS
            pbt_overlap = pbt_peaks.intersect(pbt_gene_ann_ext,
                                              wa=True,
                                              wb=True)
            df_overlap = pbt_overlap.to_dataframe(
                names=[x+'_p' for x in df_peaks_for_pbt.columns]
                + [x+'_g' for x in df_gene_ann_for_pbt.columns])
            # peaks overlapping with gene body
            pbt_overlap2 = pbt_peaks.intersect(pbt_gene_gb_ext,
                                               wa=True,
                                               wb=True)
            df_overlap2 = pbt_overlap2.to_dataframe(
                names=[x+'_p' for x in df_peaks_for_pbt.columns]
                + [x+'_g' for x in df_gene_ann_for_pbt.columns])

            # add distance and weight for each overlap
            df_overlap_updated = df_overlap.copy()
            df_overlap_updated['dist'] = 0

            for i, x in enumerate(df_overlap['symbol_g'].unique()):
                # peaks within the extended TSS
                df_overlap_x = \
                    df_overlap[df_overlap['symbol_g'] == x].copy()
                # peaks within the gene body
                df_overlap2_x = \
                    df_overlap2[df_overlap2['symbol_g'] == x].copy()
                # peaks that are not intersecting with the promoter
                # and gene body of gene x
                id_overlap = df_overlap_x.index[
                    ~np.isin(df_overlap_x['id_p'], df_overlap2_x['id_p'])]
                mask_x = (df_gene_ann['symbol'] == x)
                range_x = df_gene_ann[mask_x][['start', 'end']].values\
                    .flatten()
                if(df_overlap_x['strand_g'].iloc[0] == '+'):
                    df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                        [abs(df_overlap_x.loc[id_overlap, 'start_p']
                             - (range_x[1])),
                         abs(df_overlap_x.loc[id_overlap, 'end_p']
                             - max(0, range_x[0]-self.gb_upstream))],
                        axis=1, sort=False).min(axis=1)
                else:
                    df_overlap_updated.loc[id_overlap, 'dist'] = pd.concat(
                        [abs(df_overlap_x.loc[id_overlap, 'start_p']
                             - (range_x[1]+self.gb_upstream)),
                         abs(df_overlap_x.loc[id_overlap, 'end_p']
                             - (range_x[0]))],
                        axis=1, sort=False).min(axis=1)

                n_batch = int(df_gene_ann_for_pbt.shape[0]/5)
                if(i % n_batch == 0):
                    print(f'Processing: {i/df_gene_ann_for_pbt.shape[0]:.1%}')
            df_overlap_updated['dist'] = df_overlap_updated['dist']\
                .astype(float)

            adata.uns['gene_scores'] = dict()
            adata.uns['gene_scores']['overlap'] = df_overlap_updated.copy()

        df_overlap_updated['weight'] = np.exp(
            -(df_overlap_updated['dist'].values/self.gb_upstream))
        mask_w = (df_overlap_updated['weight'] < self.cutoff_weight)
        df_overlap_updated.loc[mask_w, 'weight'] = 0
        # construct genes-by-peaks matrix
        mat_GP = csr_matrix(coo_matrix((df_overlap_updated['weight'],
                                       (df_overlap_updated['id_g'],
                                        df_overlap_updated['id_p'])),
                                       shape=(df_gene_ann.shape[0],
                                              df_peaks.shape[0])))
        # adata_GP = ad.AnnData(X=csr_matrix(mat_GP),
        #                       obs=df_gene_ann,
        #                       var=df_peaks)
        # adata_GP.layers['weight'] = adata_GP.X.copy()
        if self.use_gene_weigt:
            gene_weights = self._weight_genes()
            gene_scores = adata[:, mask_p].X * \
                (mat_GP.T.multiply(gene_weights))
        else:
            gene_scores = adata[:, mask_p].X * mat_GP.T
        adata_CG_atac = ad.AnnData(gene_scores,
                                   obs=adata.obs.copy(),
                                   var=df_gene_ann.copy())
        return adata_CG_atac


def gene_scores(adata,
                genome,
                gene_anno=None,
                tss_upstream=1e5,
                tss_downsteam=1e5,
                gb_upstream=5000,
                cutoff_weight=1,
                use_top_pcs=True,
                use_precomputed=True,
                use_gene_weigt=True,
                min_w=1,
                max_w=5):
    """Calculate gene scores

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genome : `str`
        Reference genome. Choose from {'hg19', 'hg38', 'mm9', 'mm10'}
    gene_anno : `pandas.DataFrame`, optional (default: None)
        Dataframe of gene annotation.
        If None, built-in gene annotation will be used depending on `genome`;
        If provided, custom gene annotation will be used instead.
    tss_upstream : `int`, optional (default: 1e5)
        The number of base pairs upstream of TSS
    tss_downsteam : `int`, optional (default: 1e5)
        The number of base pairs downstream of TSS
    gb_upstream : `int`, optional (default: 5000)
        The number of base pairs upstream by which gene body is extended.
        Peaks within the extended gene body are given the weight of 1.
    cutoff_weight : `float`, optional (default: 1)
        Weight cutoff for peaks
    use_top_pcs : `bool`, optional (default: True)
        If True, only peaks associated with top PCs will be used
    use_precomputed : `bool`, optional (default: True)
        If True, overlap bewteen peaks and genes
        (stored in `adata.uns['gene_scores']['overlap']`) will be imported
    use_gene_weigt : `bool`, optional (default: True)
        If True, for each gene, the number of peaks assigned to it
        will be rescaled based on gene size
    min_w : `int`, optional (default: 1)
        The minimum weight for each gene.
        Only valid if `use_gene_weigt` is True
    max_w : `int`, optional (default: 5)
        The maximum weight for each gene.
        Only valid if `use_gene_weigt` is True

    Returns
    -------
    adata_new: AnnData
        Annotated data matrix.
        Stores #cells x #genes gene score matrix

    updates `adata` with the following fields.
    overlap: `pandas.DataFrame`, (`adata.uns['gene_scores']['overlap']`)
        Dataframe of overlap between peaks and genes
    """
    GS = GeneScores(adata,
                    genome,
                    gene_anno=gene_anno,
                    tss_upstream=tss_upstream,
                    tss_downsteam=tss_downsteam,
                    gb_upstream=gb_upstream,
                    cutoff_weight=cutoff_weight,
                    use_top_pcs=use_top_pcs,
                    use_precomputed=use_precomputed,
                    use_gene_weigt=use_gene_weigt,
                    min_w=min_w,
                    max_w=max_w)
    adata_CG_atac = GS.cal_gene_scores()
    return adata_CG_atac
