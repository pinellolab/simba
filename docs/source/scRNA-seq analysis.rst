scRNA-seq analysis
==================


.. code:: ipython3

    %load_ext autoreload
    %autoreload 2

.. code:: ipython3

    import sys
    sys.path[:3]




.. parsed-literal::

    ['/data/pinello/PROJECTS/2019_08_Embedding/SIMBA_RESULTS/rna_10xpmbc_3k',
     '/data/pinello/SHARED_SOFTWARE/anaconda_latest/envs/hc_simba/lib/python37.zip',
     '/data/pinello/SHARED_SOFTWARE/anaconda_latest/envs/hc_simba/lib/python3.7']



.. code:: ipython3

    sys.path.insert(1,'/data/pinello/PROJECTS/2019_08_Embedding/Github/simba/')
    sys.path[:3]




.. parsed-literal::

    ['/data/pinello/PROJECTS/2019_08_Embedding/SIMBA_RESULTS/rna_10xpmbc_3k',
     '/data/pinello/PROJECTS/2019_08_Embedding/Github/simba/',
     '/data/pinello/SHARED_SOFTWARE/anaconda_latest/envs/hc_simba/lib/python37.zip']



.. code:: ipython3

    import numpy as np
    import pandas as pd
    import simba as si
    import os
    
    import matplotlib as mpl
    
    si.__path__




.. parsed-literal::

    ['/data/pinello/PROJECTS/2019_08_Embedding/Github/simba/simba']



.. code:: ipython3

    workdir = 'result_10xpbmc_rna_all_genes'
    si.settings.set_workdir(workdir)


.. parsed-literal::

    Saving results in: result_10xpmbc_rna_all_genes


.. code:: ipython3

    si.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5,5],
                                  rc={'image.cmap': 'viridis'})

.. code:: ipython3

    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('retina')


preprocessing
-------------

.. code:: ipython3

    adata_CG = si.read_h5ad("./input/data_processed/rna/rna_seq.h5ad")

.. code:: ipython3

    adata_CG




.. parsed-literal::

    AnnData object with n_obs × n_vars = 2700 × 32738
        obs: 'celltype'
        var: 'gene_ids'



.. code:: ipython3

    # si.pp.filter_cells_rna(adata,min_n_genes=100)
    si.pp.filter_genes(adata_CG,min_n_cells=3)


.. parsed-literal::

    Before filtering: 
    2700 cells, 32738 genes
    Filter genes based on min_n_cells
    After filtering out low-expressed genes: 
    2700 cells, 13714 genes


.. parsed-literal::

    /data/pinello/SHARED_SOFTWARE/anaconda_latest/envs/hc_simba/lib/python3.7/site-packages/pandas/core/arrays/categorical.py:2487: FutureWarning: The `inplace` parameter in pandas.Categorical.remove_unused_categories is deprecated and will be removed in a future version.
      res = method(*args, **kwargs)


.. code:: ipython3

    si.pp.cal_qc_rna(adata_CG)

.. code:: ipython3

    si.pl.violin(adata_CG,list_obs=['n_counts','n_genes','pct_mt'])



.. image:: output_13_0.png
   :width: 740px
   :height: 223px


.. code:: ipython3

    si.pp.normalize(adata_CG,method='lib_size')

.. code:: ipython3

    si.pp.log_transform(adata_CG)

.. code:: ipython3

    # si.pp.select_variable_genes(adata_CG)

.. code:: ipython3

    # si.pl.variable_genes(adata_CG,show_texts=True)


discretize RNA expression
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    si.tl.discretize(adata_CG,n_bins=5)

.. code:: ipython3

    si.pl.discretize(adata_CG,kde=False)


.. parsed-literal::

    [0.48992336 1.5519998  2.1158602  2.934613   3.9790487  7.4695992 ]



.. image:: output_21_1.png
   :width: 385px
   :height: 624px


.. code:: ipython3

    # si.pl.discretize(adata_CG,kde=True,save_fig=True)


Generate Graph
--------------

.. code:: ipython3

    si.tl.gen_graph(list_CG=[adata_CG],
                    copy=False,
                    use_highly_variable=False,
                    dirname='graph0')


.. parsed-literal::

    relation0: source: C, destination: G
    #edges: 599381
    relation1: source: C, destination: G
    #edges: 1009575
    relation2: source: C, destination: G
    #edges: 386586
    relation3: source: C, destination: G
    #edges: 191955
    relation4: source: C, destination: G
    #edges: 95479
    Total number of edges: 2282976
    Writing graph file "pbg_graph.txt" to "result_10xpmbc_rna_all_genes/pbg/graph0" ...
    Finished.



PBG training
------------

.. code:: ipython3

    si.settings.pbg_params




.. parsed-literal::

    {'entity_path': 'result_10xpmbc_rna_all_genes/pbg/graph0/input/entity',
     'edge_paths': ['result_10xpmbc_rna_all_genes/pbg/graph0/input/edge'],
     'checkpoint_path': '',
     'entities': {'C': {'num_partitions': 1}, 'G': {'num_partitions': 1}},
     'relations': [{'name': 'r0',
       'lhs': 'C',
       'rhs': 'G',
       'operator': 'none',
       'weight': 1.0},
      {'name': 'r1', 'lhs': 'C', 'rhs': 'G', 'operator': 'none', 'weight': 2.0},
      {'name': 'r2', 'lhs': 'C', 'rhs': 'G', 'operator': 'none', 'weight': 3.0},
      {'name': 'r3', 'lhs': 'C', 'rhs': 'G', 'operator': 'none', 'weight': 4.0},
      {'name': 'r4', 'lhs': 'C', 'rhs': 'G', 'operator': 'none', 'weight': 5.0}],
     'dynamic_relations': False,
     'dimension': 50,
     'global_emb': False,
     'comparator': 'dot',
     'num_epochs': 10,
     'workers': 12,
     'num_batch_negs': 50,
     'num_uniform_negs': 50,
     'loss_fn': 'softmax',
     'lr': 0.1,
     'early_stopping': False,
     'regularization_coef': 0.0,
     'wd': 0.0,
     'wd_interval': 50,
     'eval_fraction': 0.05,
     'eval_num_batch_negs': 50,
     'eval_num_uniform_negs': 50,
     'checkpoint_preservation_interval': None}



.. code:: ipython3

    dict_config = si.settings.pbg_params.copy()
    ## start training
    # dict_config['wd'] = 0.03
    dict_config['wd_interval'] = 10
    si.tl.pbg_train(pbg_params = dict_config, auto_wd=True, output='model')


.. parsed-literal::

    Auto-estimated weight decay is 0.015521
    Converting input data ...
    [2021-04-12 08:52:43.016684] Using the 5 relation types given in the config
    [2021-04-12 08:52:43.017061] Searching for the entities in the edge files...
    [2021-04-12 08:52:46.054140] Entity type C:
    [2021-04-12 08:52:46.054735] - Found 2700 entities
    [2021-04-12 08:52:46.055019] - Removing the ones with fewer than 1 occurrences...
    [2021-04-12 08:52:46.055694] - Left with 2700 entities
    [2021-04-12 08:52:46.055964] - Shuffling them...
    [2021-04-12 08:52:46.057864] Entity type G:
    [2021-04-12 08:52:46.058136] - Found 13714 entities
    [2021-04-12 08:52:46.058401] - Removing the ones with fewer than 1 occurrences...
    [2021-04-12 08:52:46.060205] - Left with 13714 entities
    [2021-04-12 08:52:46.060490] - Shuffling them...
    [2021-04-12 08:52:46.068813] Preparing counts and dictionaries for entities and relation types:
    [2021-04-12 08:52:46.077696] - Writing count of entity type C and partition 0
    [2021-04-12 08:52:46.090715] - Writing count of entity type G and partition 0
    [2021-04-12 08:52:46.113092] Preparing edge path result_10xpmbc_rna_all_genes/pbg/graph0/input/edge, out of the edges found in result_10xpmbc_rna_all_genes/pbg/graph0/pbg_graph.txt
    using fast version
    [2021-04-12 08:52:46.118034] Taking the fast train!
    [2021-04-12 08:52:46.619746] - Processed 100000 edges so far...
    [2021-04-12 08:52:47.108569] - Processed 200000 edges so far...
    [2021-04-12 08:52:47.607510] - Processed 300000 edges so far...
    [2021-04-12 08:52:48.098836] - Processed 400000 edges so far...
    [2021-04-12 08:52:48.587013] - Processed 500000 edges so far...
    [2021-04-12 08:52:49.083782] - Processed 600000 edges so far...
    [2021-04-12 08:52:49.571705] - Processed 700000 edges so far...
    [2021-04-12 08:52:50.064365] - Processed 800000 edges so far...
    [2021-04-12 08:52:50.553970] - Processed 900000 edges so far...
    [2021-04-12 08:52:51.044757] - Processed 1000000 edges so far...
    [2021-04-12 08:52:51.540628] - Processed 1100000 edges so far...
    [2021-04-12 08:52:52.027092] - Processed 1200000 edges so far...
    [2021-04-12 08:52:52.520503] - Processed 1300000 edges so far...
    [2021-04-12 08:52:53.008818] - Processed 1400000 edges so far...
    [2021-04-12 08:52:53.503935] - Processed 1500000 edges so far...
    [2021-04-12 08:52:53.994155] - Processed 1600000 edges so far...
    [2021-04-12 08:52:54.481127] - Processed 1700000 edges so far...
    [2021-04-12 08:52:54.974594] - Processed 1800000 edges so far...
    [2021-04-12 08:52:55.465706] - Processed 1900000 edges so far...
    [2021-04-12 08:52:55.959469] - Processed 2000000 edges so far...
    [2021-04-12 08:52:56.448359] - Processed 2100000 edges so far...
    [2021-04-12 08:52:56.941507] - Processed 2200000 edges so far...
    [2021-04-12 08:52:59.784640] - Processed 2282976 edges in total
    Starting training ...



.. code:: ipython3

    si.settings.pbg_params = dict_config.copy()

.. code:: ipython3

    si.pl.pbg_metrics(fig_ncol=1)



.. image:: output_32_0.png
   :width: 405px
   :height: 704px


.. code:: ipython3

    si.pl.pbg_metrics(fig_ncol=1,save_fig=True,fig_name='graph0_model.pdf')


Post-training Analysis
----------------------

.. code:: ipython3

    palette_celltype={'B':'#1f77b4',
                      'CD4 T':'#ff7f0e', 
                      'CD8 T':'#279e68',
                      'Dendritic':"#aa40fc",
                      'CD14 Monocytes':'#d62728',
                      'FCGR3A Monocytes':'#8c564b',
                      'Megakaryocytes':'#e377c2',
                      'NK':'#b5bd61'}

.. code:: ipython3

    dict_adata = si.read_embedding()

.. code:: ipython3

    dict_adata




.. parsed-literal::

    {'G': AnnData object with n_obs × n_vars = 13714 × 50,
     'C': AnnData object with n_obs × n_vars = 2700 × 50}



.. code:: ipython3

    adata_C = dict_adata['C']  # embeddings for cells
    adata_G = dict_adata['G']  # embeddings for genes

.. code:: ipython3

    adata_C




.. parsed-literal::

    AnnData object with n_obs × n_vars = 2700 × 50



.. code:: ipython3

    adata_G




.. parsed-literal::

    AnnData object with n_obs × n_vars = 13714 × 50




visualize embeddings of cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ## Add annotation of celltypes (optional)
    adata_C.obs['celltype'] = adata_CG[adata_C.obs_names,:].obs['celltype'].copy()


.. parsed-literal::

    /data/pinello/SHARED_SOFTWARE/anaconda_latest/envs/hc_simba/lib/python3.7/site-packages/pandas/core/arrays/categorical.py:2487: FutureWarning: The `inplace` parameter in pandas.Categorical.remove_unused_categories is deprecated and will be removed in a future version.
      res = method(*args, **kwargs)


.. code:: ipython3

    adata_C




.. parsed-literal::

    AnnData object with n_obs × n_vars = 2700 × 50
        obs: 'celltype'



.. code:: ipython3

    si.tl.umap(adata_C,n_neighbors=15,n_components=2)

.. code:: ipython3

    adata_C




.. parsed-literal::

    AnnData object with n_obs × n_vars = 2700 × 50
        obs: 'celltype'
        obsm: 'X_umap'



.. code:: ipython3

    si.pl.umap(adata_C,color=['celltype'],dict_palette={'celltype': palette_celltype},fig_size=(6,4),
               drawing_order='random')



.. image:: output_48_0.png
   :width: 488px
   :height: 304px


.. code:: ipython3

    si.pl.umap(adata_C,color=['celltype'],dict_palette={'celltype': palette_celltype},fig_size=(6,4),
               drawing_order='random',
               save_fig=True,
               fig_name='umap_graph0_model.pdf')


visualize embeddings of cells and genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SIMBA embed genes into the same UMAP space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])


.. parsed-literal::

    Performing softmax transformation for query data 0;


.. code:: ipython3

    adata_all.obs




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>celltype</th>
          <th>id_dataset</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>GACTCCTGTTGGTG-1</th>
          <td>CD14 Monocytes</td>
          <td>ref</td>
        </tr>
        <tr>
          <th>TCTAACACCAGTTG-1</th>
          <td>FCGR3A Monocytes</td>
          <td>ref</td>
        </tr>
        <tr>
          <th>GAAACCTGTGCTAG-1</th>
          <td>CD4 T</td>
          <td>ref</td>
        </tr>
        <tr>
          <th>CATTACACCAACTG-1</th>
          <td>FCGR3A Monocytes</td>
          <td>ref</td>
        </tr>
        <tr>
          <th>ACTCAGGATTCGTT-1</th>
          <td>CD14 Monocytes</td>
          <td>ref</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>OAZ1</th>
          <td>NaN</td>
          <td>query_0</td>
        </tr>
        <tr>
          <th>TMEM131</th>
          <td>NaN</td>
          <td>query_0</td>
        </tr>
        <tr>
          <th>FAS</th>
          <td>NaN</td>
          <td>query_0</td>
        </tr>
        <tr>
          <th>ASAP1</th>
          <td>NaN</td>
          <td>query_0</td>
        </tr>
        <tr>
          <th>CCDC120</th>
          <td>NaN</td>
          <td>query_0</td>
        </tr>
      </tbody>
    </table>
    <p>16414 rows × 2 columns</p>
    </div>



.. code:: ipython3

    ## add annotations of cells and genes
    adata_all.obs['entity_anno'] = ""
    adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'celltype']
    adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'

.. code:: ipython3

    adata_all.obs




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>celltype</th>
          <th>id_dataset</th>
          <th>entity_anno</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>GACTCCTGTTGGTG-1</th>
          <td>CD14 Monocytes</td>
          <td>ref</td>
          <td>CD14 Monocytes</td>
        </tr>
        <tr>
          <th>TCTAACACCAGTTG-1</th>
          <td>FCGR3A Monocytes</td>
          <td>ref</td>
          <td>FCGR3A Monocytes</td>
        </tr>
        <tr>
          <th>GAAACCTGTGCTAG-1</th>
          <td>CD4 T</td>
          <td>ref</td>
          <td>CD4 T</td>
        </tr>
        <tr>
          <th>CATTACACCAACTG-1</th>
          <td>FCGR3A Monocytes</td>
          <td>ref</td>
          <td>FCGR3A Monocytes</td>
        </tr>
        <tr>
          <th>ACTCAGGATTCGTT-1</th>
          <td>CD14 Monocytes</td>
          <td>ref</td>
          <td>CD14 Monocytes</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>OAZ1</th>
          <td>NaN</td>
          <td>query_0</td>
          <td>gene</td>
        </tr>
        <tr>
          <th>TMEM131</th>
          <td>NaN</td>
          <td>query_0</td>
          <td>gene</td>
        </tr>
        <tr>
          <th>FAS</th>
          <td>NaN</td>
          <td>query_0</td>
          <td>gene</td>
        </tr>
        <tr>
          <th>ASAP1</th>
          <td>NaN</td>
          <td>query_0</td>
          <td>gene</td>
        </tr>
        <tr>
          <th>CCDC120</th>
          <td>NaN</td>
          <td>query_0</td>
          <td>gene</td>
        </tr>
      </tbody>
    </table>
    <p>16414 rows × 3 columns</p>
    </div>



.. code:: ipython3

    si.tl.umap(adata_all,n_neighbors=15,n_components=2)

.. code:: ipython3

    palette_entity_anno = palette_celltype.copy()
    palette_entity_anno['gene'] = "#30598a"

.. code:: ipython3

    si.pl.umap(adata_all,color=['id_dataset','entity_anno'],dict_palette={'entity_anno': palette_entity_anno},
               drawing_order='original',
               fig_size=(6,5))



.. image:: output_59_0.png
   :width: 992px
   :height: 384px


.. code:: ipython3

    marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                    'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                    'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

.. code:: ipython3

    si.pl.umap(adata_all[::-1,],color=['entity_anno'],dict_palette={'entity_anno': palette_entity_anno},
               drawing_order='original',
               texts=marker_genes,
               show_texts=True,
               fig_size=(8,6))



.. image:: output_61_0.png
   :width: 656px
   :height: 464px



.. code:: ipython3

    adata_CG.write(os.path.join(workdir, 'adata_CG.h5ad'))
    adata_all.write(os.path.join(workdir, 'adata_all.h5ad'))
    adata_C.write(os.path.join(workdir, 'adata_C.h5ad'))
    adata_G.write(os.path.join(workdir, 'adata_G.h5ad'))


.. parsed-literal::

    ... storing 'celltype' as categorical
    ... storing 'id_dataset' as categorical
    ... storing 'entity_anno' as categorical



