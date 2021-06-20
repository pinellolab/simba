.. automodule:: simba

API
===

Import simba as::

   import simba as si

Configuration for SIMBA
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   settings.set_figure_params
   settings.set_pbg_params
   settings.set_workdir


Reading
~~~~~~~

.. autosummary::
   :toctree: _autosummary

   read_h5ad
   read_embedding
   load_pbg_config
   load_graph_stats

Preprocessing
~~~~~~~~~~~~~

Preprocessing functions

.. autosummary::
   :toctree: _autosummary

   pp.log_transform
   pp.normalize
   pp.binarize
   pp.cal_qc
   pp.cal_qc_rna
   pp.cal_qc_atac
   pp.filter_samples
   pp.filter_cells_rna
   pp.filter_cells_atac
   pp.filter_features
   pp.filter_genes
   pp.filter_peaks
   pp.pca
   pp.select_pcs
   pp.select_pcs_features
   pp.select_variable_genes   

Tools
~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.discretize
   tl.umap
   tl.gene_scores
   tl.infer_edges
   tl.trim_edges
   tl.gen_graph
   tl.pbg_train
   tl.softmax
   tl.embed
   tl.compare_entities
   tl.query
   tl.find_master_regulators
   tl.find_target_genes


Plotting
~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   pl.pca_variance_ratio
   pl.pcs_features
   pl.variable_genes
   pl.violin
   pl.hist
   pl.umap
   pl.discretize
   pl.node_similarity
   pl.svd_nodes
   pl.pbg_metrics
   pl.entity_metrics
   pl.entity_barcode
   pl.query