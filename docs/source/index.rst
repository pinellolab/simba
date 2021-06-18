.. SIMBA documentation master file, created by
   sphinx-quickstart on Mon Feb  1 17:17:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**SIMBA**: **SI**\ ngle-cell e\ **MB**\ edding **A**\ long with features
========================================================================

SIMBA is a method to embed cells along with their defining features such as gene expression, transcription factor binding sequences and chromatin accessibility peaks into the same latent space. The joint embedding of cells and features allows SIMBA to perform various types of single cell tasks, including but not limited to single-modal analysis (e.g. scRNA-seq and scATAC-seq analysis), multimodal analysis, batch correction, and multi-omic integration.


.. image:: _static/img/Figure1.png
   :align: center
   :width: 600
   :alt: SIMBA overview


.. toctree::
   :maxdepth: 2
   :caption: Get started

   Installation
   API


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   scRNA-seq analysis
   scATAC-seq analysis
   Multimodal analysis
   Batch correction
   Multi-omics integration
