================
Basic concepts
================


Graph construction
~~~~~~~~~~~~~~~~~~
SIMBA encodes entities of different types, including but not limited to cells, genes, open chromatin regions (peaks or bins), and DNA sequences (transcription factor motifs or k-mers), into a single large graph based on the relation between them. In this graph, nodes represent different entities and edges indicate the relation between entities. 

* In scRNA-seq analysis, each node represents either a cell or a gene. If a gene is expressed in a cell, then an edge is added between this gene and cell. The gene expression level is encoded into the weight of this edge.

* In scATAC-seq analysis, each node represents either a cell or a region (peak/bin). If a region is open in a cell, then an edge is added between this region and cell. Optionally, if DNA sequences (TF motifs or k-mers) are also used, each node represents a cell, or a region, or a DNA sequence. In addition to the relation between a cell and a region, if a DNA sequence is found within the open region, then an edge is added between this DNA sequence and open region.

* In multimodal analysis, each node can be any of these entities, including a cell, a gene, a open region , a DNA sequence, etc. Edges are added similarly as in scRNA-seq analysis and scATAC-seq analysis.

* In batch correction analysis, in addition to the edges constructed as described above, an edge will be also added between two cells of different batches if they are mutual nearest neighbors in the common latent space.

* In multiomics integration analysis, the edges are added similarly as in batch correction analysis. 

PBG training
~~~~~~~~~~~~