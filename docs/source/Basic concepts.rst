================
Basic concepts
================


Graph construction
~~~~~~~~~~~~~~~~~~
SIMBA encodes entities of different types, including genes, open chromatin regions (peaks or bins), and DNA sequences (transcription factor motifs or k-mers), into a single large graph based on the relation between them. In this graph, nodes represent different entities and edges indicate the relation between entities. 

* In scRNA-seq analysis, each node represents either a cell or a gene. If a gene is expressed in a cell, then an edge is added between this gene and cell. The gene expression level is encoded into the weight of this edge.

* In scATAC-seq analysis, each node represents either a cell or a region (peak/bin). If a region is open in a cell, then an edge is added between this region and cell. Optionally, if DNA sequences (TF motifs or k-mers) are also used, each node represents a cell, or a region, or a DNA sequence. In addition to the relation between a cell and a region, if a DNA sequence is found within the open region, then an edge is added between this DNA sequence and open region.

* In multimodal analysis, each node can be any of these entities, including a cell, a gene, a open region , a DNA sequence, etc. Edges are added similarly as in scRNA-seq analysis and scATAC-seq analysis.

* In batch correction analysis, in addition to the experimentally measured edges as described above, batch correction is further enhanced with the computationally inferred edges between cell nodes across datasets using a truncated randomized singular value decomposition (SVD)-based procedure

* In multiomics integration analysis (scRNA-seq and scATAC-seq), SIMBA first builds one graph for scRNA-seq data and one graph for scATAC-seq data independently as described above. To connect these two graphs, SIMBA calculates gene scores by summarizing accessible regions from scATAC-seq data and then infer edges between cells of different omics based on their shared gene expression modules through a similar procedure as in batch correction.

PBG training
~~~~~~~~~~~~
Following the construction of a multi-relational graph between biological entities, we adapt graph embedding techniques from the knowledge graph and recommendation systems literature to construct unsupervised representations for these entities.

We use the PyTorch-BigGraph(PBG) framework, which provides efficient computation of multi-relation graph embeddings over multiple entity types and can scale to graphs with millions or billions of entities. 

In SIMBA, several key modifications have been made based on PBG, including:

* Type-constrainted negative sampling

* Negative samples are produced in two ways: 

  * by corrupting the edge with a source or destination sampled uniformly from the nodes with the correct types for this      relation;
  
  * by corrupting the edge with a source or destination node sampled with probability proportional to its degree.

* Introducing a weight decay procedure to solve overfitting problem.

The resulting graph embeddings have two desirable properties that we will take advantage of:

#. First-order similarity: for two entity types  with a relation between them, edges with high likelihood should have higher dot product.
#. Second-order similarity: within a single entity type, entities that have ‘similar contexts’, i.e., a similar distribution of edge probabilities, should have similar embeddings. 

Evaluation during training
~~~~~~~~~~~~~~~~~~~~~~~~~~
During the PBG training procedure, a small percent of edges is held out (by default, the evaluation fraction is set to 5%) to monitor overfitting and evaluate the final model. 

Five metrics are computed on the reserved set of edges, including mean reciprocal rank (MRR, the average of the reciprocal of the ranks of all positives), R1 (the fraction of positives that rank better than all their negatives, i.e., have a rank of 1), R10 (the fraction of positives that rank in the top 10 among their negatives), R50 (the fraction of positives that rank in the top 50 among their negatives), and AUC (Area Under the Curve). 

By default, we show MRR along with training loss and validation loss while other metric are also available in SIMBA package (Supplementary Fig. 1a).  The learning curves for validation loss and these metrics can be used to determine when training has completed. The relative values of training and validation loss along with these evaluation metrics can be used to identify issues with training (underfitting vs overfitting) and tune the hyperparameters weight decay, embedding dimension, and number of training epochs appropriately. However, for most datasets we find that the default parameters do not need tuning. 

Softmax transformation
~~~~~~~~~~~~~~~~~~~~~~
PyTorch-BigGraph training provides initial embeddings of all entities (nodes).  However, entities of different types (e.g., cells vs peaks, cells of different batches or modalities) have different edge distributions and thus may lie on different manifolds of the latent space. To make the embeddings of entities of different types comparable, we transform the embeddings of features with Softmax function by utilizing the first-order similarity between cells (reference) and features (query). In the case of batch correction or multi-omics integration, the SoftMax transformation is also performed based on the first-order similarity between cells of different batches or modalities. 
