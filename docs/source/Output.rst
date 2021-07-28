Output
======

SIMBA result structure will look like this:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    result_simba
    ├── figures
    └── pbg
        └── graph0
            ├── pbg_graph.txt
            ├── graph_stats.json
            ├── entity_alias.txt
            └── input          
                ├── edge
                └── entity
            └── model0          
                ├── config.json
                ├── training_stats.json
                ├── checkpoint_version.txt  
                ├── embeddings.h5  
                └── model.h5
            └── model1          
                ├── config.json
                ├── training_stats.json
                ├── checkpoint_version.txt  
                ├── embeddings.h5  
                └── model.h5
            └── model2          
                ├── config.json
                ├── training_stats.json
                ├── checkpoint_version.txt  
                ├── embeddings.h5  
                └── model.h5
        └── graph1
            ├── pbg_graph.txt
            ├── graph_stats.json
            ├── entity_alias.txt
            └── input          
                ├── edge
                └── entity
            └── model          
                ├── config.json
                ├── training_stats.json
                ├── checkpoint_version.txt  
                ├── embeddings.h5  
                └── model.h5

By default, all figures will be saved under ``result_simba/figures``

PBG training will be saved under the folder ``result_simba/pbg``. Within this folder, each constructed graph is saved into a separate folder (by default ``graph0``) under ``pbg``. For each graph:

- ``pbg_graph.txt`` stores its edges on which PBG training is performed;
- ``graph_stats.json`` stores the statistics related to this graph;
- ``entity_alias.txt`` keeps the mapping between the original entity IDs and their aliases. 
- ``input`` stores the extracted nodes (entities) and edges from ``pbg_graph.txt``, which are prepared for PBG training.
- ``model`` stores the training result of one parameter configuration. (by default ``model``)