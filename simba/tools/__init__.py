"""The core functionality"""

from ._general import (
    discretize,
)
from ._umap import umap
from ._gene_scores import gene_scores
from ._integration import (
    infer_edges,
    trim_edges
)
from ._pbg import (
    gen_graph,
    pbg_train
)
from ._post_training import (
    softmax,
    embed,
    compare_entities,
    query,
    find_master_regulators,
    find_target_genes,
)
