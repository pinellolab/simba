"""The core functionality"""

from ._general import (
    discretize,
)
from ._umap import umap
from ._gene_scores import gene_scores
from ._integration import (
    node_dist,
    infer_edges
)
from ._pbg import (
    gen_graph,
    pbg_train
)
from ._post_training import (
    softmax,
    embed,
    compare_entities
)
