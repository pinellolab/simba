"""post-training plotting functions"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from ._utils import (
    generate_palette
)


from .._settings import settings


def pbg_metrics(metrics=['mrr'],
                path_emb=None,
                fig_size=(5, 3),
                fig_ncol=1,
                save_fig=None,
                fig_path=None,
                fig_name='pbg_metrics.pdf',
                pad=1.08,
                w_pad=None,
                h_pad=None,
                **kwargs):
    """Plot PBG training metrics

    Parameters
    ----------
    metrics: `list`, optional (default: ['mrr])
        Evalulation metrics for PBG training.
        Possible metrics:
        - 'pos_rank' : the average of the ranks of all positives
          (lower is better, best is 1).
        - 'mrr' : the average of the reciprocal of the ranks of all positives
          (higher is better, best is 1).
        - 'r1' : the fraction of positives that rank better than
           all their negatives, i.e., have a rank of 1
           (higher is better, best is 1).
        - 'r10' : the fraction of positives that rank in the top 10
           among their negatives
           (higher is better, best is 1).
        - 'r50' : the fraction of positives that rank in the top 50
           among their negatives
           (higher is better, best is 1).
        - 'auc' : Area Under the Curve (AUC)

    Returns
    -------
    None
    """

    assert isinstance(metrics, list), "`metrics` must be list"
    for x in metrics:
        if(x not in ['pos_rank', 'mrr', 'r1',
                     'r10', 'r50', 'auc']):
            raise ValueError(f'unrecognized metric {x}')
    pbg_params = settings.pbg_params
    if path_emb is None:
        path_emb = pbg_params['checkpoint_path']
    training_loss = []
    eval_stats_before = dict()
    with open(os.path.join(path_emb, 'training_stats.json'), 'r') as f:
        for line in f:
            line_json = json.loads(line)
            if('stats' in line_json.keys()):
                training_loss.append(line_json['stats']['metrics']['loss'])
                line_stats_before = line_json['eval_stats_before']['metrics']
                for x in line_stats_before.keys():
                    if(x not in eval_stats_before.keys()):
                        eval_stats_before[x] = [line_stats_before[x]]
                    else:
                        eval_stats_before[x].append(line_stats_before[x])
    df_metrics = pd.DataFrame(index=range(pbg_params['num_epochs']))
    df_metrics['epoch'] = range(pbg_params['num_epochs'])
    df_metrics['training_loss'] = training_loss
    df_metrics['validation_loss'] = eval_stats_before['loss']
    for x in metrics:
        df_metrics[x] = eval_stats_before[x]

    fig_nrow = int(np.ceil((df_metrics.shape[1]-1)/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                              fig_size[1]*fig_nrow))
    dict_palette = generate_palette(df_metrics.columns[1:].values)
    for i, metric in enumerate(df_metrics.columns[1:]):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        ax_i.scatter(df_metrics['epoch'],
                     df_metrics[metric],
                     c=dict_palette[metric],
                     **kwargs)
        ax_i.set_title(metric)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)
