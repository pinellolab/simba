"""post-training plotting functions"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from adjustText import adjust_text
from pandas.api.types import (
    is_numeric_dtype
)
from scipy.stats import rankdata

from ._utils import (
    get_colors,
    generate_palette
)
from .._settings import settings
from ._plot import _scatterplot2d


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

    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

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
        ax_i.set_xlabel('epoch')
        ax_i.set_ylabel(metric)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def entity_metrics(adata_cmp,
                   x,
                   y,
                   text_size=10,
                   show_texts=True,
                   show_cutoff=False,
                   cutoff_x=0,
                   cutoff_y=0,
                   n_texts=10,
                   texts=None,
                   fig_size=None,
                   save_fig=None,
                   fig_path=None,
                   fig_name='entity_metrics.pdf',
                   pad=1.08,
                   w_pad=None,
                   h_pad=None,
                   **kwargs):
    """Plot entity metrics

    Parameters
    ----------
    adata_cmp: `AnnData`
        Anndata object from `compare_entities`
    x, y: `str`
        Variables that specify positions on the x and y axes.
        Possible values:
        - max (The average maximum dot product of top-rank reference entities,
        based on normalized dot product)
        - std (standard deviation of reference entities,
        based on dot product)
        - gini (Gini coefficients of reference entities,
        based on softmax probability)
        - entropy (The entropy of reference entities,
        based on softmax probability)
    texts: `list` optional(default: None)
        Entity names to plot

    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    assert (x in ['max', 'std', 'gini', 'entropy']), \
        "x must be one of ['max','std','gini','entropy']"
    assert (y in ['max', 'std', 'gini', 'entropy']), \
        "y must be one of ['max','std','gini','entropy']"

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(adata_cmp.var[x],
               adata_cmp.var[y],
               **kwargs)
    if show_texts:
        if texts is not None:
            plt_texts = [plt.text(adata_cmp.var[x][t],
                                  adata_cmp.var[y][t],
                                  t,
                         fontdict={'family': 'serif',
                                   'color': 'black',
                                   'weight': 'normal',
                                   'size': text_size})
                         for t in texts]
        else:
            if x == 'entropy':
                ranks_x = rankdata(-adata_cmp.var[x])
            else:
                ranks_x = rankdata(adata_cmp.var[x])
            if y == 'entropy':
                ranks_y = rankdata(-adata_cmp.var[y])
            else:
                ranks_y = rankdata(adata_cmp.var[y])
            ids = np.argsort(ranks_x + ranks_y)[::-1][:n_texts]
            plt_texts = [plt.text(adata_cmp.var[x][i],
                                  adata_cmp.var[y][i],
                                  adata_cmp.var_names[i],
                         fontdict={'family': 'serif',
                                   'color': 'black',
                                   'weight': 'normal',
                                   'size': text_size})
                         for i in ids]
        adjust_text(plt_texts,
                    arrowprops=dict(arrowstyle='-', color='black'))
    if show_cutoff:
        ax.axvline(x=cutoff_x, linestyle='--')
        ax.axhline(y=cutoff_y, linestyle='--')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.locator_params(axis='x', tight=True)
    ax.locator_params(axis='y', tight=True)
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def entity_barcode(adata_cmp,
                   entities,
                   anno_ref=None,
                   layer='softmax',
                   palette=None, 
                   alpha=0.8,
                   linewidths=1,
                   show_cutoff=False,
                   cutoff=0.5,
                   fig_size=(6, 2),
                   fig_ncol=1,
                   save_fig=None,
                   fig_path=None,
                   fig_name='barcode.pdf',
                   pad=1.08,
                   w_pad=None,
                   h_pad=None,
                   min_rank=None,
                   max_rank=None,
                   **kwargs
                   ):
    """Plot query entity barcode

    Parameters
    ----------
    adata_cmp: `AnnData`
        Anndata object from `compare_entities`
    entities: `list`
        Entity names to plot.
    anno_ref:  `str`
        Annotation used for reference entity
    palette: `dict`
        Color palette used for `anno_ref`
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    assert isinstance(entities, list), "`entities` must be list"

    if layer is None:
        X = adata_cmp[:, entities].X.copy()
    else:
        X = adata_cmp[:, entities].layers[layer].copy()
    df_scores = pd.DataFrame(
        data=X,
        index=adata_cmp.obs_names,
        columns=entities)

    if min_rank is None:
        min_rank = 0
    if max_rank is None:
        max_rank = df_scores.shape[0]

    n_plots = len(entities)
    fig_nrow = int(np.ceil(n_plots/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                              fig_size[1]*fig_nrow))

    for i, x in enumerate(entities):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        scores_x_sorted = df_scores[x].sort_values(ascending=False)
        lines = []
        for xx, yy in zip(np.arange(len(scores_x_sorted))[min_rank:max_rank],
                          scores_x_sorted[min_rank:max_rank]):
            lines.append([(xx, 0), (xx, yy)])
        if anno_ref is None:
            colors = get_colors(np.array([""]*len(scores_x_sorted)))
        else:
            ids_ref = scores_x_sorted.index
            if palette is None:
                colors = get_colors(adata_cmp[ids_ref, :].obs[anno_ref])
            else:
                colors = [palette[adata_cmp.obs.loc[xx, anno_ref]]
                          for xx in scores_x_sorted.index]
        stemlines = LineCollection(
            lines,
            colors=colors,
            alpha=alpha,
            linewidths=linewidths,
            **kwargs)
        ax_i.add_collection(stemlines)
        ax_i.autoscale()
        ax_i.set_title(x)
        ax_i.set_ylabel(layer)
        ax_i.locator_params(axis='y', tight=True)
        if show_cutoff:
            ax_i.axhline(y=cutoff,
                         color='#CC6F47',
                         linestyle='--')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def query(adata,
          comp1=1,
          comp2=2,
          color=None,
          dict_palette=None,
          size=8,
          drawing_order='random',
          dict_drawing_order=None,
          show_texts=False,
          texts=None,
          text_size=10,
          n_texts=8,
          fig_size=None,
          fig_ncol=3,
          fig_legend_ncol=1,
          fig_legend_order=None,
          alpha=0.8,
          alpha_bg=0.5,
          pad=1.08,
          w_pad=None,
          h_pad=None,
          save_fig=None,
          fig_path=None,
          fig_name='plot_query.pdf',
          vmin=None,
          vmax=None,
          **kwargs):
    """Plot query output
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    if dict_palette is None:
        dict_palette = dict()

    query_output = adata.uns['query']['output']
    nn = query_output.index.tolist()  # nearest neighbors
    query_params = adata.uns['query']['params']
    obsm = query_params['obsm']
    layer = query_params['layer']
    pin = query_params['pin']
    use_radius = query_params['use_radius']
    r = query_params['r']
    if(obsm is not None):
        X = adata.obsm[obsm].copy()
        X_nn = adata[nn, :].obsm[obsm].copy()
    elif(layer is not None):
        X = adata.layers[layer].copy()
        X_nn = adata[nn, :].layers[layer].copy()
    else:
        X = adata.X.copy()
        X_nn = adata[nn, :].X.copy()
    df_plot = pd.DataFrame(index=adata.obs.index,
                           data=X[:, [comp1-1, comp2-1]],
                           columns=[f'Dim {comp1}', f'Dim {comp2}'])
    df_plot_nn = pd.DataFrame(index=adata[nn, :].obs.index,
                              data=X_nn[:, [comp1-1, comp2-1]],
                              columns=[f'Dim {comp1}', f'Dim {comp2}'])
    if show_texts:
        if texts is None:
            texts = nn[:n_texts]
    if color is None:
        list_ax = _scatterplot2d(df_plot,
                                 x=f'Dim {comp1}',
                                 y=f'Dim {comp2}',
                                 drawing_order=drawing_order,
                                 size=size,
                                 fig_size=fig_size,
                                 alpha=alpha_bg,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=False,
                                 copy=True,
                                 **kwargs)
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if(ann in adata.obs_keys()):
                df_plot[ann] = adata.obs[ann]
                if(not is_numeric_dtype(df_plot[ann])):
                    if 'color' not in adata.uns_keys():
                        adata.uns['color'] = dict()

                    if ann not in dict_palette.keys():
                        if (ann+'_color' in adata.uns['color'].keys()) \
                            and \
                            (all(np.isin(np.unique(df_plot[ann]),
                                         list(adata.uns['color']
                                         [ann+'_color'].keys())))):
                            dict_palette[ann] = \
                                adata.uns['color'][ann+'_color']
                        else:
                            dict_palette[ann] = \
                                generate_palette(adata.obs[ann])
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()
                    else:
                        if ann+'_color' not in adata.uns['color'].keys():
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()

            elif(ann in adata.var_names):
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(f"could not find {ann} in `adata.obs.columns`"
                                 " and `adata.var_names`")
        list_ax = _scatterplot2d(df_plot,
                                 x=f'Dim {comp1}',
                                 y=f'Dim {comp2}',
                                 list_hue=color,
                                 hue_palette=dict_palette,
                                 drawing_order=drawing_order,
                                 dict_drawing_order=dict_drawing_order,
                                 size=size,
                                 fig_size=fig_size,
                                 fig_ncol=fig_ncol,
                                 fig_legend_ncol=fig_legend_ncol,
                                 fig_legend_order=fig_legend_order,
                                 vmin=vmin,
                                 vmax=vmax,
                                 alpha=alpha_bg,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=False,
                                 copy=True,
                                 **kwargs)
    for ax in list_ax:
        ax.scatter(df_plot_nn[f'Dim {comp1}'],
                   df_plot_nn[f'Dim {comp2}'],
                   s=size,
                   color='#AE6C68',
                   alpha=alpha,
                   lw=0)
        ax.scatter(pin[:, 0],
                   pin[:, 1],
                   s=20*size,
                   marker='+',
                   color='#B33831')
        if use_radius:
            circle = plt.Circle((pin[:, 0],
                                 pin[:, 1]),
                                radius=r,
                                color='#B33831',
                                fill=False)
            ax.add_artist(circle)
        if show_texts:
            plt_texts = [ax.text(df_plot_nn[f'Dim {comp1}'][t],
                                 df_plot_nn[f'Dim {comp2}'][t],
                                 t,
                                 fontdict={'family': 'serif',
                                           'color': 'black',
                                           'weight': 'normal',
                                           'size': text_size})
                         for t in texts]
            adjust_text(plt_texts,
                        ax=ax,
                        arrowprops=dict(arrowstyle='->', color='black'))
    if save_fig:
        fig = plt.gcf()
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)
