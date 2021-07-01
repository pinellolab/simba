"""Configuration for SIMBA"""

import os
import seaborn as sns
import matplotlib as mpl


class SimbaConfig:
    """configuration class for SIMBA"""

    def __init__(self,
                 workdir='./result_simba',
                 save_fig=False,
                 n_jobs=1):
        self.workdir = workdir
        self.save_fig = save_fig
        self.n_jobs = n_jobs
        self.set_pbg_params()
        self.graph_stats = dict()

    def set_figure_params(self,
                          context='notebook',
                          style='white',
                          palette='deep',
                          font='sans-serif',
                          font_scale=1.1,
                          color_codes=True,
                          dpi=80,
                          dpi_save=150,
                          fig_size=[5.4, 4.8],
                          rc=None):
        """ Set global parameters for figures. Modified from sns.set()

        Parameters
        ----------
        context : string or dict
            Plotting context parameters, see `seaborn.plotting_context`
        style: `string`,optional (default: 'white')
            Axes style parameters, see `seaborn.axes_style`
        palette : string or sequence
            Color palette, see `seaborn.color_palette`
        font_scale: `float`, optional (default: 1.3)
            Separate scaling factor to independently
            scale the size of the font elements.
        color_codes : `bool`, optional (default: True)
            If ``True`` and ``palette`` is a seaborn palette,
            remap the shorthand color codes (e.g. "b", "g", "r", etc.)
            to the colors from this palette.
        dpi: `int`,optional (default: 80)
            Resolution of rendered figures.
        dpi_save: `int`,optional (default: 150)
            Resolution of saved figures.
        rc: `dict`,optional (default: None)
            rc settings properties.
            Parameter mappings to override the values in the preset style.
            Please see "`matplotlibrc file
            <https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file>`__"
        """
        sns.set(context=context,
                style=style,
                palette=palette,
                font=font,
                font_scale=font_scale,
                color_codes=color_codes,
                rc={'figure.dpi': dpi,
                    'savefig.dpi': dpi_save,
                    'figure.figsize': fig_size,
                    'image.cmap': 'viridis',
                    'lines.markersize': 6,
                    'legend.columnspacing': 0.1,
                    'legend.borderaxespad': 0.1,
                    'legend.handletextpad': 0.1,
                    'pdf.fonttype': 42,
                    })
        if rc is not None:
            assert isinstance(rc, dict), "rc must be dict"
            for key, value in rc.items():
                if key in mpl.rcParams.keys():
                    mpl.rcParams[key] = value
                else:
                    raise Exception("unrecognized property '%s'" % key)

    def set_workdir(self, workdir=None):
        """Set working directory.

        Parameters
        ----------
        workdir: `str`, optional (default: None)
            Working directory.

        Returns
        -------
        """
        if(workdir is None):
            workdir = self.workdir
            print("Using default working directory.")
        if(not os.path.exists(workdir)):
            os.makedirs(workdir)
        self.workdir = workdir
        self.set_pbg_params()
        print('Saving results in: %s' % workdir)

    def set_pbg_params(self, config=None):
        """Set PBG parameters

        Parameters
        ----------
        config : `dict`, optional (default: None)
            PBG training configuration parameters.
            By default it resets parameters to the default setting.

        Returns
        -------
        """
        if config is None:
            config = dict(
                # I/O data
                entity_path="",
                edge_paths=["", ],
                checkpoint_path="",

                # Graph structure
                entities={},
                relations=[],
                dynamic_relations=False,

                # Scoring model
                dimension=50,
                global_emb=False,
                comparator='dot',

                # Training
                num_epochs=10,
                workers=4,
                num_batch_negs=50,
                num_uniform_negs=50,
                loss_fn='softmax',
                lr=0.1,

                early_stopping=False,
                regularization_coef=0.0,
                wd=0.0,
                wd_interval=50,

                # Evaluation during training
                eval_fraction=0.05,
                eval_num_batch_negs=50,
                eval_num_uniform_negs=50,

                checkpoint_preservation_interval=None,
            )
        assert isinstance(config, dict), "`config` must be dict"
        self.pbg_params = config


settings = SimbaConfig()
