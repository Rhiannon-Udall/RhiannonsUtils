import copy
import logging
import os

import corner
import matplotlib.lines as mpllines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.gw.conversion import generate_mass_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_rift_to_bilby = {
    "m1": "mass_1",
    "m2": "mass_2",
    "a1x": "spin_1x",
    "a1y": "spin_1y",
    "a1z": "spin_1z",
    "a2x": "spin_2x",
    "a2y": "spin_2y",
    "a2z": "spin_2z",
    "mc": "chirp_mass",
    "eta": "symmetric_mass_ratio",
    "ra": "ra",
    "dec": "dec",
    "phiorb": "phase",
    "incl": "iota",
    "psi": "psi",
    "mtotal": "total_mass",
    "q": "mass_ratio",
    "dist": "luminosity_distance",
}

_bilby_to_tex = {
    "mass_1": "$m_1$",
    "mass_2": "$m_2$",
    "spin_1x": r"$S_{1,x}$",
    "spin_1y": r"$S_{1,y}$",
    "spin_1z": r"$S_{1,z}$",
    "spin_2x": r"$S_{2,x}$",
    "spin_2y": r"$S_{2,y}$",
    "spin_2z": r"$S_{2,z}$",
    "chirp_mass": r"$\mathcal{M}$",
    "symmetric_mass_ratio": r"$\eta$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "phase": r"$\phi$",
    "iota": r"$\iota$",
    "psi": r"$\psi$",
    "total_mass": r"$M$",
    "mass_ratio": r"$q$",
    "luminosity_distance": r"$D_L$",
}


def get_posterior_dataframe(posterior_file):
    with open(posterior_file, "r") as f:
        header = f.readline()
    header = header.replace("#", "").replace("  ", " ").strip().split(" ")
    bilby_header = [
        (lambda x: _rift_to_bilby[x] if (x in _rift_to_bilby.keys()) else x)(x)
        for x in header
    ]
    return pd.read_csv(
        posterior_file, sep=" ", names=bilby_header, header=None, skiprows=1
    )


def make_results_dict(posterior_file_paths_list, posterior_names, relative_rundir=""):
    posterior_dataframes = dict()
    for ii, posterior_file in enumerate(posterior_file_paths_list):
        path_to_file = os.path.expanduser(os.path.join(relative_rundir, posterior_file))
        posterior_dataframes[posterior_names[ii]] = get_posterior_dataframe(
            path_to_file
        )
    return posterior_dataframes


def get_ile_points(composite_path):
    data = np.loadtxt(composite_path, delimiter=" ")
    dataframe = pd.DataFrame()
    dataframe["mass_1"] = data[:, 1]
    dataframe["mass_2"] = data[:, 2]
    dataframe["spin_1x"] = data[:, 3]
    dataframe["spin_1y"] = data[:, 4]
    dataframe["spin_1z"] = data[:, 5]
    dataframe["spin_2x"] = data[:, 6]
    dataframe["spin_2y"] = data[:, 7]
    dataframe["spin_2z"] = data[:, 8]
    dataframe["lnL"] = data[:, 9]
    dataframe = generate_mass_parameters(dataframe)
    return dataframe


def plot_multiple_RIFT(
    results_dict,
    parameters_to_plot,
    ile_points=None,
    lnL_span_cut=None,
    title=None,
    **kwargs,
):
    # Setup kwargs for plotting - use Bilby defaults
    defaults_kwargs = dict(
        bins=50,
        smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16),
        truth_color="tab:orange",
        quantiles=[],
        # levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        levels=(0.90,),
        plot_density=False,
        plot_datapoints=False,
        no_fill_contours=True,
        # fill_contours_with_hatches=True,
        max_n_ticks=3,
        hist_kwargs=dict(density=True),
        hatch_base="/",
    )
    defaults_kwargs.update(kwargs)

    lines = []
    labels = []

    # Loop over posteriors to be plotted
    for ii, (posterior_name, posterior_samples) in enumerate(results_dict.items()):
        # Make a local copy of the kwargs so they can be modified
        kwargs_to_use = copy.deepcopy(defaults_kwargs)
        # Get a fresh dataframe
        plotting_df = pd.DataFrame()
        # Populate with params of interest
        # Note: corner.py wants you to use arviz
        # This, however, is very annoying, so I am not doing so
        # Sorry for the warnings
        for param in parameters_to_plot:
            plotting_df[_bilby_to_tex[param]] = results_dict[posterior_name][param]
        # Get default color based on how many posteriors have already been plotted

        c = f"C{ii}"
        # If color is provided use it, and propagate to 1d histograms
        # Else use our default
        kwargs_to_use["color"] = kwargs_to_use.get("color", c)
        kwargs_to_use["hist_kwargs"]["color"] = kwargs_to_use["hist_kwargs"].get(
            "color", kwargs_to_use["color"]
        )

        # If it's the first posterior, make the new corner
        if ii == 0:
            corner_fig = corner.corner(
                plotting_df, label=posterior_name, **kwargs_to_use
            )
        # Else overplot
        else:
            corner_fig = corner.corner(
                plotting_df, label=posterior_name, fig=corner_fig, **kwargs_to_use
            )
        lines.append(mpllines.Line2D([0], [0], color=kwargs_to_use["color"]))
        labels.append(posterior_name)

    ndim = len(parameters_to_plot)

    figaxes = corner_fig.get_axes()
    figaxes[ndim - 1].legend(lines, labels, prop={"size": "x-large"})
    figaxes[ndim - 2].set_title(title, size="x-large")

    # If composite points are provided plot them
    if ile_points is not None:
        axes = corner.core._get_fig_axes(corner_fig, ndim)

        ile_points.sort_values("lnL", inplace=True)
        # Make a copy of lnLs so our manipulations don't damage it
        color_scale_data = copy.deepcopy(ile_points["lnL"].array)
        vmax = color_scale_data.max()
        if lnL_span_cut is None:
            vmin = color_scale_data.min()
        else:
            vmin = vmax - lnL_span_cut

        cmap = plt.get_cmap("viridis")
        cmap.set_under("gray")
        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])

        for ii, param_ii in enumerate(parameters_to_plot):
            # ii is the y-axis, descending
            for jj, param_jj in enumerate(parameters_to_plot):
                # jj is the x axis, left to right
                if ii == jj or jj > ii:
                    # This is the upper corner and the 1d histograms
                    pass
                else:
                    # This
                    array_1 = ile_points[param_jj]
                    array_2 = ile_points[param_ii]

                    axes[ii, jj].scatter(
                        array_1,
                        array_2,
                        c=color_scale_data,
                        cmap=cmap,
                        s=1 / 16,
                        vmin=vmin,
                    )
        cbar = plt.colorbar(sm, ax=axes)
        cbar.set_label(r"Grid Point $\ln \mathcal{L}$", size="x-large")
    return corner_fig


def parser(config_path):
    import configargparse as cfg

    parser = cfg.ArgumentParser()

    return parser
