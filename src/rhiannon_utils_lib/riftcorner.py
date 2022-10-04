import ast
import copy
import logging
import os
from itertools import cycle
from types import NoneType
from typing import Dict, List, Tuple, Union

import configargparse as cfg
import corner
import matplotlib.lines as mpllines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.gw.conversion import generate_mass_parameters

from .fileutils import get_suffixed_path, write_altered_config

logger = logging.getLogger(__name__)

# Map RIFT names to bilby names
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

# Map bilby names to latex representations
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


def get_posterior_dataframe(posterior_file: str) -> pd.DataFrame:
    """
    Get the posterior dataframe from a posteriors_samples.dat file.
    #TODO Make this more flexible - may involve expanding the above.

    Parameters
    ----------
    posterior_file : str
        The path to the file to read.

    Returns
    -------
    df : pd.DataFrame
        The dataframe representing the data in the file
        using bilby names where possible.
    """
    # Get the header, process it as necessary
    with open(posterior_file, "r") as f:
        header = f.readline()
    header = header.replace("#", "").replace("  ", " ").strip().split(" ")
    # Get the corresponding bilby header, mapping to bilby names where possible
    bilby_header = [
        (lambda x: _rift_to_bilby[x] if (x in _rift_to_bilby.keys()) else x)(x)
        for x in header
    ]
    # Read the data file
    return pd.read_csv(
        posterior_file, sep=" ", names=bilby_header, header=None, skiprows=1
    )


def make_results_dict(
    posterior_file_paths_list: List[str],
    posterior_names: List[str],
    relative_dir="",
) -> Dict[str, pd.DataFrame]:
    """
    For a list of paths and corresponding names, read them in and make a dict of dataframes

    Parameters
    ----------
    posterior_file_paths_list : list[str]
        A list of paths to posterior_samples.dat files.
        May be relative, in which case relative_dir is prepended,
        with user home expanded out
    posterior_names : list[str]
        A list of the names to give the posteriors.
    relative_dir : str
        A path to prepend to all the file names.
        E.g. can be the directory they are all located,
        then only their names need be passed.

    Returns
    -------
    posterior_dataframes : dict[str, pd.DataFrame]
    """
    posterior_dataframes = dict()
    # For each posterior file, get the correct path, get the dataframe, assign by name
    for ii, posterior_file in enumerate(posterior_file_paths_list):
        path_to_file = os.path.expanduser(os.path.join(relative_dir, posterior_file))
        posterior_dataframes[posterior_names[ii]] = get_posterior_dataframe(
            path_to_file
        )
    return posterior_dataframes


def get_ile_points(composite_path: str) -> pd.DataFrame:
    """
    Read the ILE points from a composite file

    Parameters
    ----------
    composite_path : str
        The path to the composite (.composite or all.net) file

    Returns
    -------
    dataframe : pd.DataFrame

    """
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
    results_dict: Dict[str, pd.DataFrame],
    parameters_to_plot: List[str],
    ile_points: Union[NoneType, pd.DataFrame] = None,
    lnL_span_cut: Union[NoneType, float] = None,
    title: Union[NoneType, str] = None,
    vlines: Dict[str, Tuple[List[str], List[float]]] = {},
    **kwargs,
) -> plt.figure:
    """
    Using posteriors and optionally grid points, produce a corner plot

    Parameters
    ----------
    results_dict : Dict[str, pd.DataFrame]
        A dict which maps the name of the posterior (on the legend)
        to the dataframe containing that posterior.
    parameters_to_plot : List[str]
        The list of parameters to plot on the corner plot.
        Naturally, all posteriors must contain these parameters.
    ile_points : Union[NoneType, pd.DataFrame]
        Optionally, the ILE points to plot on the corner.
    lnL_span_cut : Union[NoneType, float]
        Optionally, the range of lnL's to plot in color scale (all others are grayed).
        If None all grid points are colored, this may make the scale uninformative.
    title : Union[NoneType, str]
        Optionally, the title of the plot.
    vlines : Dict[str, Tuple[List[str], List[float]]]
        A dictionary containing lines to overplot.
        The key is the label for the line.
        The first element of the tuple is a list of parameters to plot.
        The second element are the corresponding parameter values.
    """
    # Setup kwargs for plotting - in most cases use Bilby defaults
    defaults_kwargs = dict(
        bins=50,
        smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16),
        truth_color="tab:orange",
        quantiles=[],
        levels=(0.90,),
        plot_density=False,
        plot_datapoints=False,
        no_fill_contours=True,
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
        # Do stuff to help form the legend
        lines.append(mpllines.Line2D([0], [0], color=kwargs_to_use["color"]))
        labels.append(posterior_name)

    ndim = len(parameters_to_plot)

    figaxes = corner_fig.get_axes()

    # Optionally add a title - tends to be pretty ugly.
    if title is not None:
        figaxes[ndim - 2].set_title(title, size="x-large")

    # Setup color for lines, and linestyles to iterate cyclically
    linestyles = cycle([":", "--", "-.", "-"])
    vline_color = "black"
    for line_label, (params, param_values) in vlines.items():
        # Get the linestyle to use
        vline_linestyle = next(linestyles)
        xs = []
        for parameter in parameters_to_plot:
            if parameter in params:
                # If this parameter is one to plot, then grab the corresponding value
                value_for_param = param_values[params.index(parameter)]
                xs.append(value_for_param)
            else:
                # Else use None to not plot on this subaxis
                xs.append(None)
        # Overplot and add to lists for legend
        corner.overplot_lines(
            corner_fig, xs, linestyle=vline_linestyle, color=vline_color
        )
        lines.append(
            mpllines.Line2D([0], [0], color=vline_color, linestyle=vline_linestyle)
        )
        labels.append(line_label)

    # Make the legend
    figaxes[ndim - 1].legend(lines, labels, prop={"size": "x-large"})

    # If composite points are provided plot them
    if ile_points is not None:
        axes = corner.core._get_fig_axes(corner_fig, ndim)

        ile_points.sort_values("lnL", inplace=True)
        # Make a copy of lnLs so our manipulations don't damage it
        color_scale_data = copy.deepcopy(ile_points["lnL"].array)
        # Get vmax, and either fetch vmin or construct an lnL span
        vmax = color_scale_data.max()
        if lnL_span_cut is None:
            vmin = color_scale_data.min()
        else:
            vmin = vmax - lnL_span_cut

        # Handle color stuff
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
                    # This is the upper right triangle and the 1d histograms
                    pass
                else:
                    # This is the lower left triangle
                    array_1 = ile_points[param_jj]
                    array_2 = ile_points[param_ii]

                    # Plot the points
                    axes[ii, jj].scatter(
                        array_1,
                        array_2,
                        c=color_scale_data,
                        cmap=cmap,
                        s=1 / 16,
                        vmin=vmin,
                    )
        # Make the colorbar
        cbar = plt.colorbar(sm, ax=axes)
        cbar.set_label(r"Grid Point $\ln \mathcal{L}$", size="x-large")
    return corner_fig


def parse_corner_args(config_path=None):
    if config_path is not None:
        parser = cfg.ArgumentParser(config_path)
    else:
        parser = cfg.ArgumentParser()

    parser.add_argument(
        "config", is_config_file=True, help="The path to the config file to use"
    )
    file_parser = parser.add_argument_group(
        title="File Arguments",
        description="Arguments describing paths of files to use, and names for posteriors",
    )
    file_parser.add_argument(
        "--posterior-files",
        action="append",
        help="A path (absolute or relative - see relative dir) to the posterior file(s) to use",
    )
    file_parser.add_argument(
        "--relative-dir",
        type=str,
        help="The relative directory for all of the posteriors (e.g. the run directory) and the composite file",
        default="",
    )
    file_parser.add_argument(
        "--composite-file",
        default=None,
        help="Optionally, the path to a file with the composite points",
    )
    file_parser.add_argument(
        "--output-dir",
        default="~",
        help="The output directory for the corner plot,\
        and a config file updated with any command line arguments passed\
        If this directory already exists, a numerical suffix will be appended\
        (e.g. output_dir --> output_dir_1, output_dir_1 --> output_dir_2)"
        # TODO could also add other stuff to this directory?
        # As many plots as the heart desires...
    )

    plot_parser = parser.add_argument_group(
        title="Plot Arguments", description="Arguments for plotting"
    )
    plot_parser.add_argument(
        "--posterior-names",
        action="append",
        help="",
    )
    plot_parser.add_argument(
        "--log-l-span-cut",
        type=float,
        default=None,
        help="If passed, then color scale from (max lnL - cut, max lnL), else go min to max",
    )
    plot_parser.add_argument(
        "--params-to-plot",
        action="append",
        help="Pass multiple times for the parameters to plot on the corner plot",
    )
    plot_parser.add_argument(
        "--title",
        default=None,
        help="If passed will put title onto plot - note it's a bit ugly",
    )
    plot_parser.add_argument(
        "--vlines",
        default="{}",
        help="A dict of {label:(param, value)} by which to plot vertical lines",
    )
    plot_parser.add_argument(
        "--extra-kwargs",
        default="{}",
        help="Any extra plotting kwargs to use - passed as the string rep of a dict",
    )
    arguments = parser.parse_args()
    return arguments, parser


def main():
    # Standard logging for a main process
    logging.basicConfig(level=logging.INFO)

    # Setup parser
    arguments, parser = parse_corner_args()

    arguments.posterior_files = [x.replace("'", "") for x in arguments.posterior_files]
    arguments.params_to_plot = [x.replace("'", "") for x in arguments.params_to_plot]
    arguments.posterior_names = [x.replace("'", "") for x in arguments.posterior_names]

    # Get results from posteriors
    results_dict = make_results_dict(
        arguments.posterior_files,
        arguments.posterior_names,
        arguments.relative_dir,
    )

    # If applicable get composite points
    if arguments.composite_file is not None:
        ile_points = get_ile_points(
            os.path.join(arguments.relative_dir, arguments.composite_file)
        )

    # Do the plotting
    corner_fig = plot_multiple_RIFT(
        results_dict,
        arguments.params_to_plot,
        ile_points=ile_points,
        lnL_span_cut=arguments.log_l_span_cut,
        title=arguments.title,
        vlines=ast.literal_eval(arguments.vlines),
        **ast.literal_eval(arguments.extra_kwargs),
    )

    # Make a unique (almost) requested directory to write to
    write_dir = os.path.expanduser(get_suffixed_path(arguments.output_dir))
    os.makedirs(write_dir)
    # Write things
    write_altered_config(
        arguments,
        parser,
        os.path.join(write_dir, "modified_config.cfg"),
    )
    corner_fig.savefig(os.path.join(write_dir, "rift_corner.jpg"))
