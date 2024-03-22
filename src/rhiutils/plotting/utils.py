import numpy as np
import os
import bilby
import copy

from bilby_pipe.utils import DataDump, parse_args
from bilby_glitch.pipe_data_analysis import GlitchDataAnalysisInput
from bilby_glitch.result import JointCBCGlitchResult
from bilby_glitch.joint_likelihood import JointGlitchGravitationalWaveTransient, GravitationalWaveTransient
from bilby_glitch.pipe_parser import glitch_create_parser

from typing import Union, Tuple, List

def set_preferred_matplotlib_defaults():
    import matplotlib

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fontsize = 16
    # ALWAYS USE figsize = (3.375, X) for column plots 
    # figsize = (6.75, X) for rows 
    params = {
    'axes.labelsize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': 8,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'axes.titlesize':fontsize,
    'lines.linewidth':1,  
    'xtick.direction':'in',
    'ytick.direction':'in',
    'text.usetex': True,
    'font.family':'Serif',
    'font.serif':'Computer Modern Roman',
    'axes.grid':True,
    'figure.figsize': (10.125, 6),
    'figure.dpi':250, 
    }

    defaults_corner_kwargs = dict(
                bins=50, smooth=0.9,
                title_kwargs=dict(fontsize=16), color='#0072C1',
                truth_color='tab:orange', quantiles=[0.16, 0.84],
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                plot_density=False, plot_datapoints=True, fill_contours=True,
                max_n_ticks=3, hist_kwargs=dict(density=True))


    for param in params.keys():
        matplotlib.rcParams[param] = params[param]

def get_analysis_from_dag(path_to_dag, case=0):
    with open(path_to_dag, "r") as f:
        lines = f.readlines()
        analysis_line = [x for x in lines if "VARS" in x and "analysis_H1L1" in x and f"data{case}_" in x and "par0" in x][0]
        args, unknown_args = parse_args(analysis_line.split('"')[1].split(" "),
            glitch_create_parser(top_level=False))
    analysis = GlitchDataAnalysisInput(args, unknown_args)
    return analysis

def draw_dict_sample_from_posterior(posterior):
        sample = posterior.sample()
        return {k:[v for v in sample_dict.values()][0] for k, sample_dict in sample.to_dict().items()}

def get_all_analysis_components(
        base_directory : str,
        case : int=0,
        analysis_ifos : str="H1L1",
        frequency_cut_ifo : str = "L1",
        frequencies : Union[None, List[Tuple]] = None,
        joint=True
    ) -> Tuple[
        bilby.core.result.Result,
        DataDump,
        GlitchDataAnalysisInput,
        bilby.gw.likelihood.GravitationalWaveTransient,
        List[bilby.gw.likelihood.GravitationalWaveTransient]
    ]:
    """Get all the junk we need from a run
    
    Parameters
    ==========
    base_directory : str
        The analysis directory
    case : int
        If multiple data gens, the data index
    analysis_ifos : str
        For coherence tests, the result to use (i.e. from which combo of ifos)
    frequency_cut_ifo : str
        The ifo to apply the frequency cut in
    frequencies : Union[None, List[Tuple]]
        If None, just returns original likelihood, if provided also returns likelihoods with different min/max frequencies
        First element min, second element max, if either is None the original value is used
    joint : bool
        Whether to use the joint likelihood class
    
    Returns
    =======
    bilby.core.result.Result
        The analysis result
    DataDump
        The data dump object
    GlitchDataAnalysisInput
        The reconsructed analysis object
    bilby.gw.likelihood.GravitationalWaveTransient
        The likelihood used in the analysis
    List[bilby.gw.likelihood.GravitationalWaveTransient]
        A list of likelihoods as the base, but with different frequency cuts
    """
    merge_result_file = [os.path.join(base_directory, "final_result", x) for x in os.listdir(os.path.join(base_directory, "final_result")) if f"data{case}_" in x and f"_{analysis_ifos}_" in x and "result.hdf5" in x][0]
    result = bilby.core.result.read_in_result(merge_result_file)
    result = JointCBCGlitchResult(result)

    data_file = [os.path.join(base_directory, "data", x) for x in os.listdir(os.path.join(base_directory, "data")) if f"data{case}_" in x and "data_dump.pickle" in x][0]
    data = DataDump.from_pickle(data_file)

    owd = os.getcwd()
    os.chdir(os.path.join(base_directory, ".."))
    submit_dag = [os.path.join(base_directory, "submit", x) for x in os.listdir(os.path.join(base_directory, "submit")) if f"dag" in x and x.split(".")[-1] == "submit"][0]
    analysis = get_analysis_from_dag(submit_dag, case=case)
    os.chdir(owd)

    likelihood_args = [
            analysis.interferometers,
            analysis.waveform_generator,
    ]
    likelihood_kwargs = dict(
        priors = copy.deepcopy(result.priors),
        distance_marginalization=True
    )
    if joint:
        likelihood_class = JointGlitchGravitationalWaveTransient
        likelihood_args.append(analysis.joint_waveform_generators)
    else:
        likelihood_class = GravitationalWaveTransient
    
    base_likelihood = likelihood_class(*likelihood_args, **likelihood_kwargs)

    if frequencies is None:
        return result, data, analysis, base_likelihood, []
    test_frequency_likelihoods = []
    for frequency_tuple in frequencies:
        test_interferometers = copy.deepcopy(base_likelihood.interferometers)
        frequency_cut_ifo_index = [ii for ii,x in enumerate(base_likelihood.interferometers) if x.name == frequency_cut_ifo][0]

        minimum_frequency = frequency_tuple[0] if frequency_tuple[0] is not None else base_likelihood.interferometers[frequency_cut_ifo_index].minimum_frequency
        maximum_frequency = frequency_tuple[1] if frequency_tuple[1] is not None else base_likelihood.interferometers[frequency_cut_ifo_index].maximum_frequency

        test_interferometers[frequency_cut_ifo_index].minimum_frequency = minimum_frequency
        test_interferometers[frequency_cut_ifo_index].maximum_frequency = maximum_frequency

        likelihood_args[0] = test_interferometers
        likelihood_kwargs['priors'] = copy.deepcopy(result.priors)

        test_frequency_likelihoods.append(likelihood_class(*likelihood_args, **likelihood_kwargs))
    return result, data, analysis, base_likelihood, test_frequency_likelihoods
        
