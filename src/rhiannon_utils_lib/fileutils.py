import copy
import logging
import os
from typing import List

import configargparse as cfg

logger = logging.getLogger(__name__)


def write_altered_config(
    arguments: cfg.Namespace,
    parser: cfg.ArgumentParser,
    output_path: str,
    argument_groups_to_pop: List[str] = ["positional arguments", "options"],
    arguments_to_pop: List[str] = ["config"],
):
    """
    A function to take a parsed configargparser object, clean it up, and write it to a .cfg

    Parameters
    ---------------
    arguments : Namespace
        The arguments parsed out of some ArgumentParser
    parser : ArgumentParser
        The parser associated with the arguments
    output_path : str
        The path for the new config file
    argument_groups_to_pop : list
        A list containing the names of argument groups which should be popped out of the config
        e.g. 'parser_condor' to remove condor arguments
    arguments_to_pop : list
        A list containing individual argument names which should be removed, e.g. 'config'

    Writes
    ----------------
    Parsed Config
        A config file which contains the full arguments (both config and command line) from the parser
        Grouped by argument group
    """
    import configparser

    # Dictionaries are easy to work with
    arguments_dict = arguments.__dict__

    # Use the parser to get the argument groups, which should correspond to section headers
    argument_groups = copy.copy(parser._action_groups)

    # This will be a dict of all the arguments and their values, grouped by argument group
    grouped_argument_names = dict()

    # Loop over argument groups
    for group in argument_groups:
        # Preset the internal dictionary for the group
        parameter_dict = dict()
        # For each action in the group
        # Use the name to get the value from the argument group
        # Then make the key value pair
        for action in group._group_actions:
            action_name = action.dest
            if action_name not in ["help"]:
                parameter_dict[action_name] = arguments_dict[action_name]
        # Write the internal dict to the larger dict, under the group name
        grouped_argument_names[group.title] = parameter_dict

    # Copy out the dictionary now, as we prepare to clean it
    cleaned_grouped_arguments = copy.copy(grouped_argument_names)
    cleaned_grouped_arguments_keys = list(cleaned_grouped_arguments.keys())

    # Loop over groups
    for group in cleaned_grouped_arguments_keys:
        # Remove the whole group if appropriate
        if group in argument_groups_to_pop:
            cleaned_grouped_arguments.pop(group)
            continue
        # Otherwise, loop over arguments in the pop_list and pop them if they are in this group
        for key in arguments_to_pop:
            if key in cleaned_grouped_arguments[group]:
                cleaned_grouped_arguments[group].pop(key)

        # Set original keys to loop over
        original_cleaned_group_keys = list(cleaned_grouped_arguments[group].keys())

        # Now cast the rest back to strings
        for key in original_cleaned_group_keys:
            named_value = cleaned_grouped_arguments[group].pop(key)
            cleaned_grouped_arguments[group][key.replace("_", "-")] = str(named_value)

    # Make a new parser, to write to the config
    writing_parser = configparser.ConfigParser()
    # Looping over groups again, assign section headers for each group
    for group in cleaned_grouped_arguments:
        writing_parser[group] = cleaned_grouped_arguments[group]
    # write it all to the output path
    with open(output_path, "w") as f:
        writing_parser.write(f)


def get_suffixed_path(
    file_path: str,
) -> str:
    """
    Sometimes you want to make a file or directory, but it already exists
    This checks if it exists.
    If it does it puts a suffix on the end (e.g. _1)
    and if it already has one it counts up.

    Parameters
    ----------
    file_path : str
        The base file path to check and append to

    Returns
    -------
    new_file_path : str
        The unique file path to make to, altered if necessary
    """

    if os.path.exists(os.path.expanduser(file_path)):
        # Get the pieces of the path
        file_path_split = file_path.rstrip("/").split("/")
        # Get the file path, if any, and the file name
        if len(file_path_split) > 1:
            path_base = os.path.join(*file_path_split[:-1])
        else:
            path_base = ""
        path_name = file_path_split[-1]
        # Split the name by underscores, to check if we've been here before
        path_name_split = path_name.split("_")
        if path_name_split[-1].isdigit():
            # If there is a numerical suffix, we will increment it
            path_name_prefix = "_".join(path_name_split[:-1])
            path_name_suffix = path_name_split[-1]
            # incrementing
            path_name_suffix = str(int(path_name_suffix) + 1)
            return get_suffixed_path(
                os.path.join(path_base, f"{path_name_prefix}_{path_name_suffix}")
            )
        else:
            # If there isn't a number at the end, we can just stick a 1 on it
            return get_suffixed_path(os.path.join(path_base, path_name + "_1"))
    else:
        return file_path
