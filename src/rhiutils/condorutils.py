import logging
import os
import shutil

from glue import pipeline

from .fileutils import get_suffixed_path

logger = logging.getLogger(__name__)


def standard_job_constructor(
    executable_name,
    job_name,
    job_arguments,
    universe="vanilla",
    user_name=None,
    accounting=None,
    request_memory="1 Gb",
    request_disk="1 Gb",
    initialdir=None,
    logdir=None,
    extra_cmds=None,
):
    # Setup cluster info
    if user_name is None:
        try:
            user_name = os.environ["LIGO_ACCOUNTING_USER"]
        except KeyError:
            raise KeyError(
                "No LIGO user name given, and none is present in environment.\
                Accordingly, cannot auto-initialize condor."
            )
    if accounting is None:
        try:
            accounting = os.environ["LIGO_ACCOUNTING"]
        except KeyError:
            raise KeyError(
                "No LIGO accounting info given, and none is present in environment.\
                Accordingly, cannot auto-initialize condor."
            )

    # Set default initial directory
    if initialdir is None:
        initialdir = get_suffixed_path(os.getcwd())
    # If no logs path is given, auto set
    if logdir is None:
        logdir = os.path.join(initialdir, "logs")

    # Fetch the exe path
    job_exe = shutil.which(executable_name)

    # Setup the dagjob
    job = pipeline.CondorDAGJob(universe=universe, executable=job_exe)

    # Set the log/err/out file destinations
    job.set_log_file(os.path.join(logdir, f"{job_name}_$(cluster).log"))
    job.set_stdout_file(os.path.join(logdir, f"{job_name}_$(cluster).out"))
    job.set_stderr_file(os.path.join(logdir, f"{job_name}_$(cluster).err"))

    # Setup igwn info
    job.add_condor_cmd("accounting_group", accounting)
    job.add_condor_cmd("accounting_group_user", user_name)

    # Setup job requests
    job.add_condor_cmd("request_memory", request_memory)
    job.add_condor_cmd("request_disk", request_disk)

    # Don't get notified
    # TODO - get notified?
    job.add_condor_cmd("notification", "never")

    # Set the initialdir
    job.add_condor_cmd("initialdir", initialdir)

    # I have no idea what this does but apparently it was needed at some point in the past?
    job._CondorJob__queue = 1

    # Add the arguments
    job.add_arg(job_arguments)

    # Set the subfile
    job.set_sub_file(os.path.join(initialdir, f"{job_name}.sub"))

    # If there are extra lines to add do so here
    if extra_cmds is not None:
        for cmd in extra_cmds:
            job.add_condor_cmd(cmd[0], cmd[1])

    return job


def standard_node_constructor(job, dag, parents=[], macros=[], retry=5):
    # Basic setup
    job_node = pipeline.CondorDAGNode(job)
    # Add parents iteratively
    for parent in parents:
        job_node.add_parent(parent)
    # Add any macros
    for macro in macros:
        job_node.add_macro(macro[0], macro[1])
    # Set the retry number
    job_node.set_retry(retry)
    # Add to the dag
    dag.add_node(job_node)

    return job_node
