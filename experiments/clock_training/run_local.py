"""
This script has to be executed when the experiment 'clock training' should be started on your local machine. Based on the
number of parameter sets in parameters_space.py the corresponding number of jobs are launched.
Make sure, if you want to change certain simulation properties, that you do this before in parameters_space.py.
Set NUMBER_OF_PARALLEL_JOBS to determine how many jobs will be executed in parallel.
"""
import os
from pathlib import Path
from joblib import Parallel, delayed

from mbc_network.helper import training_helper
from experiments.parameters_space import param_recurrent

NUMBER_OF_PARALLEL_JOBS = 2

if __name__ == '__main__':

    simulation_script_path = Path(__file__).parent.joinpath("training.py")  # run_local.py and training.py must be in the same directory
    assert simulation_script_path.exists(), f"The path {simulation_script_path} does not exist!"
    print(f"{simulation_script_path=}")

    # Parameters list
    parameter_set_list = training_helper.parameter_set_list(param_recurrent)

    # Simulation
    num_tasks = len(parameter_set_list)
    print(f"{num_tasks} tasks are being executed...")

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    Parallel(n_jobs=NUMBER_OF_PARALLEL_JOBS)(delayed(training_helper.run_training)(simulation_script_path, idx) for idx in range(num_tasks))
