"""
This script has to be executed when the experiment 'clock training' should be started on a computer cluster. Based on the
number of parameter sets in parameters_space.py the corresponding number of jobs are launched by sbatch submitting a batch script to Slurm.
Make sure, if you want to change certain simulation properties, that you do this before in parameters_space.py.
You also need to modify the instructions in the batch script so that it is tailored to the cluster you are using.
"""
import os
from pathlib import Path
import yaml

from mbc_network.helper import training_helper
from experiments.parameters_space import param_recurrent

# Slurm stuff TODO
assert os.path.isfile('../config.yaml'), "\n>>> ERROR: Create a config file containing a dictionary with your email: config['email']" \
                                         + " and a path to where you want to store the log files config['path'] \n"

with open('../config.yaml', 'r') as cfgfile:
    params_config = yaml.load(cfgfile, Loader=yaml.FullLoader)
    email = params_config['email']
    path = params_config['path']

logpath = os.path.join(path, 'log')
if not os.path.exists(logpath):
    os.makedirs(logpath)
    print("Created", logpath)

# get path to training script
simulation_script_path = Path(__file__).parent.joinpath("training.py")                              # run_cluster.py and training.py must be in the same directory
assert simulation_script_path.exists(), f"The path {simulation_script_path} does not exist!"

# get all parameter sets to determine the number of jobs
parameter_set_list = training_helper.parameter_set_list(param_recurrent)                            # get all parameter sets
params = parameter_set_list[0]                                                                      # get the first parameter set for basic settings (name of bash script,...) TODO should I prohibit that in some cases one can set up a ParameterRange()?
num_tasks = len(parameter_set_list)                                                                 # number of tasks
print(f"\nNumber of parameter sets: {num_tasks}\n")

# create submission script and fill with simulation instructions
submission_script = f"{params['data_path']['project_name']}.sh"                                     # name of bash script
file = open(submission_script, 'w')                                                                 # open bash script
file.write('#!/bin/bash\n')                                                                         # TODO what does this mean?
file.write('#SBATCH --job-name ' + params['data_path']['project_name'] + '\n')                      # set the name of the job
file.write('#SBATCH --array 0-%d\n' % (num_tasks - 1))                                              # launch an array of jobs
file.write('#SBATCH --time 5-00:00:00\n')                                                           # specify a time limit
file.write('#SBATCH --ntasks 1\n')                                                                  # TODO what does this mean?
file.write('#SBATCH --cpus-per-task %d\n' % params['n_threads'])                                    # specify how many CPUs per task should be used
file.write('#SBATCH -o %s' % path + '/log/job_%A_%a.o\n')                                           # redirect stderr and stdout to the same file
file.write('#SBATCH -e %s' % path + '/log/job_%A_%a.e\n')                                           # redirect stderr and stdout to the same file
file.write('#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE\n')                                          # send email notifications
file.write('#SBATCH --mail-user=%s\n' % email)                                                      # set email for notifications
file.write('#SBATCH --partition=blaustein,hambach\n')                                               # defines access for a group of nodes in descending order
file.write('#SBATCH --mem=10000\n')                                                                 # reserve memory
file.write('srun python %s $SLURM_ARRAY_TASK_ID \n' % simulation_script_path)                       # call simulation script
file.write('scontrol show jobid ${SLURM_JOBID} -dd # Job summary at exit')                          # call to view Slurm configuration
file.close()                                                                                        # close bash script

# execute submission_script
print("submitting %s" % (submission_script))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.system("sbatch ./%s" % submission_script)
