import os
import sys
from joblib import Parallel, delayed

from clock_net import helper 

def sim(i):
    cmd = "python3 %s 0 %d 0" % (simulation_script, i)
    print(cmd)
    os.system(cmd)

# get commmand line arguments
#try:
#    simulation_script = sys.argv[1]
#except:
#    print("provide simulation script!")
#    sys.exit(1)

if len(sys.argv) == 2:
    simulation_script = sys.argv[1]
elif len(sys.argv) == 1:
    simulation_script = ''
else:
    print("Too many arguments")
    sys.exit(1)

if simulation_script == "training.py":

    import parameters_space as data_pars 
 
    # get parameters 
    PS = data_pars.p
else:
    path_dict = {}
    path_dict['data_root_path'] = "data"
    path_dict['project_name'] = "sequence_learning_performance"
    path_dict['parameterspace_label'] = "stimulus_timing_analysis"

    # get parameters 
    PS, data_path = helper.get_parameter_set(path_dict)
 
# parameters list 
PL = helper.parameter_set_list(PS)

# save parameters.py  
if simulation_script == "training.py":
    helper.copy_scripts(PS['data_path'], "parameters_space.py")

# simulation 
N = len(PL)
print(f"{N=}" + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
#Parallel(n_jobs=2)(delayed(sim)(i) for i in range(N))

#TODO: remove test
#just for testing
N = 1
for i in range(N):
     sim(i)
