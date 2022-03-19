"""
This script instantiates the Clock Network model, creates, connects, and simulates the network 

Authors
~~~~~~~
Jette Oberlaender, Younes Bouhadjar
"""

import nest
import sys
import time
import numpy as np
import os
from pprint import pprint

from clock_net import model, helper

def generate_reference_data():

    # ###########################################################
    # import nestml module
    # ===========================================================
    #nest.Install('../../module/nestml_active_dend_module')

    #############################################################
    # get network and training parameters 
    # ===========================================================
    PS = model.get_parameters()

    # parameter-set id from command line (submission script)
    PL = helper.parameter_set_list(PS) 
   
    #TODO: use argparse with default values
    try: 
        batch_id=int(sys.argv[1])
        batch_array_id=int(sys.argv[2])
        JOBMAX=int(sys.argv[3])
        array_id=batch_id*JOBMAX+batch_array_id
    except:
        array_id = 0

    params = PL[array_id]
    resultpath = helper.get_data_path(params['data_path'], params['label'])

    # start time 
    time_start = time.time()

    # ###############################################################
    # specify sequences
    # ===============================================================
    sequences, _, vocabulary = helper.generate_sequences(params['task'], params['data_path'], params['label'])

    # ###############################################################
    # create network
    # ===============================================================
    model_instance = model.Model(params, sequences, vocabulary)
    time_model = time.time()

    model_instance.create()
    time_create = time.time()

    # ###############################################################
    # save params to txt file
    # ===============================================================
    parameterspacepath = os.path.join(resultpath, 'parameter_space.txt')
    with open(parameterspacepath, 'wt') as file:
        pprint(params, stream=file)
        file.close()

    # TODO: Add time measurement 

    # ###############################################################
    # connect the network
    # ===============================================================
    model_instance.connect()
    time_connect = time.time()
    
    # store connections before learning
    if params['store_connections']:
        model_instance.save_connections(synapse_model=params['syn_dict_ee']['synapse_model'], fname='ee_connections_before')
        model_instance.save_connections(synapse_model=params['syn_dict_ei']['synapse_model'], fname='ei_connections_before')
    time_store_connection_before = time.time()

    # ###############################################################
    # simulate the network
    # ===============================================================
    model_instance.simulate()
    time_simulate = time.time()

    # store connections after learning
    if params['store_connections']:
        model_instance.save_connections(synapse_model=params['syn_dict_ee']['synapse_model'], fname='ee_connections')
        model_instance.save_connections(synapse_model=params['syn_dict_ei']['synapse_model'], fname='ei_connections')
    time_store_connection_after = time.time()

    def print_times(file=sys.stdout):

        print(
        '\nTimes of Rank {}:\n'.format(
            nest.Rank()) +
        '  Total time:                 {:.3f} s\n'.format(
            time_store_connection_after -
            time_start) +
        '  Time to initialize:         {:.3f} s\n'.format(
            time_model -
            time_start) +
        '  Time to create:             {:.3f} s\n'.format(
            time_create -
            time_model) +
        '  Time to connect:            {:.3f} s\n'.format(
            time_connect -
            time_create) +
        '  Time to store connections:  {:.3f} s\n'.format(
            time_store_connection_before -
            time_connect) +    
        '  Time to simulate:           {:.3f} s\n'.format(
            time_simulate -
            time_store_connection_before) +
        '  Time to store connections:  {:.3f} s\n'.format(
            time_store_connection_after -
            time_simulate),
            
            file=file)
    
    print_times()

    # ###############################################################
    # save times to txt file
    # ===============================================================
    timepath = os.path.join(resultpath, 'simulation_times.txt')
    print_times(file=open(timepath, 'a'))


    return resultpath

if __name__ == '__main__':    
    from figures import plot_results
    import matplotlib.pyplot as plt

    resultpath = generate_reference_data()
    plot_results.plot_results(os.path.join(resultpath,f"simulation_finished"))
    plt.show()

    
    
