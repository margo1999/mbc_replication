"""
This script instantiates the Clock Network model, creates, connects, and simulates the network 

Authors
~~~~~~~
Jette Oberländer, Younes Bouhadjar
"""

import nest
import sys
import time
import numpy as np

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
    # connect the netwok
    # ===============================================================
    model_instance.connect()
    time_connect = time.time()
    
    # store connections before learning
    if params['store_connections']:
        model_instance.save_connections(fname='ee_connections_before')

    # ###############################################################
    # simulate the network
    # ===============================================================
    model_instance.simulate()
    time_simulate = time.time()

    # store connections after learning
    if params['store_connections']:
        model_instance.save_connections(fname='ee_connections')

    print(
        '\nTimes of Rank {}:\n'.format(
            nest.Rank()) +
        '  Total time:          {:.3f} s\n'.format(
            time_simulate -
            time_start) +
        '  Time to initialize:  {:.3f} s\n'.format(
            time_model -
            time_start) +
        '  Time to create:      {:.3f} s\n'.format(
            time_create -
            time_model) +
        '  Time to connect:     {:.3f} s\n'.format(
            time_connect -
            time_create) +
        '  Time to simulate:    {:.3f} s\n'.format(
            time_simulate -
            time_connect))


generate_reference_data()
