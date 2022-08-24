"""
This script instantiates the Clock Network model, creates, connects, and simulates the network TODO

Authors
~~~~~~~
Jette Oberlaender, Younes Bouhadjar
"""

import os
import sys
import time
from pprint import pprint

import nest
import numpy as np

import mbc_network.models.clock_model as clock_model
import mbc_network.helper.training_helper as training_helper
from experiments.parameters_space import param_recurrent as paramspace_recurrent


def generate_reference_data():
    """_summary_ TODO documentation

    Returns:
        _type_: _description_
    """

    # ===========================================================
    # import nestml module
    # ===========================================================
    # nest.Install('../../module/nestml_active_dend_module')

    # ===============================================================
    # get parameter set
    # ===============================================================

    task_id = 0

    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])

    parameterset_idx = task_id

    # parameter-set id from command line (submission script)
    paramset_recurrent = training_helper.parameter_set_list(paramspace_recurrent)[parameterset_idx]  
    resultpath = training_helper.get_data_path(paramset_recurrent['data_path'], paramset_recurrent['label'])

    # start time
    time_start = time.time()

    # ===========================================================
    # create network
    # ===========================================================
    model_instance = clock_model.Model(paramset_recurrent)
    time_model = time.time()

    model_instance.create()
    time_create = time.time()

    # ===========================================================
    # save params to txt file
    # ===========================================================
    parameterspacepath = os.path.join(resultpath, 'parameter_space.txt')
    with open(parameterspacepath, 'wt') as file:
        pprint(paramset_recurrent, stream=file)
        file.close()

    # TODO: Add time measurement

    # ===========================================================
    # connect the network
    # ===========================================================
    model_instance.connect()
    time_connect = time.time()

    # store connections before learning
    if paramset_recurrent['store_connections']:
        model_instance.save_connections(synapse_model=paramset_recurrent['syn_dict_ee']['synapse_model'], fname='ee_connections_before')
        if paramset_recurrent['syn_dict_ei']['synapse_model'] != 'static_synapse':
            model_instance.save_connections(synapse_model=paramset_recurrent['syn_dict_ei']['synapse_model'], fname='ei_connections_before')  # TODO: better ask for source and target
        model_instance.save_connections(fname='all_connections_before')
    time_store_connection_before = time.time()

    # ===========================================================
    # simulate the network
    # ===========================================================
    model_instance.simulate()
    time_simulate = time.time()

    # store connections after learning
    if paramset_recurrent['store_connections']:
        model_instance.save_connections(synapse_model=paramset_recurrent['syn_dict_ee']['synapse_model'], fname='ee_connections')
        if paramset_recurrent['syn_dict_ei']['synapse_model'] != 'static_synapse':
            model_instance.save_connections(synapse_model=paramset_recurrent['syn_dict_ei']['synapse_model'], fname='ei_connections')  # TODO: better ask for source and target
        model_instance.save_connections(fname='all_connections')
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

    # ===========================================================
    # save times to txt file
    # ===========================================================
    timepath = os.path.join(resultpath, 'simulation_times.txt')
    print_times(file=open(timepath, 'a'))

    return resultpath


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from mbc_network.plotting import plot_results
    matplotlib.use("Agg")

    resultpath = generate_reference_data()
    plot_results.plot_results(os.path.join(resultpath, "simulation_finished"))
    plt.show()
