""" TODO
"""
import os
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
from clock_net import plot_helper
from clock_net import helper
from clock_net.helper import load_data, load_spike_data


def plot_spikes(ax=None):
    """[summary]
    """
    assert ax is not None, 'Need axes object.'

    path_dict = {}
    path_dict['data_root_path'] = 'data'
    path_dict['project_name'] = 'sequence_learning_performance'
    path_dict['parameterspace_label'] = 'sequence_learning_and_prediction'

    # get parameters
    PS, PS_path = helper.get_parameter_set(path_dict)  # TODO remove helper.parameter_set_list() and fix data path
    replay = False

    # PS['DeltaT'] = 40.
    PL = helper.parameter_set_list(PS)
    params = PL[0]

    # get trained sequences
    # TODO load data training fails if the sequences are not of the same length
    sequences = load_data(PS_path, 'training_data')
    vocabulary = load_data(PS_path, 'vocabulary')

    print('#### sequences used for training ### ')
    for i, sequence in enumerate(sequences):
        seq = ''
        for char in sequence:
            seq += str(char).ljust(2)
        print('sequence %d: %s' % (i, seq))

    # get data path
    if replay:
        data_path = helper.get_data_path(params['data_path'], params['label'], 'replay')
    else:
        data_path = helper.get_data_path(params['data_path'], params['label'])

    allspiketypes = True
    if not allspiketypes:
        filename = os.path.join(data_path, f"spikes_{params['training_iterations']-1}.pickle")
        spikes = load(open(filename, "rb"))
        exh_spikes_array = np.transpose([spikes['sr_senders_exh'], spikes['sr_times_exh']])

        plot_helper.plot_spikes(ax=ax, exh_spikes=exh_spikes_array)
    else:
        inh_spikes = load_spike_data(data_path, 'inh_spikes')
        exh_spikes = load_spike_data(data_path, 'exh_spikes')
        gen_spikes = load_spike_data(data_path, 'generator_spikes')
        plot_helper.plot_spikes(ax=ax, exh_spikes=exh_spikes, inh_spikes=inh_spikes, gen_spikes=gen_spikes)


def plot_2_mins_spikes(ax=None, filename=None):
    assert ax is not None, 'Need axes object.'
    assert filename is not None, 'Need filename.'

    # load spikes from reference data
    spikes = load(open(filename, "rb"))

    exh_spikes_array = np.transpose([spikes['sr_senders_exh'], spikes['sr_times_exh']])
    print(f"{exh_spikes_array.size=}", f"{len(spikes['sr_senders_exh'])=}", f"{len(spikes['sr_times_exh'])=}")

    plot_helper.plot_spikes(ax=ax, exh_spikes=exh_spikes_array)


if __name__ == '__main__':
    with plt.rc_context({
                        # plot settings
                        'font.size': 8,
                        'legend.fontsize': 6,
                        'figure.figsize': (10, 5),
                        'font.family': 'sans-serif',
                        'text.usetex': False
                        }):
        figure, axes = plt.subplots(1, 1)
        plot_spikes(axes)
        figure.tight_layout()
        plt.show()
