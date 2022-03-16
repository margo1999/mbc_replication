import os
import numpy as np
import matplotlib.pyplot as plt
from clock_net import helper, plot_helper

def plot_weight_matrices(axes=None):

    assert axes is not None, 'Need axes object.'

    data_path = get_data_path()
    path_old_weights = os.path.join(data_path, 'ee_connections_before.npy')
    path_new_weights = os.path.join(data_path, 'ee_connections.npy')

    old_connections = np.load(path_old_weights)
    new_connections = np.load(path_new_weights)

    plot_helper.plot_weight_matrix(ax=axes[0], connections=old_connections, title='old weight matrix')
    plot_helper.plot_weight_matrix(ax=axes[1], connections=new_connections, title='new weight matrix')
    plot_helper.plot_diff_weight_matrix(ax=axes[2], connections_new=new_connections, connections_old=old_connections, title='difference of matrices')

    print(np.allclose(plot_helper.matrix_from_connections(old_connections), plot_helper.matrix_from_connections(new_connections)))

def plot_2_mins_weight_matrix(ax=None, filename=None):
    assert ax is not None, 'Need axes object.'
    assert filename is not None, 'Need filename.'

    connections = np.load(filename)
    plot_helper.plot_weight_matrix(ax=ax, connections=connections, title='weight strength')


def get_data_path():
    path_dict = {} 
    path_dict['data_root_path'] = 'data'
    path_dict['project_name'] = 'sequence_learning_performance' 
    path_dict['parameterspace_label'] = 'sequence_learning_and_prediction'

    # get parameters 
    PS, PS_path = helper.get_parameter_set(path_dict)
    replay = False
    PL = helper.parameter_set_list(PS)
    params = PL[0]

    # get data path
    if replay:
        data_path = helper.get_data_path(params['data_path'], params['label'], 'replay')
    else:
        data_path = helper.get_data_path(params['data_path'], params['label'])
    
    return data_path

if __name__ == '__main__':
    with plt.rc_context({

            # plot settings 
            'font.size': 8,
            'legend.fontsize': 6,
            'figure.figsize': (10,5),
            'font.family': 'sans-serif',
            'text.usetex': False
            }):
        figure, axes = plt.subplots(1,3)
        plot_weight_matrices(axes)
        figure.tight_layout()
        plt.show()