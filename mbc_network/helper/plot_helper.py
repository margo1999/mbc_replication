"""TODO
"""
import os
from pickle import load, dump

import matplotlib.pyplot as plt
import nest
import numpy as np


# TODO: display important simulation parameters like number of excitatory and inhibitory neurons, cluster size, simulation time,...
def plot_spikes(ax=None, exh_spikes=None, inh_spikes=None, gen_spikes=None, R_spikes=None, S_spikes=None, H_spikes=None):
    """[summary]

    Args:
        ax (AxisSubplot, optional): [description]. Defaults to None.
        exh_spikes ([type], optional): [description]. Defaults to None.
        inh_spikes ([type], optional): [description]. Defaults to None.
        gen_spikes ([type], optional): [description]. Defaults to None.
        R_spikes ([type], optional): [description]. Defaults to None.
        S_spikes ([type], optional): [description]. Defaults to None.
        H_spikes ([type], optional): [description]. Defaults to None.
    """

    if ax is None:
        ax = plt.gca()

    with plt.rc_context({

                        # plot settings
                        'font.size': 8,
                        'legend.fontsize': 6,
                        'font.family': 'sans-serif',
                        'text.usetex': False
                        }):

        if exh_spikes is not None and exh_spikes[0].size != 0:
            exh_id, exh_time = zip(*exh_spikes)
            ax.plot(exh_time, exh_id, ls="", marker='.', ms=1, label='exh_spikes')
        if inh_spikes is not None and inh_spikes[0].size != 0:
            inh_id, inh_time = zip(*inh_spikes)
            ax.plot(inh_time, inh_id, ls="", marker='.', ms=1, label='inh_spikes')
        if gen_spikes is not None and gen_spikes[0].size != 0:
            gen_id, gen_time = zip(*gen_spikes)
            ax.plot(gen_time, gen_id, ls="", marker='.', ms=1, label='gen_spikes')
        if R_spikes is not None and R_spikes[0].size != 0:
            R_id, R_time = zip(*R_spikes)
            ax.plot(R_time, R_id, ls="", marker='.', ms=1, label='R_spikes')
        if S_spikes is not None and S_spikes[0].size != 0:
            S_id, S_time = zip(*S_spikes)
            ax.plot(S_time, S_id, ls="", marker='.', ms=1, label='S_spikes')
        if H_spikes is not None and H_spikes[0].size != 0:
            H_id, H_time = zip(*H_spikes)
            ax.plot(H_time, H_id, ls="", marker='.', ms=1, label='H_spikes')

        ax.legend()
        ax.set_title('Spikes during simulation', fontsize='xx-large')
        ax.set_xlabel('time [ms]')
        ax.set_ylabel('neuron [id]')


def load_spikes(filepath=None, filename=None):
    
    assert filename is not None, 'Need filename.'

    # load spikes from reference data
    spikes = load(open(os.path.join(filepath, filename), "rb"))

    exh_spikes_array = np.transpose([spikes['sr_senders_e'], spikes['sr_times_e']])
    inh_spikes_array = np.transpose([spikes['sr_senders_i'], spikes['sr_times_i']])
    readout_spikes_array = np.transpose([spikes['sr_senders_r'], spikes['sr_times_r']])

    return exh_spikes_array, inh_spikes_array, readout_spikes_array


def plot_spectrum(ax=None, connections=None, title='spectrum'):
    if ax is None:
        ax = plt.gca()

    if connections is not None:
        weight_matrix = matrix_from_connections(connections)
        eigenvals = np.linalg.eigvals(weight_matrix.T)

        ax.set_title(title, fontsize='xx-large')
        ax.set_xlabel('Real Axis')
        ax.set_ylabel('Imaginary Axis')
        # ax.plot(np.real(eigenvals), np.imag(eigenvals), ls="", marker='.', ms=1)
        ax.scatter(np.real(eigenvals), np.imag(eigenvals), marker="o", linewidths=0)


def plot_weight_distribution(ax=None, connections=None, params=None, title='weight distribution'):
    if ax is None:
        ax = plt.gca()

    if connections is not None and params is not None:
        weight_list = []
        for post, pre, weight in connections:
            if post <= params['num_exc_neurons'] and weight != 0:
                weight_list.append(weight)

        ax.set_title(title, fontsize='xx-large')
        ax.set_xlabel('weights')
        plt.hist(weight_list, bins=100)


def plot_weight_matrix(ax=None, connections=None, title='', cmap='binary'):
    """[summary]

    Args:
        ax (AxisSubplot, optional): [description]. Defaults to None.
        connections (ndarray, optional): [description]. Defaults to None.
    """

    if ax is None:
        ax = plt.gca()

    if connections is not None:
        weight_matrix = matrix_from_connections(connections)

        ax.set_title(title, fontsize='xx-large')
        ax.set_xlabel('presynaptic neuron [id]')
        ax.set_ylabel('postsynaptic neuron [id]')
        mappable = ax.matshow(weight_matrix, cmap=cmap, aspect='auto')
        plt.colorbar(mappable, ax=ax)


def plot_diff_weight_matrix(ax=None, connections_new=None, connections_old=None, title=''):
    """[summary]

    Args:
        ax (AxisSubplot, optional): [description]. Defaults to None.
        connections (ndarray, optional): [description]. Defaults to None.
    """

    if ax is None:
        ax = plt.gca()

    if connections_new is None or connections_old is None:
        raise Exception("Both connections needed.")

    weight_matrix_new = matrix_from_connections(connections_new)
    weight_matrix_old = matrix_from_connections(connections_old)
    weight_matrix = weight_matrix_new - weight_matrix_old

    # Plots inh and/or exc pool-matrices for better debugging
    inhibitory_plasticity = False  # TODO make this general
    excitatory_plasticity = True  # TODO make this general

    if excitatory_plasticity:
        poollayers = (80, 80)  # TODO make this general
        poolsize = (30, 30)  # TODO make this general
        cluster_max_change = np.zeros(poollayers)
        cluster_min_change = np.zeros(poollayers)
        cluster_mean_change = np.zeros(poollayers)
        for j in range(poollayers[0]):
            j_start = j * poolsize[0]
            j_stop = j_start + poolsize[0]

            for k in range(poollayers[0]):
                k_start = k * poolsize[0]
                k_stop = k_start + poolsize[0]

                cluster_max_change[j, k] = weight_matrix[j_start:j_stop, k_start:k_stop].max()
                cluster_min_change[j, k] = weight_matrix[j_start:j_stop, k_start:k_stop].min()
                cluster_mean_change[j, k] = weight_matrix[j_start:j_stop, k_start:k_stop].mean()

        np.set_printoptions(linewidth=300, suppress=True, precision=15)
        print('Max-pool:')
        print(cluster_max_change.T)
        print('Min-pool:')
        print(cluster_min_change.T)
        print('Mean-pool:')
        print(cluster_mean_change.T)

    if inhibitory_plasticity:
        print('inh Max-pool:')
        print(weight_matrix.max())
        print('inh Min-pool:')
        print(weight_matrix.min())
        print('inh Mean-pool:')
        print(weight_matrix.mean())

    ax.set_title(title, fontsize='xx-large')
    ax.set_xlabel('presynaptic neuron [id]')
    ax.set_ylabel('postsynaptic neuron [id]')
    mappable = ax.imshow(weight_matrix)
    plt.colorbar(mappable, ax=ax)


def matrix_from_connections(connections):
    connections = np.asarray(connections)
    max_neuron_source = int(connections[:, 1].max())
    min_neuron_source = int(connections[:, 1].min())
    max_neuron_target = int(connections[:, 0].max())
    min_neuron_target = int(connections[:, 0].min())
    weight_matrix = np.zeros((max_neuron_target - min_neuron_target + 1, max_neuron_source - min_neuron_source + 1))

    for post, pre, weight in connections:
        #               y           x
        weight_matrix[int(post) - min_neuron_target, int(pre) - min_neuron_source] = weight

    return weight_matrix


def plot_behaviour_of_inh_neuron(multimeter, data_path, params):  # TODO: STOP RECORDING AFTER ONE ROUND
    events = nest.GetStatus(multimeter)[0]['events']
    times = events['times']
    neuronids = events['senders']
    neuronid = neuronids[0]
    assert np.all(neuronids == neuronid)  # Multimeter should only be conntected to one neuron
    neuronid -= params['num_exc_neurons']
    assert 0 < neuronid and neuronid <= params['num_inh_neurons']  # Should be have an index that fits to an inhibitory neuron

    plt.rcParams['axes.grid'] = True
    fig, axis = plt.subplots(1, 2)
    fig.suptitle('neuron ' + str(neuronid) + ' /clock_network')

    axis[0].plot(times, events['V_m'], label='V_m')
    axis[0].set_title('Membrane Potential')
    axis[0].set_xlabel('time [ms]')
    axis[0].set_ylabel('[mV]')
    axis[0].legend()

    axis[1].plot(times, events['g_ex__X__spikeExc'], label='g_ex')
    axis[1].plot(times, events['g_in__X__spikeInh'], label='g_in')
    axis[1].set_title('Conductances')
    axis[1].set_xlabel('time [ms]')
    axis[1].set_ylabel('[nS]')
    axis[1].legend()

    fig.set_size_inches(17, 9)
    plt.tight_layout()

    filename = os.path.join(data_path, f"SingleInhNeuronBehaviour_{neuronid}")
    dump(fig, open(filename, "wb"))
    fig.savefig(filename + '.png')
    plt.close(fig)


def plot_behaviour_of_exc_neuron(multimeter, data_path, params):  # TODO: STOP RECORDING AFTER ONE ROUND

    events = nest.GetStatus(multimeter)[0]['events']
    times = events['times']
    neuronids = events['senders']
    neuronid = neuronids[0]
    assert np.all(neuronids == neuronid)  # Multimeter should only be conntected to one neuron
    assert 0 < neuronid and neuronid <= params['num_exc_neurons']  # Should be have an index that fits to an excitatory neuron

    plt.rcParams['axes.grid'] = True
    fig, axis = plt.subplots(2, 2)
    fig.suptitle('neuron ' + str(neuronid) + ' /clock_network')

    axis[0, 0].plot(times, events['V_m'], label='V_m')
    axis[0, 0].plot(times, events['V_th'], label='V_th')
    axis[0, 0].set_title('Membrane Potential & Adaptive Threshold')
    axis[0, 0].set_xlabel('time [ms]')
    axis[0, 0].set_ylabel('[mV]')
    axis[0, 0].legend()

    axis[0, 1].plot(times, events['g_ex'], label='g_ex')
    axis[0, 1].plot(times, events['g_in'], label='g_in')
    axis[0, 1].set_title('Conductances')
    axis[0, 1].set_xlabel('time [ms]')
    axis[0, 1].set_ylabel('[nS]')
    axis[0, 1].legend()

    axis[1, 0].plot(times, events['w'], label='w')
    axis[1, 0].set_title('Adaption Current')
    axis[1, 0].set_xlabel('time [ms]')
    axis[1, 0].set_ylabel('[pA]')
    axis[1, 0].legend()

    axis[1, 1].plot(times, events['u_bar_minus'], label='u_bar_minus')
    axis[1, 1].plot(times, events['u_bar_plus'], label='u_bar_plus')
    axis[1, 1].set_title('Low-pass Filtered Membrane Potential')
    axis[1, 1].set_xlabel('time [ms]')
    axis[1, 1].set_ylabel('[mV]')
    axis[1, 1].legend()

    fig.set_size_inches(17, 9)
    plt.tight_layout()

    filename = os.path.join(data_path, f"SingleExcNeuronBehaviour_{neuronid}")
    dump(fig, open(filename, "wb"))
    fig.savefig(filename + '.png')
    plt.close(fig)


def plot_behaviour_of_exc_connection(weight_recorder, data_path):
    events = nest.GetStatus(weight_recorder, "events")[0]
    print(f"{events['times']=}")
    print(f"{events=}")
    plt.rcParams['axes.grid'] = True
    fig, axis = plt.subplots(1, 2)
    fig.suptitle('connection ' + str(events['senders'][0]) + '->' + str(events['targets'][0]) + ' /clock_network')

    axis[0].plot(events['times'], events['weights'], marker='+', label='weight')
    axis[0].set_title('Weight Change')
    axis[0].set_xlabel('time [ms]')
    axis[0].set_ylabel('[pF]')
    axis[0].legend()

    axis[1].plot(events['times'], events['x_bar'], marker='+', label='x_bar')
    axis[1].set_title('Low-pass Filtered Spike Train')
    axis[1].set_xlabel('time [ms]')
    axis[1].set_ylabel('[1/s]')
    axis[1].legend()

    fig.set_size_inches(13, 5)
    plt.tight_layout()

    filename = os.path.join(data_path, "SingleSynapseBehaviour")
    dump(fig, open(filename, "wb"))
    fig.savefig(filename + '.png')
    plt.close(fig)


if __name__ == '__main__':
    import numpy as np

    with plt.rc_context({

                        # plot settings
                        'font.size': 8,
                        'legend.fontsize': 6,
                        'figure.figsize': (10, 5),
                        'font.family': 'sans-serif',
                        'text.usetex': False
                        }):

        spikes = np.random.randint(0, 10, size=((20, 2)))
        connections = [[1., 3., 0.5], [3., 6., 1.5], [6., 9., 2.5]]  # TODO
        new_connections = [[1., 3., -0.5], [3., 6., -1.5], [6., 9., -2.5]]  # TODO

        figure, axes = plt.subplots(1, 3)
        plot_spikes(ax=axes[0], exh_spikes=spikes)
        plot_weight_matrix(ax=axes[1], connections=connections, title='old connection matrix')
        plot_weight_matrix(ax=axes[2], connections=new_connections, title='new connection matrix')

        figure.tight_layout()
        plt.show()
