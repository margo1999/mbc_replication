import matplotlib.pyplot as plt
import numpy as np

# TODO: display important simulation parameters like number of excitatory and inhibitory neurons, cluster size, simulation time,...
def plot_spikes(
        ax=None,
        exh_spikes=None, inh_spikes=None, gen_spikes=None, R_spikes=None, S_spikes=None, H_spikes=None):
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

        if exh_spikes is not None and exh_spikes[0].size != 0: # TODO: This is not pretty
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

def plot_weight_matrix(ax=None, connections=None, title=''):
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
        ax.imshow(weight_matrix)

def matrix_from_connections(connections):
    connections = np.asarray(connections)
    max_neuron = int(max(connections[:,0].max(), connections[:,1].max()))
    weight_matrix = np.zeros((max_neuron + 1, max_neuron + 1))

    for post, pre, weight in connections:
        #               y           x
        weight_matrix[int(post), int(pre)] = weight 

    return weight_matrix

if __name__ == '__main__':
    import numpy as np

    with plt.rc_context({

            # plot settings 
            'font.size': 8,
            'legend.fontsize': 6,
            'figure.figsize': (10,5),
            'font.family': 'sans-serif',
            'text.usetex': False
            }):

        spikes = np.random.randint(0, 10, size=((20, 2)))
        connections = [[1., 3., 0.5], [3.,6., 1.5], [6., 9., 2.5]] # TODO
        new_connections = [[1., 3., -0.5], [3.,6., -1.5], [6., 9., -2.5]] # TODO

        figure, axes = plt.subplots(1, 3)
        plot_spikes(ax=axes[0], exh_spikes=spikes)
        plot_weight_matrix(ax=axes[1], connections=connections, title='old connection matrix')
        plot_weight_matrix(ax=axes[2], connections=new_connections, title='new connection matrix')
        
        figure.tight_layout()
        plt.show()