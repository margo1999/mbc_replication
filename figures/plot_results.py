import figures.plot_spikes as plot_spikes
import figures.plot_weight_matrix as plot_weight_matrix

import matplotlib.pyplot as plt

def plot_results():

    with plt.rc_context({

            # plot settings 
            'font.size': 8,
            'legend.fontsize': 6,
            'figure.figsize': (10,5),
            'font.family': 'sans-serif',
            'text.usetex': False
            }):
        figure, axes = plt.subplots(1,3)
        plot_spikes.plot_spikes(axes[0])
        plot_weight_matrix.plot_weight_matrices(axes=[axes[1], axes[2]])
        figure.tight_layout()
    

if __name__ == '__main__':
    plot_results()
    plt.show()