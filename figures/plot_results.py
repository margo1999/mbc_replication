""" TODO
"""
from pickle import dump

import matplotlib.pyplot as plt

import figures.plot_spectrum as plot_spectrum
import figures.plot_spikes as plot_spikes
import figures.plot_weight_distribution as plot_weight_distribution
import figures.plot_weight_matrix as plot_weight_matrix


def plot_results(outfilename: str = None):

    with plt.rc_context({
                        # plot settings
                        'font.size': 8,
                        'legend.fontsize': 6,
                        'figure.figsize': (10, 5),
                        'font.family': 'sans-serif',
                        'text.usetex': False
                        }):
        figure, axes = plt.subplots(1, 4)
        plot_spikes.plot_spikes(axes[0])
        plot_weight_matrix.plot_weight_matrices(axes=[axes[1], axes[2], axes[3]])
        figure.tight_layout()

        if outfilename:
            dump(figure, open(outfilename, "wb"))
            figure.savefig(outfilename + '.png')


def plot_2_mins_results(spikefilename, connectionsfilename, allconnectionsfilename, params=None, outfilename=None):
    with plt.rc_context({
                        # plot settings
                        'font.size': 8,
                        'legend.fontsize': 6,
                        'figure.figsize': (10, 5),
                        'font.family': 'sans-serif',
                        'text.usetex': False
                        }):
        figure, axes = plt.subplots(2, 2)
        plot_spikes.plot_2_mins_spikes(axes[0, 0], spikefilename)
        plot_weight_matrix.plot_2_mins_weight_matrix(axes[0, 1], connectionsfilename)
        plot_spectrum.plot_2_mins_spectrum(axes[1, 0], allconnectionsfilename)
        plot_weight_distribution.plot_2_mins_weight_distribution(axes[1, 1], params=params, filename=allconnectionsfilename)
        figure.tight_layout()

        if outfilename:
            dump(figure, open(outfilename + '.pickle', "wb"))
            figure.savefig(outfilename + '.png')
        plt.close(fig=figure)


if __name__ == '__main__':
    plot_results()
    plt.show()
