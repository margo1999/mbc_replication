""" TODO
"""

import numpy as np
from mbc_network.helper import plot_helper

# TODO: Plot in the end of simulation is missing


def plot_2_mins_weight_distribution(ax=None, params=None, filename=None):
    assert ax is not None, 'Need axes object.'
    assert filename is not None, 'Need filename.'

    allconnections = np.load(filename)
    plot_helper.plot_weight_distribution(ax=ax, connections=allconnections, params=params, title='weight distribution')
