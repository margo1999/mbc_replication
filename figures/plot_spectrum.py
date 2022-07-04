""" TODO
"""
import numpy as np
import clock_net.plot_helper as plot_helper


# TODO: Plot in the end of simulation is missing

def plot_2_mins_spectrum(ax=None, filename: str = None):
    assert ax is not None, 'Need axes object.'
    assert filename is not None, 'Need filename.'

    allconnections = np.load(filename)
    plot_helper.plot_spectrum(ax=ax, connections=allconnections, title='spectrum')
