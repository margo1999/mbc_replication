import os
import sys 
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import defaultdict
import clock_net.plot_helper as plot_helper
from clock_net.helper import load_data, load_spike_data
from clock_net import helper as helper
from pickle import load

# TODO: Plot in the end of simulation is missing 

def plot_2_mins_weight_distribution(ax=None, params=None, filename=None):
    assert ax is not None, 'Need axes object.'
    assert filename is not None, 'Need filename.'

    allconnections = np.load(filename)
    plot_helper.plot_weight_distribution(ax=ax, connections=allconnections, params=params, title='weight distribution')