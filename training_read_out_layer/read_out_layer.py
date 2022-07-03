import nest
import sys
import time
import numpy as np
from tqdm import tqdm
import os
import parameterspace_read_out as data_pars_ro
from experiments.sequential_dynamics import parameters_space as data_pars
import matplotlib.pyplot as plt
from pickle import dump
#from figures.plot_results import plot_2_mins_results
#import parameters_space as data_pars

from clock_net import model, helper, plot_helper

def create_read_out_layer_population(*, params_ro, alphabet):
    neuron_num = len(alphabet)
    r_neurons = nest.Create(params_ro['read_out_model'], neuron_num, params_ro['read_out_params'])
    s_neurons = nest.Create(params_ro['supervisor_model'], neuron_num, params_ro['supervisor_params'])
    h_neurons = nest.Create(params_ro['interneuron_model'], neuron_num, params_ro['interneuron_params'])

    return r_neurons, s_neurons, h_neurons

def create_generators_for_spatial_dimension(*, params_ro):
    # Create Generator for H neurons
    gen_h = nest.Create('poisson_generator',  params=dict(rate=params_ro['exh_rate_hx']))
    # Create Generators for S neurons
    gen_s_baseline = nest.Create('poisson_generator',  params=dict(rate=params_ro['baseline_rate_sx']))
    gen_s = nest.Create('poisson_generator',  params=dict(rate=params_ro['exh_rate_sx']))

    return gen_h, gen_s_baseline, gen_s

def connect_RNN_to_read_out_layer(*, model_instance, r_neurons, params_ro):
    nest.Connect(model_instance.exc_neurons, r_neurons, params_ro['conn_dict_re'], params_ro['syn_dict_re'])

def connect_read_out_layer_population(*, r_neurons, s_neurons, h_neurons, params_ro):
    nest.Connect(h_neurons, r_neurons, params_ro['conn_dict_rh'], params_ro['syn_dict_rh'])
    nest.Connect(r_neurons, h_neurons, params_ro['conn_dict_hr'], params_ro['syn_dict_hr'])
    nest.Connect(s_neurons, r_neurons, params_ro['conn_dict_rs'], params_ro['syn_dict_rs'])

def connect_generators_for_spatial_dimension(*, gen_h, gen_s_baseline, gen_s, h_neurons, s_neurons,  params_ro):
    nest.Connect(gen_h, h_neurons, conn_spec=params_ro['conn_dict_hx'] , syn_spec=params_ro['syn_dict_hx'])
    nest.Connect(gen_s_baseline, s_neurons, conn_spec=params_ro['conn_dict_sx'] , syn_spec=params_ro['syn_dict_sx'])
    nest.Connect(gen_s, s_neurons, conn_spec=params_ro['conn_dict_sx'] , syn_spec=params_ro['syn_dict_sx'])
    conn = nest.GetConnections(source=gen_s, target=s_neurons)
    conn.weight = 0.0

def create_spike_recorders_for_detection():
    sr_first = nest.Create("spike_recorder", params={'record_to':'memory', 'stop': 0.0})
    sr_last = nest.Create("spike_recorder", params={'record_to':'memory'})

    return sr_first, sr_last

def connect_spike_recorders_for_detection(*, model_instance, sr_first, sr_last):
    num_exc_neurons = model_instance.num_exc_neurons
    num_exc_clusters = model_instance.num_exc_clusters
    cluster_size = num_exc_neurons // num_exc_clusters
    nest.Connect(model_instance.exc_neurons[:cluster_size], sr_first)
    nest.Connect(model_instance.exc_neurons[(num_exc_neurons - 30):num_exc_neurons], sr_last)

def create_spike_recorders():
    sr_e = nest.Create("spike_recorder", params={'record_to':'memory', 'stop': 0.0})
    sr_i = nest.Create("spike_recorder", params={'record_to':'memory', 'stop': 0.0})
    sr_r = nest.Create("spike_recorder", params={'record_to':'memory', 'stop': 0.0})

    return sr_e, sr_i, sr_r

def connect_spike_recorders(*, model_instance, r_neurons, sr_e, sr_i, sr_r):
    nest.Connect(model_instance.exc_neurons, sr_e)
    nest.Connect(model_instance.inh_neurons, sr_i)
    nest.Connect(r_neurons, sr_r)

def save_spikes_after_sim(sr_e, sr_i, sr_r):
    sr_times_e =  sr_e.events['times']
    sr_senders_e = sr_e.events['senders']

    sr_times_i =  sr_i.events['times']
    sr_senders_i = sr_i.events['senders']

    sr_times_r =  sr_r.events['times']
    sr_senders_r = sr_r.events['senders']

    spikes = dict(sr_times_e=sr_times_e, sr_senders_e=sr_senders_e, sr_times_i=sr_times_i, sr_senders_i=sr_senders_i, sr_times_r=sr_times_r, sr_senders_r=sr_senders_r)
    spikefilepath = os.path.join('/Users/Jette/Desktop/results/NEST/job_3822329/1787e7674087ddd1d5749039f947a2cd/', f"spikes_after_learning.pickle")
    dump(spikes, open(spikefilepath, "wb"))

def simulate_spatial_dimension_training(*, model_instance, params, sim_time, sequence, r_neurons, s_neurons, gen_s_baseline, gen_s, sr_first, sr_last):
    training_sequence = list(sequence)
    alphabet = sorted(set(sequence))
    assert len(s_neurons) == len(alphabet)

    read_out_dict = {}
    for idx, item in enumerate(alphabet):
         read_out_dict[item] = s_neurons[idx]

    # TODO: put this in parameterspace_read_out
    stimulationtime = 75
    leadtime = 50
    trainingtime = stimulationtime * len(training_sequence)
    lagtime = params['round_time'] - trainingtime
    beginning = False

    recording_time = 1500
    sr_e, sr_i, sr_r = create_spike_recorders()
    connect_spike_recorders(model_instance=model_instance, r_neurons=r_neurons, sr_e=sr_e, sr_i=sr_i, sr_r=sr_r)

    # Simulate
    print(f"{nest.biological_time=}")

    nest.Simulate(leadtime)

    while (nest.biological_time < sim_time):
        beginning = False
        sr_last.stop = 1000000.0
        while(not beginning):
            start = nest.biological_time
            nest.Simulate(5.0)
            if sr_last.n_events > 5:
                sr_last.stop = nest.biological_time
                sr_first.stop = 1000000.0
                print("inner-loop")
                while(not beginning):
                    nest.Simulate(2.0)
                    if sr_first.n_events > 5:
                        beginning = True
                    sr_first.n_events = 0
                    assert nest.biological_time <= start + 700.0
            sr_last.n_events = 0
            assert nest.biological_time <= start + 700.0

        sr_last.stop = nest.biological_time
        sr_first.stop = nest.biological_time

        nest.Simulate(30)
        nest.Prepare()
        for item in tqdm(training_sequence):
            conn = nest.GetConnections(target=read_out_dict[item])
            conn.weight = list(reversed(conn.weight))
            nest.Run(stimulationtime)
            conn.weight = list(reversed(conn.weight))
        nest.Cleanup()

    sr_r.origin = nest.biological_time
    sr_r.stop = recording_time
    sr_e.origin = nest.biological_time
    sr_e.stop = recording_time
    sr_i.origin = nest.biological_time
    sr_i.stop = recording_time
    nest.Simulate(recording_time) 
    save_spikes_after_sim(sr_e=sr_e, sr_i=sr_i, sr_r=sr_r) 

if __name__ == "__main__": 
 
    sim_time = 12000 # ms
    recording_time = 1500

    # Get parameters 
    PS = data_pars.p
    PSRO = data_pars_ro.p

    # parameter-set id from command line (submission script)
    params = helper.parameter_set_list(PS)[0]
    params_ro = helper.parameter_set_list(PSRO)[0]
    resultpath = helper.get_data_path(params['data_path'], params['label'])


    # ###############################################################
    # specify sequences
    # ===============================================================
    sequences, _, vocabulary = helper.generate_sequences(params['task'], params['data_path'], params['label'])
    sequence = 'ABCBA'
    alphabet = set(sequence)

    # ###############################################################
    # create network
    # ===============================================================
    model_instance = model.Model(params, sequences, vocabulary)
    model_instance._Model__create_RNN_populations()
    model_instance._Model__load_connections(label='all_connections')

    # # Plot loaded weights
    # conns = nest.GetConnections()
    # conns = nest.GetStatus(conns, ['target', 'source', 'weight'])
    # plot_helper.plot_weight_matrix(ax=None, connections=conns, title='initial trained weight matrix')
    # plt.show()

    # Create neurons and connections to get the final model architecture
    r_neurons, s_neurons, h_neurons = create_read_out_layer_population(params_ro=params_ro, alphabet=alphabet)
    connect_RNN_to_read_out_layer(model_instance=model_instance, r_neurons=r_neurons, params_ro=params_ro)
    connect_read_out_layer_population(r_neurons=r_neurons, s_neurons=s_neurons, h_neurons=h_neurons, params_ro=params_ro)

    assert nest.network_size == (model_instance.num_exc_neurons + model_instance.num_inh_neurons + 3 * len(alphabet)) 

    # # Do spontaneous dynamics
    # model_instance._Model__create_recording_devices()
    # model_instance._Model__connect_neurons_to_spike_recorders()
    # initial_weight_inputs_dict = model_instance.get_initial_weight_sums_dict_opt(neurons=model_instance.exc_neurons, synapse_type='static_synapse')
    # sr_times_exh, sr_senders_exh, sr_times_inh, sr_senders_inh = model_instance.record_exc_spike_behaviour(2000.0, 4000.0, initial_weight_inputs_dict)
    # spikes = dict(sr_times_exh=sr_times_exh, sr_senders_exh=sr_senders_exh, sr_times_inh=sr_times_inh, sr_senders_inh=sr_senders_inh)
    # spikefilepath = os.path.join('/Users/Jette/Desktop/results/NEST/job_3822329/1787e7674087ddd1d5749039f947a2cd/', f"spikes_stable.pickle")
    # dump(spikes, open(spikefilepath, "wb"))

    # exit()

    # Create Poisson Generators and connect them to achive spontaneous dynamics in RNN
    model_instance.set_up_spontaneous_dynamics(sim_time + recording_time)

    # Create Poisson Generators for S and H neurons
    gen_h, gen_s_baseline, gen_s = create_generators_for_spatial_dimension(params_ro=params_ro)
    connect_generators_for_spatial_dimension(gen_h=gen_h, gen_s_baseline=gen_s_baseline, gen_s=gen_s, h_neurons=h_neurons, s_neurons=s_neurons,  params_ro=params_ro)

    print(f"{nest.network_size=}")
    print(nest.GetConnections(target=s_neurons))
    assert nest.network_size == (model_instance.num_exc_neurons + model_instance.num_inh_neurons + 3 * (1 + len(alphabet)) + 2)

    # Create and connect spike recorders to identify beginning of sequential dynamics
    sr_first, sr_last = create_spike_recorders_for_detection()
    connect_spike_recorders_for_detection(model_instance=model_instance, sr_first=sr_first, sr_last=sr_last)

    # Simulate training
    simulate_spatial_dimension_training(model_instance=model_instance, params=params, sim_time=sim_time, sequence=sequence, r_neurons=r_neurons, s_neurons=s_neurons, gen_s_baseline=gen_s_baseline, gen_s=gen_s, sr_first=sr_first, sr_last=sr_last)

    # results
    conns = nest.GetConnections(source=model_instance.exc_neurons , target=r_neurons)
    conns = nest.GetStatus(conns, ['target', 'source', 'weight'])
    np.save(os.path.join('/Users/Jette/Desktop/results/NEST/job_3822329/1787e7674087ddd1d5749039f947a2cd/', 'readout_weights'), conns)
    # for connection in conns:
    #     if conns[2] >= 0.1:
    #         print(connection)
    #plt.axis('scaled')
    plt.rcParams["figure.figsize"] = (10,10)
    plot_helper.plot_weight_matrix(ax=None, connections=conns, title='trained read-out synapses', cmap='viridis')
    plt.show()

