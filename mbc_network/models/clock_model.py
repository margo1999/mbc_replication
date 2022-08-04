"""PyNEST Clock Network: Model Class [1]
----------------------------------------

Main file of the Clock Network defining ``Model`` class with function
to build, connect and simulate the network.

Authors
~~~~~~~
Jette Oberlaender, Younes Bouhadjar
"""
import os
from pickle import dump
import random
import nest
import nest.ll_api
import copy
import numpy as np
from collections import defaultdict
import sys
import time
from tqdm import tqdm
from nest import voltage_trace
import matplotlib.pyplot as plt
from parameters import ParameterSpace
from mbc_network.helper import plot_helper
from time import perf_counter
from mbc_network.plotting.plot_results import plot_2_mins_results

from mbc_network.helper import training_helper

TWO_MIN = 2 * 60 * 1000

class Model:
    """Instantiation of the Clock Network model and its PyNEST implementation.

    The model provides the following member functions:

    __init__(parameters)
    create()
    connect()
    simulate(t_sim)

    In addition, each model may implement other model-specific member functions.
    """

    def __init__(self, params: ParameterSpace):
        """Initialize model and simulation instance, including

        1) Parameter setting,
        2) Generate sequence data,
        3) Configuration of the NEST kernel,
        4) Setting random-number generator seed, and

        Parameters
        ----------
        params:     dict
                    Parameter dictionary
        """

        print('\nInitialising model and simulation...')

        # Set parameters derived from base parameters
        self.params = training_helper.derived_parameters(params)

        # Data directory
        self.data_path = training_helper.get_data_path(self.params['data_path'], self.params['label'])
        print(f"Simulationlabel {self.params['label']} is generated based on parameter set...")
        print(f"Generated data will be stored at {self.data_path}...")

        # Set up data directory
        if nest.Rank() == 0:
            if self.data_path.is_dir():
                message = "Directory already existed."
                if self.params['overwrite_files']:
                    message += "Old data will be overwritten."
            else:
                self.data_path.mkdir(parents=True, exist_ok=True)
                message = "Directory has been created."
            print("Data will be written to: {}\n{}\n".format(self.data_path, message))

        # Set network size parameters
        self.num_exc_neurons = params['num_exc_neurons']
        self.num_inh_neurons = params['num_inh_neurons']
        self.num_exc_clusters = params['num_exc_clusters']


        if self.params['read_out_off'] is False:  # TODO need better solution for this
            self.random_dynamics_ex = None
            self.random_dynamics_ix = None

        # Initialize RNG
        np.random.seed(self.params['seed'])  # TODO do I need this for the sequence_model as well?
        random.seed(self.params['seed'])

        # initialize the NEST kernel
        self.__setup_nest()

    def __setup_nest(self):  # TODO do I need this for the sequence_model as well?
        """Initializes the NEST kernel.
        """

        nest.Install('nestmlmodule')
        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")
        nest.SetKernelStatus({
            'resolution': self.params['dt'],
            'print_time': self.params['print_simulation_progress'],
            'local_num_threads': self.params['n_threads'],
            'rng_seed': self.params['seed'],
            'dict_miss_is_error': True,
            'overwrite_files': self.params['overwrite_files'],
            'data_path': str(self.data_path),
            'data_prefix': ''
        })

    def create(self):
        """Create and configure all network nodes (neurons + recording and stimulus devices)
        """

        print('\nCreating and configuring nodes...')

        # Create excitatory and inhibitory population of RNN
        self.create_rnn_populations()

        # Create recording devices
        self.__create_recording_devices()

        # Create spike generators
        self.__create_spike_generators()

        # Create spike generators and connections for spontaneous dynamics
        self.create_spontaneous_dynamics_nodes()

        # Set up recording node and establish connection
        self.record_behaviour_of_exc_neuron(neuronid=10)
        self.record_behaviour_of_inh_neuron(neuronid=5)

    def connect(self):
        """Connects network and devices
        """

        print('\nConnecting network and devices...')

        # Connect excitatory population (EE) TODO: also EI connections are plastic
        self.__connect_excitatory_neurons()

        # Connect inhibitory population (II, EI, IE)
        self.__connect_RNN_neurons()

        # Connect external input
        self.__connect_external_inputs_to_clusters()

        # Connect neurons to the spike recorder
        self.__connect_neurons_to_spike_recorders()

        # self.record_behaviour_of_exc_connection()
        print('\nAll nodes connected...')
        print(f"Number of EE connections: {len(nest.GetConnections(source=self.exc_neurons, target=self.exc_neurons))=}")
        print(f"Number of IE connections: {len(nest.GetConnections(source=self.exc_neurons, target=self.inh_neurons))=}")
        print(f"Number of EI connections: {len(nest.GetConnections(source=self.inh_neurons, target=self.exc_neurons))=}")
        print(f"Number of II connections: {len(nest.GetConnections(source=self.inh_neurons, target=self.inh_neurons))=}")
        print(f"Number of IX connections: {len(nest.GetConnections(source=self.external_node_to_inh_neuron, target=self.inh_neurons))=}")

# TODO: rename to simulate_RNN or train_RNN?
    def simulate(self):
        """Run simulation by stopping after each round to reset the times of the spike generators.
        """
        test_normalization = True
        opt = True
        if test_normalization:
            print("Normalization")

            print(" Create Dict")
            # Takes 20-25 seconds (+ saving of source and target ~30s) compared to get_initial_weight_sums_dict_opt where computation takes 70-80 seconds
            start = time.perf_counter()
            initial_weight_inputs_dict = self.get_initial_weight_sums_dict_opt(neurons=self.exc_neurons, synapse_type='clopath_synapse')
            stop = time.perf_counter()
            print(f"{stop-start=}")

            if opt:
                self.save_connections(synapse_model='clopath_synapse', fname='opt_normalization_before')
                nest.Simulate(450.0)
                print("Start Counter")
                start = time.perf_counter()
                self.normalize_weights_opt(self.exc_neurons, initial_weight_inputs_dict)
                stop = time.perf_counter()
                print(f"{stop-start=}")
                self.save_connections(synapse_model='clopath_synapse', fname='opt_normalization_after')
            else:
                self.save_connections(synapse_model='clopath_synapse', fname='normalization_before')
                nest.Simulate(450.0)
                print("Start Counter")
                start = time.perf_counter()
                self.normalize_weights(self.exc_neurons, initial_weight_inputs_dict)
                stop = time.perf_counter()
                print(f"{stop-start=}")
                self.save_connections(synapse_model='clopath_synapse', fname='normalization_after')
        else:
            round_duration = self.params['round_time']
            random_dynamics_time = self.params['random_dynamics_time']
            normalization_time = self.params['normalization_time']
            initial_weight_inputs_dict = self.get_initial_weight_sums_dict_opt(neurons=self.exc_neurons, synapse_type='clopath_synapse')

            # Rank() returns the MPI rank of the local process TODO: simulating time not correct
            if nest.Rank() == 0:
                print('\nSimulating {} hour.'.format(1))

            self.simulate_sequential_input(round_duration, normalization_time, initial_weight_inputs_dict)

            seq_input_iters = self.params['training_iterations']
            if self.params['random_dynamics']:
                iterations = int(np.ceil(random_dynamics_time / 120000.0))
                for i in range(iterations):
                    self.simulate_random_dynamics(120000.0, normalization_time, initial_weight_inputs_dict)
                    sr_times_exh, sr_senders_exh = self.record_exc_spike_behaviour(self.params['plotting_time'], normalization_time, initial_weight_inputs_dict)

                    file_name = f"ee_connections_{seq_input_iters+i}.npy"
                    self.save_connections(synapse_model=self.params['syn_dict_ee']['synapse_model'], fname=file_name)
                    connectionsfilepath = os.path.join(self.data_path, file_name)

                    file_name = f"all_connections_{seq_input_iters+i}.npy"
                    self.save_connections(fname=file_name)
                    allconnectionsfilepath = os.path.join(self.data_path, file_name)

                    spikes = dict(sr_times_exh=sr_times_exh, sr_senders_exh=sr_senders_exh)
                    spikefilepath = os.path.join(self.data_path, f"spikes_{seq_input_iters + i}.pickle")
                    dump(spikes, open(spikefilepath, "wb"))

                    # Plot and save plot of connection and spike behaviour as png and pickle file
                    plotsfilepath = os.path.join(self.data_path, f"plots_{seq_input_iters + i}")
                    plot_2_mins_results(spikefilepath, connectionsfilepath, allconnectionsfilepath, params=self.params, outfilename=plotsfilepath)

    def simulate_sequential_input(self, round_duration: int, normalization_time: int, initial_weight_inputs_dict: dict):
        """_summary_

        Args:
            round_duration (int): _description_
            normalization_time (int): _description_
            initial_weight_inputs_dict (dict): _description_
        """ 

        # the training is divided into training sessions where one training iteration lasts for 2 minutes, as in the original implementation
        training_iterations = self.params['training_iterations']
        rounds = int(np.ceil(TWO_MIN / round_duration))
        plotting_time = self.params['plotting_time']

        for two_min_unit in tqdm(range(training_iterations)):

            for round_ in tqdm(range(rounds)):

                # Checking if the current time is correct with the estimated time
                esttime = ((round_ + (rounds * two_min_unit)) * (round_duration)) + two_min_unit * plotting_time # TODO make this more clear
                curtime = nest.biological_time
                assert esttime == curtime

                simulate_steps = int(round_duration // normalization_time)

                nest.Prepare()
                for normalization_unit in range(simulate_steps):
                    nest.Run(normalization_time)
                    self.normalize_weights_opt(self.exc_neurons, initial_weight_inputs_dict)

                if round_duration % normalization_time != 0:
                    remaining_time = round_duration - normalization_time * simulate_steps
                    nest.Run(remaining_time)
                    assert remaining_time < normalization_time
                nest.Cleanup()

                # Turn off all spike recorders and set spike recorder for exc neurons to 'memory' to record spikes more flexible
                if two_min_unit + round_ == 0:
                    self.spike_recorder_exc.record_to = "memory"  # TODO: Weird behaviour! No spikes are stored if I don't set first record_to and then stop
                    self.spike_recorder_inh.record_to = "memory"
                    self.spike_recorder_generator.record_to = "memory"
                    self.spike_recorder_exc.stop = nest.biological_time
                    self.spike_recorder_inh.stop = nest.biological_time
                    self.spike_recorder_generator.stop = nest.biological_time

                    # plot_helper.plot_behaviour_of_exc_connection(self.wr, self.data_path)
                    plot_helper.plot_behaviour_of_exc_neuron(self.mm_exc, self.data_path, self.params)
                    plot_helper.plot_behaviour_of_inh_neuron(self.mm_inh, self.data_path, self.params)

                # Simulation is stopped to set a new reference time (origin) for start and stop of the generators, otherwise they would only spike at the beginning
                if (round_ == (rounds - 1)) and two_min_unit < (training_iterations - 1):
                    for generators_to_exc in self.external_node_to_exc_neuron_dict.values():
                        generators_to_exc[0].origin += round_duration + plotting_time
                        generators_to_exc[1].origin += round_duration + plotting_time
                        generators_to_exc[2].origin += round_duration + plotting_time
                        generators_to_exc[3].origin += round_duration + plotting_time

                    self.external_node_to_inh_neuron.origin += round_duration + plotting_time

                elif round_ < (rounds - 1):
                    for generators_to_exc in self.external_node_to_exc_neuron_dict.values():
                        generators_to_exc[0].origin += round_duration
                        generators_to_exc[1].origin += round_duration
                        generators_to_exc[2].origin += round_duration
                        generators_to_exc[3].origin += round_duration

                    self.external_node_to_inh_neuron.origin += round_duration

            # Save ee connections after two minutes
            file_name = f"ee_connections_{two_min_unit}.npy"
            self.save_connections(synapse_model=self.params['syn_dict_ee']['synapse_model'], fname=file_name)
            connectionsfilepath = os.path.join(self.data_path, file_name)

            # Save all connections after two minutes to plot spectrum
            file_name = f"all_connections_{two_min_unit}.npy"
            self.save_connections(fname=file_name)
            allconnectionsfilepath = os.path.join(self.data_path, file_name)

            # Save current spike behaviour under random input dynamics
            sr_times_exh, sr_senders_exh = self.record_exc_spike_behaviour(self.params['plotting_time'], normalization_time, initial_weight_inputs_dict)
            spikes = dict(sr_times_exh=sr_times_exh, sr_senders_exh=sr_senders_exh)
            spikefilepath = os.path.join(self.data_path, f"spikes_{two_min_unit}.pickle")
            dump(spikes, open(spikefilepath, "wb"))

            # Plot and save plot of excitatory connections, spike behaviour and spectrum as png and pickle file
            plotsfilepath = os.path.join(self.data_path, f"plots_{two_min_unit}")
            plot_2_mins_results(spikefilepath, connectionsfilepath, allconnectionsfilepath, params=self.params, outfilename=plotsfilepath)

    def simulate_random_dynamics(self, sim_time: int, normalization_time: int, initial_weight_inputs: dict):
        """_summary_

        Args:
            sim_time (int): _description_
            normalization_time (int): _description_
            initial_weight_inputs (dict): _description_
        """
        # TODO save connections, spike rasters and spectra every 2 minutes

        if self.random_dynamics_ex is None or self.random_dynamics_ix is None:
            self.set_up_spontaneous_dynamics(sim_time)
        else:
            self.random_dynamics_ex.origin = nest.biological_time
            self.random_dynamics_ix.origin = nest.biological_time
            self.random_dynamics_ex.stop = sim_time
            self.random_dynamics_ix.stop = sim_time
            self.random_dynamics_ex.start = 0.0
            self.random_dynamics_ix.start = 0.0

        simulate_steps = int(sim_time // normalization_time)

        nest.Prepare()

        for _ in range(simulate_steps):
            nest.Run(normalization_time)
            self.normalize_weights_opt(self.exc_neurons, initial_weight_inputs)

        if sim_time % normalization_time != 0:
            remaining_time = sim_time - normalization_time * simulate_steps
            nest.Run(remaining_time)
            assert remaining_time < normalization_time

        nest.Cleanup()

        # print(f"{self.random_dynamics_ex.stop=}", f"{nest.biological_time=}")
        assert (self.random_dynamics_ex.origin + sim_time) == nest.biological_time
        assert (self.random_dynamics_ix.origin + sim_time) == nest.biological_time

    def record_behaviour_of_exc_connection(self):  # TODO: only for 1 round
        """_summary_
        """
        conn = nest.GetConnections(source=self.exc_neurons[31 - 1], target=self.exc_neurons[30:60], synapse_model='clopath_synapse')[2]
        print(f"{conn.target=}")
        sourceid = conn.source
        targetid = conn.target
        self.wr = nest.Create('weight_recorder')
        syn_dict_ee = self.params['syn_dict_ee'].copy()
        syn_dict_ee['weight_recorder'] = self.wr
        del(syn_dict_ee['synapse_model'])
        nest.CopyModel('clopath_synapse', 'clopath_synapse_wr', syn_dict_ee)

        nest.Disconnect(self.exc_neurons[sourceid - 1], self.exc_neurons[targetid - 1], syn_spec={'synapse_model': 'clopath_synapse'})
        nest.Connect(self.exc_neurons[sourceid - 1], self.exc_neurons[targetid - 1], syn_spec={'synapse_model': 'clopath_synapse_wr'})  # TODO: Will be ignored in save_connections

    def record_behaviour_of_inh_neuron(self, neuronid=None):  # TODO: make record duration more general
        """_summary_

        Args:
            neuronid (_type_, optional): _description_. Defaults to None.
        """
        if neuronid is not None:
            self.mm_inh = nest.Create('multimeter', params={'record_from': ['g_ex__X__spikeExc', 'g_in__X__spikeInh', 'V_m'], 'interval': 0.1, 'stop': 120.0})
            nest.Connect(self.mm_inh, self.inh_neurons[neuronid - 1])

    def record_behaviour_of_exc_neuron(self, neuronid=None):  # TODO: make record duration more general
        """_summary_

        Args:
            neuronid (_type_, optional): _description_. Defaults to None.
        """
        if neuronid is not None:
            self.mm_exc = nest.Create('multimeter', params={'record_from': ['g_ex', 'g_in', 'u_bar_bar', 'u_bar_minus', 'u_bar_plus', 'V_m', 'V_th', 'w'], 'interval': 0.1, 'stop': 120.0})
            nest.Connect(self.mm_exc, self.exc_neurons[neuronid - 1])

    def record_exc_spike_behaviour(self, sim_time: int, normalization_time: int, initial_weight_inputs: dict):  # TODO add a return (I have to different types of return..)
        """_summary_

        Args:
            sim_time (int): _description_
            normalization_time (int): _description_
            initial_weight_inputs (dict): _description_

        Returns:
            _type_: _description_
        """

        # TEST
        conn_ee_weights_before, conn_ei_weights_before = self.get_plastic_connections()

        # freeze weights because no learning should happen
        self.freeze_weights()
        # print(f"{self.exc_neurons.A_LTP=}")

        sp_params = {'record_to': 'memory', 'origin': nest.biological_time, 'start': 0.0, 'stop': sim_time}
        nest.SetStatus(self.spike_recorder_exc, params=sp_params)

        if not self.params['read_out_off']:
            nest.SetStatus(self.spike_recorder_inh, params=sp_params)

        # self.simulate_random_dynamics(sim_time, normalization_time, initial_weight_inputs)
        self.simulate_random_dynamics(sim_time, normalization_time, initial_weight_inputs)

        # Save spikes after simulation
        sr_times_exh = self.spike_recorder_exc.events['times']
        sr_senders_exh = self.spike_recorder_exc.events['senders']

        self.spike_recorder_exc.n_events = 0  # reset the spike counts

        if not self.params['read_out_off']:
            sr_times_inh = self.spike_recorder_inh.events['times']
            sr_senders_inh = self.spike_recorder_inh.events['senders']
            self.spike_recorder_inh.n_events = 0  # reset the spike counts

        print(nest.GetStatus(self.spike_recorder_exc))

        # conn_ee_weights_between, conn_ei_weights_between = self.get_plastic_connections()
        # self.simulate_random_dynamics(sim_time, normalization_time, initial_weight_inputs)

        # unfreeze weights
        self.unfreeze_weights()

        # TEST
        conn_ee_weights_after, conn_ei_weights_after = self.get_plastic_connections()
        # if not np.allclose(conn_ee_weights_between, conn_ee_weights_after):
        #     print(f"{max(abs(np.subtract(conn_ee_weights_after, conn_ee_weights_between)))=}\n")
        #     print(f"{len((np.subtract(conn_ee_weights_after, conn_ee_weights_between)))=}\n")
        #     print(f"{np.count_nonzero(np.subtract(conn_ee_weights_after, conn_ee_weights_between))=}\n")

        # if not np.allclose(conn_ee_weights_before, conn_ee_weights_between):
        #     print(f"{np.subtract(conn_ee_weights_between, conn_ee_weights_before)=}\n")

        if not np.allclose(conn_ee_weights_before, conn_ee_weights_after):
            print(f"{np.subtract(conn_ee_weights_after, conn_ee_weights_before)=}\n")

        if not np.allclose(conn_ei_weights_before, conn_ei_weights_after):
            print(f"{np.subtract(conn_ei_weights_after, conn_ei_weights_before)=}\n")

        # assert np.allclose(conn_ee_weights_between, conn_ee_weights_after) and np.allclose(conn_ei_weights_before, conn_ei_weights_after)

        # assert (self.spike_recorder_exc.origin + sim_time) == nest.biological_time
        if not self.params['read_out_off']:
            return sr_times_exh, sr_senders_exh, sr_times_inh, sr_senders_inh

        return sr_times_exh, sr_senders_exh

    def create_rnn_populations(self):
        """Create RNN neuronal populations consisting of excitatory and inhibitory neurons.
        """

        # Create excitatory population
        self.exc_neurons = nest.Create(self.params['exhibit_model'], self.num_exc_neurons, params=self.params['exhibit_params'])
        print(f"Create {self.num_exc_neurons=} excitatory neurons...")

        # Create inhibitory population
        self.inh_neurons = nest.Create(self.params['inhibit_model'], self.num_inh_neurons, params=self.params['inhibit_params'])
        print(f"Create {self.num_inh_neurons=} inhibitory neurons...")

    def __create_spike_generators(self):
        """Create spike generators. In total, there are three types of poisson generators. The first excites neuron clusters sequentially,
        while the second inhibits all other RNN clusters constantly. The last generator stimulates the inhibitory neurons of the RNN.
        """
        self.external_node_to_exc_neuron_dict = {}
        self.external_node_to_inh_neuron_list = []

        cluster_stimulation_time = self.params['cluster_stimulation_time']
        stimulation_gap = self.params['stimulation_gap']

        for stimulation_step in range(self.num_exc_clusters):
            external_input_per_step_list = []
            start = stimulation_step * (cluster_stimulation_time + stimulation_gap)
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start + 1, rate=self.params['inh_rate_ex'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start + 1, stop=start + cluster_stimulation_time, rate=self.params['exh_rate_ex'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start + cluster_stimulation_time, stop=start + cluster_stimulation_time + stimulation_gap, rate=self.params['inh_rate_ex'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start + cluster_stimulation_time + stimulation_gap, rate=self.params['inh_rate_ex'])))
            self.external_node_to_exc_neuron_dict[stimulation_step] = external_input_per_step_list
        self.external_node_to_inh_neuron = nest.Create('poisson_generator', params=dict(start=0.0, stop=self.num_exc_clusters * (cluster_stimulation_time + stimulation_gap), rate=self.params['exh_rate_ix']))

    def __create_recording_devices(self):
        """Create recording devices
        """

        # TODO: Should the params dictionary also be in parameters_space?
        # Create spike recorder for exc neurons
        self.spike_recorder_exc = nest.Create('spike_recorder', params={'record_to': 'ascii', 'label': 'exh_spikes'})

        # Create spike recorder for inh neurons
        self.spike_recorder_inh = nest.Create('spike_recorder', params={'record_to': 'ascii', 'label': 'inh_spikes'})

        # Create spike recorder for spike generator
        self.spike_recorder_generator = nest.Create('spike_recorder', params={'record_to': 'ascii', 'label': 'generator_spikes'})

    # TODO: change function to __create_plastic_connections() and connect E to E and I to E
    def __connect_excitatory_neurons(self):
        """Connect excitatory neurons
        """
        # EE connections
        nest.Connect(self.exc_neurons, self.exc_neurons, conn_spec=self.params['conn_dict_ee'], syn_spec=self.params['syn_dict_ee'])

    # TODO: change function to __create_static_connections() and connect I to I and E to I
    def __connect_RNN_neurons(self):
        """Create II, EI, IE connections
        """
        # II connections
        nest.Connect(self.inh_neurons, self.inh_neurons, conn_spec=self.params['conn_dict_ii'], syn_spec=self.params['syn_dict_ii'])

        # EI connections
        nest.Connect(self.inh_neurons, self.exc_neurons, conn_spec=self.params['conn_dict_ei'], syn_spec=self.params['syn_dict_ei'])

        # IE connections
        nest.Connect(self.exc_neurons, self.inh_neurons, conn_spec=self.params['conn_dict_ie'], syn_spec=self.params['syn_dict_ie'])

    def __connect_external_inputs_to_clusters(self):
        """Connect external inputs to subpopulations
        """
        # TODO: generators should be only active if we simulate for at least one round
        exc_cluster_size = self.params['exc_cluster_size']

        # Connect generators to excitatory neurons
        for cluster_index, external_nodes in self.external_node_to_exc_neuron_dict.items():
            first_neuron = cluster_index * exc_cluster_size
            last_neuron = (first_neuron + exc_cluster_size) - 1
            external_node_first_inh_gap = external_nodes[0]
            external_node_exc = external_nodes[1]
            external_node_inh_gap = external_nodes[2]
            external_node_inh = external_nodes[3]
            nest.Connect(external_node_first_inh_gap, self.exc_neurons[first_neuron:(last_neuron + 1)], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])
            nest.Connect(external_node_exc, self.exc_neurons[first_neuron:(last_neuron + 1)], conn_spec=self.params['conn_dict_ex_exc'], syn_spec=self.params['syn_dict_ex_exc'])
            nest.Connect(external_node_inh_gap, self.exc_neurons[first_neuron:(last_neuron + 1)], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])
            if first_neuron > 0:
                nest.Connect(external_node_inh, self.exc_neurons[: first_neuron], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])
            if (last_neuron + 1) < len(self.exc_neurons):
                nest.Connect(external_node_inh, self.exc_neurons[(last_neuron + 1):], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])

        # Connect generator to inhibitory neurons
        nest.Connect(self.external_node_to_inh_neuron, self.inh_neurons, conn_spec=self.params['conn_dict_ix'], syn_spec=self.params['syn_dict_ix'])

    def __connect_neurons_to_spike_recorders(self):
        """Connect excitatory, inhibitory neurons and also all generators to spike recorders
        """
        # Connect excitatory neurons to spike recorder
        nest.Connect(self.exc_neurons, self.spike_recorder_exc)

        # Connect inhibitory neurons to spike recorder
        nest.Connect(self.inh_neurons, self.spike_recorder_inh)

        if self.params['read_out_off']:
            # Connect all generators to spike recorders
            for i in range(self.num_exc_clusters):
                nest.Connect(self.external_node_to_exc_neuron_dict[i][0], self.spike_recorder_generator)
                nest.Connect(self.external_node_to_exc_neuron_dict[i][1], self.spike_recorder_generator)
                nest.Connect(self.external_node_to_exc_neuron_dict[i][2], self.spike_recorder_generator)
                nest.Connect(self.external_node_to_exc_neuron_dict[i][3], self.spike_recorder_generator)
            nest.Connect(self.external_node_to_inh_neuron, self.spike_recorder_generator)

    def create_spontaneous_dynamics_nodes(self):
        self.random_dynamics_ex = nest.Create('poisson_generator', params={'rate': self.params['random_dynamics_ex'], 'stop': 0.0})
        self.random_dynamics_ix = nest.Create('poisson_generator', params={'rate': self.params['random_dynamics_ix'], 'stop': 0.0})

        nest.Connect(self.random_dynamics_ex, self.exc_neurons, conn_spec=self.params['conn_dict_ex_random'], syn_spec=self.params['syn_dict_ex_random'])
        nest.Connect(self.random_dynamics_ix, self.inh_neurons, conn_spec=self.params['conn_dict_ix_random'], syn_spec=self.params['syn_dict_ix_random'])

    def set_up_spontaneous_dynamics(self, sim_time):
        """[summary]
        """
        # Create poisson generator for excitatory neurons and inhibitory neurons
        self.random_dynamics_ex = nest.Create('poisson_generator', params={'rate': self.params['random_dynamics_ex'], 'origin': nest.biological_time, 'start': 0.0, 'stop': sim_time})
        self.random_dynamics_ix = nest.Create('poisson_generator', params={'rate': self.params['random_dynamics_ix'], 'origin': nest.biological_time, 'start': 0.0, 'stop': sim_time})

        # Connect poisson generator to excitatory neurons and inhibitory neurons
        nest.Connect(self.random_dynamics_ex, self.exc_neurons, conn_spec=self.params['conn_dict_ex_random'], syn_spec=self.params['syn_dict_ex_random'])
        nest.Connect(self.random_dynamics_ix, self.inh_neurons, conn_spec=self.params['conn_dict_ix_random'], syn_spec=self.params['syn_dict_ix_random'])

        # # Connect poisson generators to spike recorder
        # nest.Connect(self.random_dynamics_ex, self.spike_recorder_generator)
        # nest.Connect(self.random_dynamics_ix, self.spike_recorder_generator)

    def freeze_weights(self):
        self.exc_neurons.A_LTD = 0.0
        self.exc_neurons.A_LTP = 0.0
        conn_ei = nest.GetConnections(synapse_model=self.params['syn_dict_ei']['synapse_model'])
        conn_ei.eta = 0.0

    def unfreeze_weights(self):
        self.exc_neurons.A_LTD = self.params['exhibit_params']['A_LTD']
        self.exc_neurons.A_LTP = self.params['exhibit_params']['A_LTP']
        conn_ei = nest.GetConnections(synapse_model=self.params['syn_dict_ei']['synapse_model'])
        conn_ei.eta = self.params['syn_dict_ei']['eta']

    def get_plastic_connections(self):
        conn_ee = nest.GetConnections(source=self.exc_neurons, target=self.exc_neurons, synapse_model=self.params['syn_dict_ee']['synapse_model'])
        conn_ei = nest.GetConnections(source=self.inh_neurons, target=self.exc_neurons, synapse_model=self.params['syn_dict_ei']['synapse_model'])
        conn_ee_weights = conn_ee.weight
        conn_ei_weights = conn_ei.weight

        return conn_ee_weights, conn_ei_weights

    def set_plastic_connections(self, conn_ee_weights, conn_ei_weights):
        conn_ee = nest.GetConnections(source=self.exc_neurons, target=self.exc_neurons, synapse_model=self.params['syn_dict_ee']['synapse_model'])
        conn_ei = nest.GetConnections(source=self.inh_neurons, target=self.exc_neurons, synapse_model=self.params['syn_dict_ei']['synapse_model'])
        conn_ee.weight = conn_ee_weights
        conn_ei.weight = conn_ei_weights

    # TODO: maybe change fname default name to a more common one
    def save_connections(self, source=None, target=None, synapse_model=None, fname='no_label'):
        """Save connection matrix

        Parameters
        ----------
        synapse_model: str (mandatory)
            name of synapse model
        fname: str
            name of the stored file
        """

        print('\nSave connections ...')
        if source is not None and target is not None:
            connections_all = nest.GetConnections(source=source, target=target)
        elif synapse_model is None:
            RNN_neurons = self.exc_neurons + self.inh_neurons
            connections_all = nest.GetConnections(source=RNN_neurons, target=RNN_neurons)
        else:
            connections_all = nest.GetConnections(synapse_model=synapse_model)

        connections = nest.GetStatus(connections_all, ['target', 'source', 'weight'])

        np.save('%s/%s' % (self.data_path, fname), connections)

    def load_connections(self, data_path='', fname='all_connections', static=True):  # TODO rather put this function into helper file?!
        """Load connection matrix

        Parameters
        ----------
        label: str
            name of the stored file
        """
        if os.path.isfile(data_path):
            conn_path = data_path
        elif not os.path.exists(data_path):
            conn_path = os.path.join(data_path,f'{fname}.npy')
        else:
            data_path = training_helper.get_data_path(self.params['data_path'], self.params['label'])
            conn_path = os.path.join(data_path,f'{fname}.npy')
        
        if os.path.exists(conn_path):
            conns = np.load(conn_path)
        else:
            raise Exception('Connection file does not exist. Data path or file name might be incorrect.')

        print(f'\nLoad connections from {conn_path}...')

        # conns = np.load('/Users/Jette/Desktop/results/NEST/job_3822329/1787e7674087ddd1d5749039f947a2cd/all_connections_42.npy')
        conns_tg = [int(conn[0]) for conn in conns]
        conns_src = [int(conn[1]) for conn in conns]
        conns_weights = [conn[2] for conn in conns]

        if static:
            syn_dict = {'synapse_model': 'static_synapse',
                        'delay': [self.params['dt']] * len(conns_weights),
                        'weight': conns_weights}
        nest.Connect(conns_src, conns_tg, 'one_to_one', syn_dict)

    # This function is different to the one in Julia but based on the text in the Maes et al. (2020) paper, we could assume that this is the function we need
    # TODO: Test later if it makes any difference if we use this function or normalize_weights()
    def normalize_weights_L1(self, neurons_to_be_normalized, initial_weight_inputs_dict):
        Wmin, Wmax = self.params['syn_dict_ee']['Wmin'], self.params['syn_dict_ee']['Wmax']
        for neuron in neurons_to_be_normalized:
            conn = nest.GetConnections(target=neuron, synapse_model="clopath_synapse")
            w = np.array(conn.weight)
            w_normed = w / sum(abs(w))  # L1-norm
            new_weights = initial_weight_inputs_dict[neuron.global_id] * w_normed
            new_weights = np.clip(new_weights, Wmin, Wmax)
            conn.set(weight=new_weights)

            # Tests
            assert np.prod(np.array(conn.weight) <= Wmax)
            assert np.prod(np.array(conn.weight) >= Wmin)

    def normalize_weights(self, neurons_to_be_normalized, initial_weight_inputs_dict):
        Wmin, Wmax = self.params['syn_dict_ee']['Wmin'], self.params['syn_dict_ee']['Wmax']
        for neuron in neurons_to_be_normalized:
            conn = nest.GetConnections(target=neuron, synapse_model="clopath_synapse")
            w = np.array(conn.weight)
            new_weights = w - (sum(abs(w)) - initial_weight_inputs_dict[neuron.global_id]) / len(w)
            np.clip(new_weights, Wmin, Wmax, out=new_weights)
            conn.set(weight=new_weights)

            # Tests
            # assert np.prod(np.array(conn.weight) <= Wmax)
            # assert np.prod(np.array(conn.weight) >= Wmin)

    def normalize_weights_opt(self, neurons_to_be_normalized, initial_weight_inputs_dict):
        # print("Get Connections")
        # start = time.perf_counter()
        conns = nest.GetConnections(target=neurons_to_be_normalized, synapse_model="clopath_synapse")
        # stop = time.perf_counter()
        # print(f"{stop-start=}")

        # print("Get Status")
        # start = time.perf_counter()
        mod_conns = nest.GetStatus(conns, ['target', 'source', 'weight'])
        # stop = time.perf_counter()
        # print(f"{stop-start=}")

        # print("create weight matrix")
        # start = time.perf_counter()
        weight_matrix = plot_helper.matrix_from_connections(mod_conns)
        # stop = time.perf_counter()
        # print(f"{stop-start=}")

        # print("apply normalization")
        # start = time.perf_counter()
        for target_idx, row in enumerate(weight_matrix):
            row[:] = self.apply_hard_L1(target_idx, row, initial_weight_inputs_dict)
        # stop = time.perf_counter()
        # print(f"{stop-start=}")

        print("Source with mod_conns")
        start = time.perf_counter()
        sources = [conn[1] for conn in mod_conns]
        stop = time.perf_counter()
        print(f"{stop-start=}")
        # sources = self.sources
        # print("Source with conns")
        # start = time.perf_counter()
        # check_sources = conns.source
        # stop = time.perf_counter()
        # print(f"{stop-start=}")
        # assert np.allclose(sources, check_sources)
        targets = [conn[0] for conn in mod_conns]
        # targets = self.targets
        # assert np.allclose(targets, conns.target)

        if True:
            # print("set normalized weights")
            # start = time.perf_counter()
            conns.weight = weight_matrix[np.array(targets) - 1, np.array(sources) - 1]
            # stop = time.perf_counter()
            # print(f"{stop-start=}")
        else:

            print("ll_api")
            start = time.perf_counter()
            nest.ll_api.connect_arrays(sources, targets, weight_matrix[targets - 1, sources - 1], None, "clopath_synapse", None, None)
            stop = time.perf_counter()
            print(f"{stop-start=}")

    def apply_hard_L1(self, target_idx, weight_row, initial_weight_inputs_dict):
        target_neuron = self.exc_neurons[target_idx]
        Wmin, Wmax = self.params['syn_dict_ee']['Wmin'], self.params['syn_dict_ee']['Wmax']
        new_weights = weight_row.copy()
        mask, = weight_row.nonzero()
        nonzero_entries = len(mask)
        new_weights[mask] -= (sum(abs(weight_row)) - initial_weight_inputs_dict[target_neuron.global_id]) / nonzero_entries
        new_weights[mask] = np.clip(new_weights[mask], Wmin, Wmax)
        return new_weights

    def get_initial_weight_sums_dict(self, neurons, synapse_type=None):
        initial_weight_sums_dict = {}
        for neuron in neurons:
            conn = nest.GetConnections(target=neuron, synapse_model=synapse_type)
            num_connections = len(conn)
            initial_weight_sums_dict[neuron.global_id] = num_connections * self.params['syn_dict_ee']['weight']
            # This test only gives correct result if the function is called before any weight changes TODO: remove later
            assert np.allclose(initial_weight_sums_dict[neuron.global_id], sum(abs(np.array(conn.weight))))
        return initial_weight_sums_dict

    def get_initial_weight_sums_dict_opt(self, neurons, synapse_type=None):
        conns = nest.GetConnections(source=neurons, target=neurons, synapse_model=synapse_type)
        mod_conns = nest.GetStatus(conns, ['target', 'source', 'weight'])
        weight_matrix = plot_helper.matrix_from_connections(mod_conns)
        initial_weight_sums_dict = {}

        for i, row in enumerate(weight_matrix):
            assert neurons[i].global_id == i + 1
            initial_weight_sums_dict[neurons[i].global_id] = row.sum()

        return initial_weight_sums_dict

    # This function was implemented based on the text in Maes et al. (2020) but does not correspond to the code in Julia
    def __create_spike_generators_old(self):
        """Create spike generators. In total, there are three types of poisson generators. The first excites neuron clusters sequentially,
        while the second inhibits all other RNN clusters. The last generator stimulates the inhibitory neurons of the RNN.
        """
        self.external_node_to_exc_neuron_dict = {}
        self.external_node_to_inh_neuron_list = []

        cluster_stimulation_time = self.params['cluster_stimulation_time']
        stimulation_gap = self.params['stimulation_gap']

        for stimulation_step in range(self.num_exc_clusters):
            external_input_per_step_list = []
            start = stimulation_step * (cluster_stimulation_time + stimulation_gap)
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start + cluster_stimulation_time, rate=self.params['exh_rate_ex_old'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start + cluster_stimulation_time, rate=self.params['inh_rate_ex_old'])))
            self.external_node_to_exc_neuron_dict[stimulation_step] = external_input_per_step_list
            self.external_node_to_inh_neuron_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start + cluster_stimulation_time, rate=self.params['exh_rate_ix'])))

        # TODO: set spike generator status with the above computed excitation times (see Younes code above)
    # This function was implemented based on the text in Maes et al. (2020) but does not correspond to the code in Julia
    def __connect_external_inputs_to_clusters_old(self):
        """Connect external inputs to subpopulations
        """

        exc_cluster_size = self.params['exc_cluster_size']

        # Connect generators to excitatory neurons
        for cluster_index, external_nodes in self.external_node_to_exc_neuron_dict.items():
            first_neuron = cluster_index * exc_cluster_size
            last_neuron = (first_neuron + exc_cluster_size) - 1
            external_node_exc = external_nodes[0]
            external_node_inh = external_nodes[1]
            nest.Connect(external_node_exc, self.exc_neurons[first_neuron:(last_neuron + 1)], conn_spec=self.params['conn_dict_ex_exc'], syn_spec=self.params['syn_dict_ex_exc'])
            if first_neuron > 0:
                nest.Connect(external_node_inh, self.exc_neurons[: first_neuron], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])
            if (last_neuron + 1) < len(self.exc_neurons):
                nest.Connect(external_node_inh, self.exc_neurons[(last_neuron + 1):], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])

        # Connect generators to inhibitory neurons
        for external_node in self.external_node_to_inh_neuron_list:
            nest.Connect(external_node, self.inh_neurons, conn_spec=self.params['conn_dict_ix'], syn_spec=self.params['syn_dict_ix'])

    ##############################################
    # This function was implemented based on the text in Maes et al. (2020) but does not correspond to the code in Julia
    def __connect_neurons_to_spike_recorders_old(self):
        """Connect excitatory, inhibitory neurons and also all generators to spike recorders
        """
        # Connect excitatory neurons to spike recorder
        nest.Connect(self.exc_neurons, self.spike_recorder_exc)

        # Connect inhibitory neurons to spike recorder
        nest.Connect(self.inh_neurons, self.spike_recorder_inh)

        # Connect all generators to spike recorders
        for i in range(self.num_exc_clusters):
            nest.Connect(self.external_node_to_exc_neuron_dict[i][0], self.spike_recorder_generator)
            nest.Connect(self.external_node_to_exc_neuron_dict[i][1], self.spike_recorder_generator)
            nest.Connect(self.external_node_to_inh_neuron_list[i], self.spike_recorder_generator)

    ##############################################
    def __get_cluster_neurons(self, index_cluster):
        """Get neuron's indices (NEST NodeCollection) belonging to a subpopulation

        Parameters
        ---------
        index_subpopulation: int

        Returns
        -------
        NEST NodeCollection
        """

        neurons_indices = [int(index_cluster) * self.params['exc_cluster_size'] + i for i in
                           range(self.params['exc_cluster_size'])]

        return self.exc_neurons[neurons_indices]


##############################################
def get_parameters():
    """Import model-parameter file.

    Returns
    -------
    params: dict
        Parameter dictionary.
    """

    import parameters_space
    params = parameters_space.p

    return params


###########################################
def load_input_encoding(path, fname):
    """Load input encoding: association between sequence element and subpopulations

    Parameters
    ----------
    path: str
    fname: str

    Returns
    -------
    characters_to_subpopulations: dict
    """

    characters_to_subpopulations = training_helper.load_data(path, fname)

    return characters_to_subpopulations


if __name__ == '__main__':
    import experiments.parameters_space as parameters_space

    class Pseudomodel:
        pass
    nest.ResetKernel()
    nest.Install('nestmlmodule')
    model = Pseudomodel()

    model.params = parameters_space.p

    model.exc_neurons = nest.Create("aeif_cond_diff_exp_clopath", 240)
    model.inh_neurons = nest.Create("iaf_cond_diff_exp", 60)

    general_RNN_conn_dict = {'rule': 'pairwise_bernoulli',              # Connection rule
                             'p': 0.2,                     # Connection probability of neurons in RNN
                             'allow_autapses': False,                    # If False then no self-connections are allowed
                             'allow_multapses': False                    # If False then only one connection between the neurons is allowed
                             }
    p = {}

    p['syn_dict_ee'] = {'synapse_model': 'clopath_synapse',             # Name of synapse model - TODO: In MATLAB code the weights might be randomized
                        'weight': 2.83,                                 # Initial synaptic weight (pF)
                        'Wmax': 32.68,                                  # Maximum allowed weight (pF)
                        'Wmin': 1.45,                                   # Minimum allowed weight (pF)
                        'tau_x': 3.5,                                   # Time constant of low pass filtered presynaptic spike train in recurrent network (ms)
                        # 'delay': 0.0                                   # Synaptic delay (ms)
                        }

    p['syn_dict_ii'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                        'weight': - 20.91                                 # Synaptic weight (pF)
                        }

    p['syn_dict_ie'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                        'weight': 1.96                                  # Synpatic weight (pF)
                        }

    p['syn_dict_ei'] = {'synapse_model': 'vogels_sprekeler_synapse',    # Name of synapse model
                        'weight': - 62.87,                              # Initial synpatic weight (pF)
                        'eta': 1.0,                                     # TODO: Should be the same as the learning rate, in Julia code it is 1.0 but in the paper it is 10^-5
                        'alpha': 2.0 * 3.0 * 20.0,                      # TODO: set r_0 and tau_y above -> alpha = 2*r_0*tau_y
                        'Wmax': - 243.0,                                # Maximum allowed weight (pF)
                        'Wmin': -48.7                                   # Minimum allowed weight (pF)
                        }

    nest.Connect(model.exc_neurons, model.exc_neurons, conn_spec=general_RNN_conn_dict, syn_spec=p['syn_dict_ee'])
    nest.Connect(model.inh_neurons, model.inh_neurons, conn_spec=general_RNN_conn_dict, syn_spec=p['syn_dict_ii'])
    nest.Connect(model.inh_neurons, model.exc_neurons, conn_spec=general_RNN_conn_dict, syn_spec=p['syn_dict_ei'])
    nest.Connect(model.exc_neurons, model.inh_neurons, conn_spec=general_RNN_conn_dict, syn_spec=p['syn_dict_ie'])

    model.random_dynamics_ex = nest.Create('poisson_generator', params={'rate': model.params['random_dynamics_ex'], 'origin': nest.biological_time, 'start': 0, 'stop': 3000.0})
    model.random_dynamics_ix = nest.Create('poisson_generator', params={'rate': model.params['random_dynamics_ix'], 'origin': nest.biological_time, 'start': 0, 'stop': 3000.0})

    # Connect poisson generator to excitatory neurons and inhibitory neurons
    nest.Connect(model.random_dynamics_ex, model.exc_neurons, conn_spec=model.params['conn_dict_ex_random'], syn_spec=model.params['syn_dict_ex_random'])
    nest.Connect(model.random_dynamics_ix, model.inh_neurons, conn_spec=model.params['conn_dict_ix_random'], syn_spec=model.params['syn_dict_ix_random'])

    initial_weight_inputs_dict = Model.get_initial_weight_sums_dict(model, neurons=model.exc_neurons, synapse_type='clopath_synapse')

    sr = nest.Create('spike_recorder')
    sr.record_to = 'memory'
    nest.Connect(model.exc_neurons, sr)

    Model.simulate_random_dynamics(model, sim_time=3000.0, normalization_time=15.0, initial_weight_inputs=initial_weight_inputs_dict)

    print(f"{sr.events=}")
