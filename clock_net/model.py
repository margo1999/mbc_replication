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
import copy
import numpy as np
from collections import defaultdict
import sys
from tqdm import tqdm
from nest import voltage_trace
import matplotlib.pyplot as plt
from clock_net import plot_helper
from time import perf_counter
from figures.plot_results import plot_2_mins_results

from clock_net import helper


class Model:
    """Instantiation of the Clock Network model and its PyNEST implementation.

    The model provides the following member functions: 

    __init__(parameters)
    create()
    connect()
    simulate(t_sim)

    In addition, each model may implement other model-specific member functions.
    """

    def __init__(self, params, sequences, vocabulary):
        """Initialize model and simulation instance, including

        1) Parameter setting,
        2) Generate sequence data,
        3) Configuration of the NEST kernel,
        4) Setting random-number generator seed, and

        Parameters
        ----------
        params:     dict
                    Parameter dictionary
        seuqences:  list
                    Sequences to learn
        vocabulary: list
                    Vocabulary from which the sequences are constructed
        """

        print('\nInitialising model and simulation...')

        # Set parameters derived from base parameters
        self.params = helper.derived_parameters(params)

        # Data directory
        if self.params['evaluate_replay']:
            self.data_path = helper.get_data_path(self.params['data_path'], self.params['label'], 'replay')
        else:
            self.data_path = helper.get_data_path(self.params['data_path'], self.params['label'])
        print(f"{self.params['label']=}")
        print(f"{self.data_path=}")

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

        # Initialize RNG        
        np.random.seed(self.params['seed'])
        random.seed(self.params['seed'])

        # Input stream: sequence data
        self.sequences = sequences
        self.vocabulary = vocabulary
        self.length_sequence = len(self.sequences[0]) # TODO: possible that sequences do not have the same length? 
        self.num_sequences = len(self.sequences)

        if params['task']['task_name'] != 'hard_coded':
            assert self.length_sequences == params['task']['length_sequence']

        #self.random_dynamics_ex = None
        #self.random_dynamics_ix = None
        
        

        # initialize the NEST kernel
        self.__setup_nest()

    def __setup_nest(self):
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
        self.__create_RNN_populations()

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
        if self.params['load_connections']:
            self.__load_connections(label='ee_connections')
        else:
            self.__connect_excitatory_neurons()

        # Connect inhibitory population (II, EI, IE)
        self.__connect_RNN_neurons()

        # Connect external input
        self.__connect_external_inputs_to_clusters()

        # Connect neurons to the spike recorder
        self.__connect_neurons_to_spike_recorders()
        
        #self.record_behaviour_of_exc_connection()
        print('\nAll nodes connected...')
        print(f"Number of EE connections: {len(nest.GetConnections(source=self.exc_neurons, target=self.exc_neurons))=}")
        print(f"Number of IE connections: {len(nest.GetConnections(source=self.exc_neurons, target=self.inh_neurons))=}")
        print(f"Number of EI connections: {len(nest.GetConnections(source=self.inh_neurons, target=self.exc_neurons))=}")
        print(f"Number of II connections: {len(nest.GetConnections(source=self.inh_neurons, target=self.inh_neurons))=}")
        print(f"Number of IX connections: {len(nest.GetConnections(source=self.external_node_to_inh_neuron, target=self.inh_neurons))=}")

    def simulate(self):
        """Run simulation by stopping after each round to reset the times of the spike generators.
        """

        round_duration = self.params['round_time']
        random_dynamics_time = self.params['random_dynamics_time']
        normalization_time = self.params['normalization_time']
        initial_weight_inputs_dict = self.get_initial_weight_sums_dict(neurons=self.exc_neurons, synapse_type='clopath_synapse')

        # Rank() returns the MPI rank of the local process TODO: simulating time not correct
        if nest.Rank() == 0:
            print('\nSimulating {} ms.'.format(round_duration))

        self.train_RNN(round_duration, normalization_time, initial_weight_inputs_dict)

        if self.params['random_dynamics']:
            self.simulate_random_dynamics(random_dynamics_time, normalization_time, initial_weight_inputs_dict)

    def record_behaviour_of_exc_connection(self): # TODO: only for 1 round 
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
        nest.Connect(self.exc_neurons[sourceid - 1], self.exc_neurons[targetid - 1], syn_spec={'synapse_model': 'clopath_synapse_wr'}) # TODO: Will be ignored in save_connections

    def record_behaviour_of_inh_neuron(self, neuronid = None): #TODO: make record duration more general
         if neuronid is not None:
            self.mm_inh = nest.Create('multimeter', params={'record_from': ['g_ex__X__spikeExc', 'g_in__X__spikeInh', 'V_m'], 'interval': 0.1, 'stop': 120.0})
            nest.Connect(self.mm_inh, self.inh_neurons[neuronid - 1])

    def record_behaviour_of_exc_neuron(self, neuronid = None): #TODO: make record duration more general
         if neuronid is not None:
            self.mm_exc = nest.Create('multimeter', params={'record_from': ['g_ex', 'g_in', 'u_bar_bar', 'u_bar_minus', 'u_bar_plus', 'V_m', 'V_th', 'w'], 'interval': 0.1, 'stop': 120.0})
            nest.Connect(self.mm_exc, self.exc_neurons[neuronid - 1])

    def train_RNN(self, round_duration, normalization_time, initial_weight_inputs_dict):
  
        training_iterations = self.params['training_iterations']
        rounds = (-(-(2*60*1000) // int(round_duration))) # 2 min / round_duration
        for two_min_unit in tqdm(range(training_iterations)): 

            for round_ in tqdm(range(rounds)):

                esttime = ((round_ + (rounds * two_min_unit)) * (round_duration)) + two_min_unit * 3000.0
                curtime = nest.biological_time
                assert esttime == curtime
                assert round_duration % normalization_time == 0

                simulate_steps = int(round_duration // normalization_time)

                nest.Prepare()
                for twenty_ms_unit in range(simulate_steps):

                    nest.Run(normalization_time)
                
                    # TODO: Is normalization realy necessary every 20 ms. Creates large overhead.
                    #self.normalize_weights('clopath_synapse', initial_weight_inputs_dict)

                nest.Cleanup()

                # Turn off all spike recorders and set spike recorder for exc neurons to 'memory' to record spikes more flexible
                if two_min_unit + round_ == 0:
                    self.spike_recorder_exc.record_to = "memory" # TODO: Weird behaviour! No spikes are stored if I don't set first record_to and then stop
                    self.spike_recorder_inh.record_to = "memory"
                    self.spike_recorder_generator.record_to = "memory"
                    self.spike_recorder_exc.stop = nest.biological_time
                    self.spike_recorder_inh.stop = nest.biological_time
                    self.spike_recorder_generator.stop = nest.biological_time


                # Simulation is stopped to set a new reference time (origin) for start and stop of the generators, otherwise they would only spike at the beginning
                if (round_ == (rounds - 1)) and two_min_unit < (training_iterations - 1):
                    for generators_to_exc in self.external_node_to_exc_neuron_dict.values():
                        generators_to_exc[0].origin += round_duration + 3000.0
                        generators_to_exc[1].origin += round_duration + 3000.0
                        generators_to_exc[2].origin += round_duration + 3000.0
                    
                    self.external_node_to_inh_neuron.origin += round_duration + 3000.0

                elif round_ < (rounds - 1):    
                    for generators_to_exc in self.external_node_to_exc_neuron_dict.values():
                        generators_to_exc[0].origin += round_duration
                        generators_to_exc[1].origin += round_duration
                        generators_to_exc[2].origin += round_duration
                    
                    self.external_node_to_inh_neuron.origin += round_duration

            #plot_helper.plot_behaviour_of_exc_connection(self.wr, self.data_path)
            plot_helper.plot_behaviour_of_exc_neuron(self.mm_exc, self.data_path, self.params)
            plot_helper.plot_behaviour_of_inh_neuron(self.mm_inh, self.data_path, self.params)

            # Save ee connections after two minutes
            file_name = f"ee_connections_{two_min_unit}.npy"
            self.save_connections(synapse_model=self.params['syn_dict_ee']['synapse_model'], fname=file_name)
            connectionsfilepath = os.path.join(self.data_path, file_name)
            
            if True:
                # Save current spike behaviour under random input dynamics
                sr_times_exh, sr_senders_exh = self.record_exc_spike_behaviour(3000.0, normalization_time, initial_weight_inputs_dict)
                # print(f"{len(sr_times_exh)=}", f"{sr_times_exh[0]=}", f"{sr_times_exh[-1]=}")
                spikes = dict(sr_times_exh=sr_times_exh, sr_senders_exh=sr_senders_exh)
                # print(f"{len(spikes)=}", f"{len(spikes['sr_times_exh'])=}")
                spikefilepath = os.path.join(self.data_path, f"spikes_{two_min_unit}.pickle")
                dump(spikes, open(spikefilepath, "wb"))

                # Plot and save plot of connection and spike behaviour as png and pickle file
                plotsfilepath = os.path.join(self.data_path, f"plots_{two_min_unit}")
                plot_2_mins_results(spikefilepath, connectionsfilepath, plotsfilepath)

                # # TODO: Save sprectrum 


    def simulate_random_dynamics(self, sim_time, normalization_time, initial_weight_inputs):
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


        # TODO: Set stdp delay

        simulate_steps = int(sim_time // normalization_time)

        nest.Prepare()

        for i in range(simulate_steps):
            nest.Run(normalization_time)
            #self.normalize_weights(self.exc_neurons, initial_weight_inputs=initial_weight_inputs)
        
        if sim_time % normalization_time != 0:
            remaining_time = sim_time - normalization_time * simulate_steps
            nest.Run(remaining_time)
            assert remaining_time < normalization_time

        nest.Cleanup()

        # TODO: Reset stdp delay

        # print(f"{self.random_dynamics_ex.stop=}", f"{nest.biological_time=}")
        assert (self.random_dynamics_ex.origin + sim_time) == nest.biological_time
        assert (self.random_dynamics_ix.origin + sim_time) == nest.biological_time

    def record_exc_spike_behaviour(self, sim_time, normalization_time, initial_weight_inputs):

        # TEST
        conn_ee_weights_before, conn_ei_weights_before = self.get_plastic_connections()

        # freeze weights because no learning should happen
        self.freeze_weights()
        #print(f"{self.exc_neurons.A_LTP=}")

        sp_params = {'record_to': 'memory', 'origin': nest.biological_time, 'start': 0.0, 'stop': sim_time}
        nest.SetStatus(self.spike_recorder_exc, params=sp_params)

        #self.simulate_random_dynamics(sim_time, normalization_time, initial_weight_inputs)
        self.simulate_random_dynamics(sim_time, normalization_time, initial_weight_inputs)

        # Save spikes after simulation
        sr_times_exh =  self.spike_recorder_exc.events['times']
        sr_senders_exh = self.spike_recorder_exc.events['senders']

        self.spike_recorder_exc.n_events = 0 # reset the spike counts

        conn_ee_weights_between, conn_ei_weights_between = self.get_plastic_connections()
        self.simulate_random_dynamics(sim_time, normalization_time, initial_weight_inputs)

        # unfreeze weights
        self.unfreeze_weights()

        # TEST
        conn_ee_weights_after, conn_ei_weights_after = self.get_plastic_connections()
        if not np.allclose(conn_ee_weights_between, conn_ee_weights_after):
            print(f"{max(abs(np.subtract(conn_ee_weights_after, conn_ee_weights_between)))=}\n")
            print(f"{len((np.subtract(conn_ee_weights_after, conn_ee_weights_between)))=}\n")
            print(f"{np.count_nonzero(np.subtract(conn_ee_weights_after, conn_ee_weights_between))=}\n")

        if not np.allclose(conn_ee_weights_before, conn_ee_weights_between):
            print(f"{np.subtract(conn_ee_weights_between, conn_ee_weights_before)=}\n")

        if not np.allclose(conn_ee_weights_before, conn_ee_weights_after):
            print(f"{np.subtract(conn_ee_weights_after, conn_ee_weights_before)=}\n")

        if not np.allclose(conn_ei_weights_before, conn_ei_weights_after):
            print(f"{np.subtract(conn_ei_weights_after, conn_ei_weights_before)=}\n")

        #assert np.allclose(conn_ee_weights_between, conn_ee_weights_after) and np.allclose(conn_ei_weights_before, conn_ei_weights_after)

        #assert (self.spike_recorder_exc.origin + sim_time) == nest.biological_time

        return sr_times_exh, sr_senders_exh


    def __create_RNN_populations(self):
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
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time, rate=self.params['exh_rate_ex'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start+cluster_stimulation_time, stop=start+cluster_stimulation_time+stimulation_gap, rate=self.params['inh_rate_ex'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time+stimulation_gap, rate=self.params['inh_rate_ex'])))
            self.external_node_to_exc_neuron_dict[stimulation_step] = external_input_per_step_list
        self.external_node_to_inh_neuron = nest.Create('poisson_generator', params=dict(start=0.0, stop=self.num_exc_clusters*(cluster_stimulation_time+stimulation_gap), rate=self.params['exh_rate_ix']))
        
        
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

    # TODO: change function to __create_plasticity_connections() and connect E to E and I to E 
    def __connect_excitatory_neurons(self):
        """Connect excitatory neurons
        """
        # EE connections
        nest.Connect(self.exc_neurons, self.exc_neurons, conn_spec=self.params['conn_dict_ee'], syn_spec=self.params['syn_dict_ee'])

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
            external_node_exc = external_nodes[0]
            external_node_inh_gap = external_nodes[1]
            external_node_inh = external_nodes[2]
            nest.Connect(external_node_exc, self.exc_neurons[first_neuron:(last_neuron+1)], conn_spec=self.params['conn_dict_ex_exc'], syn_spec=self.params['syn_dict_ex_exc'])
            nest.Connect(external_node_inh_gap, self.exc_neurons[first_neuron:(last_neuron+1)], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])
            if first_neuron > 0:
                nest.Connect(external_node_inh, self.exc_neurons[: first_neuron], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])
            if (last_neuron+1) < len(self.exc_neurons):
                nest.Connect(external_node_inh, self.exc_neurons[(last_neuron+1):], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])     
        
        # Connect generator to inhibitory neurons
        nest.Connect(self.external_node_to_inh_neuron, self.inh_neurons, conn_spec=self.params['conn_dict_ix'], syn_spec=self.params['syn_dict_ix'])

    def __connect_neurons_to_spike_recorders(self):
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
            nest.Connect(self.external_node_to_exc_neuron_dict[i][2], self.spike_recorder_generator)
        nest.Connect(self.external_node_to_inh_neuron, self.spike_recorder_generator)

    def create_spontaneous_dynamics_nodes(self):
        self.random_dynamics_ex = nest.Create('poisson_generator', params={'rate':self.params['random_dynamics_ex'], 'stop': 0.0})
        self.random_dynamics_ix = nest.Create('poisson_generator', params={'rate':self.params['random_dynamics_ix'], 'stop': 0.0})

        nest.Connect(self.random_dynamics_ex, self.exc_neurons, conn_spec=self.params['conn_dict_ex_random'], syn_spec=self.params['syn_dict_ex_random'])
        nest.Connect(self.random_dynamics_ix, self.inh_neurons, conn_spec=self.params['conn_dict_ix_random'], syn_spec=self.params['syn_dict_ix_random'])
    
    def set_up_spontaneous_dynamics(self, sim_time):
        """[summary]
        """
        # Create poisson generator for excitatory neurons and inhibitory neurons
        self.random_dynamics_ex = nest.Create('poisson_generator', params={'rate':self.params['random_dynamics_ex'], 'origin': nest.biological_time, 'start': 0.0, 'stop': sim_time})
        self.random_dynamics_ix = nest.Create('poisson_generator', params={'rate':self.params['random_dynamics_ix'], 'origin': nest.biological_time, 'start': 0.0, 'stop': sim_time})

        # Connect poisson generator to excitatory neurons and inhibitory neurons
        nest.Connect(self.random_dynamics_ex, self.exc_neurons, conn_spec=self.params['conn_dict_ex_random'], syn_spec=self.params['syn_dict_ex_random'])
        nest.Connect(self.random_dynamics_ix, self.inh_neurons, conn_spec=self.params['conn_dict_ix_random'], syn_spec=self.params['syn_dict_ix_random'])

        # # Connect poisson generators to spike recorder
        # nest.Connect(self.random_dynamics_ex, self.spike_recorder_generator)
        # nest.Connect(self.random_dynamics_ix, self.spike_recorder_generator)

    def freeze_weights(self):
        self.exc_neurons.A_LTD = 0.0
        self.exc_neurons.A_LTP = 0.0
        conn_ei = nest.GetConnections(synapse_model= self.params['syn_dict_ei']['synapse_model'])
        conn_ei.eta = 0.0

    def unfreeze_weights(self):
        self.exc_neurons.A_LTD = self.params['exhibit_params']['A_LTD']
        self.exc_neurons.A_LTP = self.params['exhibit_params']['A_LTP']
        conn_ei = nest.GetConnections(synapse_model= self.params['syn_dict_ei']['synapse_model'])
        conn_ei.eta = self.params['syn_dict_ei']['eta']
    
    def get_plastic_connections(self):
        conn_ee = nest.GetConnections(synapse_model= self.params['syn_dict_ee']['synapse_model'])
        conn_ei = nest.GetConnections(synapse_model= self.params['syn_dict_ei']['synapse_model'])
        conn_ee_weights = conn_ee.weight
        conn_ei_weights = conn_ei.weight

        return conn_ee_weights, conn_ei_weights

    def set_plastic_connections(self, conn_ee_weights, conn_ei_weights):
        conn_ee = nest.GetConnections(synapse_model= self.params['syn_dict_ee']['synapse_model'])
        conn_ei = nest.GetConnections(synapse_model= self.params['syn_dict_ei']['synapse_model'])
        conn_ee.weight = conn_ee_weights
        conn_ei.weight = conn_ei_weights

    # TODO: maybe change fname default name to a more common one
    def save_connections(self, synapse_model=None, fname='ee_connections'):
        """Save connection matrix

        Parameters
        ----------
        synapse_model: str (mandatory)
            name of synapse model 
        fname: str
            name of the stored file
        """

        print('\nSave connections ...')
        assert synapse_model is not None, "Need parameter synapse_model!"

        connections_all = nest.GetConnections(synapse_model=synapse_model)
        connections = nest.GetStatus(connections_all, ['target', 'source', 'weight'])

        np.save('%s/%s' % (self.data_path, fname), connections)

    def __load_connections(self, label='ee_connections'):
        """Load connection matrix
        
        Parameters
        ----------
        label: str
            name of the stored file
        """

        assert self.params['syn_dict_ee']['synapse_model'] != 'stdsp_synapse_rec', "synapse model not tested yet"

        print('\nLoad connections ...')
        data_path = helper.get_data_path(self.params['data_path'], self.params['label'])
        conns = np.load('%s/%s.npy' % (data_path, label))
        conns_tg = [int(conn[0]) for conn in conns]
        conns_src = [int(conn[1]) for conn in conns]
        conns_weights = [conn[2] for conn in conns]

        if self.params['evaluate_replay']:
            syn_dict = {'receptor_type': 2,
                        'delay': [self.params['syn_dict_ee']['delay']] * len(conns_weights),
                        'weight': conns_weights}
            nest.Connect(conns_src, conns_tg, 'one_to_one', syn_dict)
        else:
            # TODO: clean up!
            syn_dict_ee = copy.deepcopy(self.params['syn_dict_ee'])

            del syn_dict_ee['synapse_model']
            del syn_dict_ee['weight']
            del syn_dict_ee['receptor_type']
            if self.params['syn_dict_ee']['synapse_model'] == 'stdsp_synapse':
                del syn_dict_ee['permanence']

            nest.SetDefaults('stdsp_synapse', syn_dict_ee)

            if self.params['syn_dict_ee']['synapse_model'] == 'stdsp_synapse':
                syn_dict = {'synapse_model': 'stdsp_synapse',
                            'receptor_type': 2,
                            'weight': conns_weights}
            else:
                syn_dict = {'synapse_model': 'stdsp_synapse',
                            'receptor_type': 2,
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

    def get_initial_weight_sums_dict(self, neurons, synapse_type=None):
        initial_weight_sums_dict = {}
        for neuron in neurons:
            conn = nest.GetConnections(target=neuron, synapse_model=synapse_type)
            num_connections = len(conn)
            initial_weight_sums_dict[neuron.global_id] = num_connections * self.params['syn_dict_ee']['weight']
            # This test only gives correct result if the function is called before any weight changes TODO: remove later
            assert np.allclose(initial_weight_sums_dict[neuron.global_id], sum(abs(np.array(conn.weight))))
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
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time, rate=self.params['exh_rate_ex_old'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time, rate=self.params['inh_rate_ex_old'])))
            self.external_node_to_exc_neuron_dict[stimulation_step] = external_input_per_step_list
            self.external_node_to_inh_neuron_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time, rate=self.params['exh_rate_ix'])))

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
            nest.Connect(external_node_exc, self.exc_neurons[first_neuron:(last_neuron+1)], conn_spec=self.params['conn_dict_ex_exc'], syn_spec=self.params['syn_dict_ex_exc'])
            if first_neuron > 0:
                nest.Connect(external_node_inh, self.exc_neurons[: first_neuron], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])
            if (last_neuron+1) < len(self.exc_neurons):
                nest.Connect(external_node_inh, self.exc_neurons[(last_neuron+1):], conn_spec=self.params['conn_dict_ex_inh'], syn_spec=self.params['syn_dict_ex_inh'])     
        
        # Connect generators to inhibitory neurons
        for external_node in self.external_node_to_inh_neuron_list:
            nest.Connect(external_node, self.inh_neurons, conn_spec=self.params['conn_dict_ix'], syn_spec=self.params['syn_dict_ix'])

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
    
    def __get_time_constant_dendritic_rate(self, DeltaT=40., DeltaT_seq=100., calibration=100, target_firing_rate=1):
        """Compute time constant of the dendritic AP rate,

        The time constant is set such that the rate captures how many dAPs a neuron generated
        all along the period of a batch
         
        Parameters
        ----------
        calibration : float
        target_firing_rate : float

        Returns
        -------
        float
           time constant of the dendritic AP rate
        """

        t_exc = ((self.length_sequence-1) * DeltaT + DeltaT_seq + calibration) \
                * self.num_sequences

        print("\nDuration of a sequence set %d ms" % t_exc)

        return target_firing_rate * t_exc
    
    def __set_min_synaptic_strength(self):
        """Set synaptic Wmin
        """

        print('\nSet min synaptic strength ...')
        connections = nest.GetConnections(synapse_model=self.params['syn_dict_ee']['synapse_model'])
 
        syn_model = self.params['syn_dict_ee']['synapse_model']
        if syn_model == 'stdsp_synapse' or syn_model == 'stdsp_synapse_rec':
            connections.set({'Pmin': connections.permanence})
        else:
            connections.set({'Wmin': connections.weight})

    
    def __stimulus_preference(self, fname='characters_to_subpopulations'):
        """Assign a subset of subpopulations to a each element in the vocabulary.

        Parameters
        ----------
        fname : str

        Returns
        -------
        characters_to_subpopulations: dict
        """

        if len(self.vocabulary) * self.params['L'] > self.num_exc_clusters:
            raise ValueError(
                "num_subpopulations needs to be large than length_user_characters*num_subpopulations_per_character")

        characters_to_subpopulations = defaultdict(list)  # a dictionary that assigns mini-subpopulation to characters

        subpopulation_indices = np.arange(self.num_exc_clusters)
        # permuted_subpopulation_indices = np.random.permutation(subpopulation_indices)
        permuted_subpopulation_indices = subpopulation_indices
        index_characters_to_subpopulations = []

        if self.params['load_connections']:
            # load connectivity: from characters to mini-subpopulations
            path = helper.get_data_path(self.params['data_path'], self.params['label'])
            characters_to_subpopulations = load_input_encoding(path, fname)
        else:
            for char in self.vocabulary:
                # randomly select a subset of mini-subpopulations for a character
                characters_to_subpopulations[char] = permuted_subpopulation_indices[:self.params['L']]
                # delete mini-subpopulations from the permuted_subpopulation_indices that are already selected
                permuted_subpopulation_indices = permuted_subpopulation_indices[self.params['L']:]

        return characters_to_subpopulations

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
    
    def __compute_timing_external_inputs(self, DeltaT, DeltaT_seq, DeltaT_cue, excitation_start, time_dend_to_somatic):
        """
        Specifies the excitation times of the external input for each sequence element,
        subsequent sequence elements are presented  with  inter-stimulus interval DeltaT,  
        subsequent sequences are separated in time by an inter-sequence time interval DeltaT_seq,
        during the replay, the presented cues are seperated by an intercue time interval Delta_cue,
        In addition this function saves the times at which a dendritic current should be recorded,
        we don't want to record the dendritic current every time step as this consumes a lot of memory,
        so we instead record the dendritic current every 'episodes_to_testing' episodes,
        recording the dendritic current is essential for computing the prediction performance,
        the dendritic current is saved only at the time of last element in the sequence,
        this is because when assessing the prediction performance, we compute the prediction error 
        only with respect to the last element in the sequence
        
        Parameters
        ---------
        DeltaT               : float
        DeltaT_seq           : float
        DeltaT_cue           : float 
        excitation_start     : float
        time_dend_to_somatic : float

        Returns:
        --------
        excitation_times: list(float)
        excitation_times_soma: dict
        """

        #TODO adapt for the case of the clock network
        excitation_times_soma = defaultdict(list)

        excitation_times = []
        sim_time = excitation_start
        for le in range(self.params['learning_episodes'] + 1):

            for seq_num, sequence in enumerate(self.sequences):
                len_seq = len(sequence)
                for i, char in enumerate(sequence):

                    if i != 0:
                        sim_time += DeltaT

                    # store time of excitation for each symbol
                    excitation_times_soma[char] += [sim_time]

                    excitation_times.append(sim_time)

                    if self.params['evaluate_replay']:
                        break

                # set timing between sequences
                if self.params['evaluate_replay']:
                    sim_time += DeltaT_cue
                else:
                    sim_time += DeltaT_seq

        # save data
        if self.params['evaluate_performance'] or self.params['evaluate_replay']:
            np.save('%s/%s' % (self.data_path, 'excitation_times_soma'),
                    excitation_times_soma)
            np.save('%s/%s' % (self.data_path, 'excitation_times'), excitation_times)

        self.sim_time = sim_time
        return excitation_times, excitation_times_soma

    def __create_spike_generators_old(self, excitation_times_dict):
        """Create spike generators
        """

        self.input_excitation_dict = {}
        for char in self.vocabulary:
            self.input_excitation_dict[char] = nest.Create('spike_generator')
            print('WARNING')
            print(nest.network_size)
            sys.exit(1)

        # set spike generator status with the above computed excitation times
        for char in self.vocabulary:
            nest.SetStatus(self.input_excitation_dict[char], {'spike_times': excitation_times_dict[char]})


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

    characters_to_subpopulations = helper.load_data(path, fname)

    return characters_to_subpopulations

if __name__ == '__main__':
    import experiments.sequential_dynamics.parameters_space as parameters_space
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
                    #'delay': 0.0                                   # Synaptic delay (ms)
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

    model.random_dynamics_ex = nest.Create('poisson_generator', params={'rate':model.params['random_dynamics_ex'], 'origin': nest.biological_time, 'start': 0, 'stop': 3000.0})
    model.random_dynamics_ix = nest.Create('poisson_generator', params={'rate':model.params['random_dynamics_ix'], 'origin': nest.biological_time, 'start': 0, 'stop': 3000.0})

        # Connect poisson generator to excitatory neurons and inhibitory neurons
    nest.Connect(model.random_dynamics_ex, model.exc_neurons, conn_spec=model.params['conn_dict_ex_random'], syn_spec=model.params['syn_dict_ex_random'])
    nest.Connect(model.random_dynamics_ix, model.inh_neurons, conn_spec=model.params['conn_dict_ix_random'], syn_spec=model.params['syn_dict_ix_random'])

    initial_weight_inputs_dict = Model.get_initial_weight_sums_dict(model,neurons=model.exc_neurons, synapse_type='clopath_synapse')

    sr = nest.Create('spike_recorder')
    sr.record_to = 'memory'
    nest.Connect(model.exc_neurons, sr)

    Model.simulate_random_dynamics(model, sim_time=3000.0, normalization_time=15.0, initial_weight_inputs=initial_weight_inputs_dict)

    print(f"{sr.events=}")

    



