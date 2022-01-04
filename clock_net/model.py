"""PyNEST Clock Network: Model Class [1]
----------------------------------------

Main file of the Clock Network defining ``Model`` class with function
to build, connect and simulate the network.

Authors
~~~~~~~
Jette Oberlaender, Younes Bouhadjar
"""

import random
import nest
import copy
import numpy as np
from collections import defaultdict
import sys
from tqdm import tqdm
from nest import voltage_trace
import matplotlib.pyplot as plt

from clock_net import helper


class Model:
    """Instantiation of the Clock Network model and its PyNEST implementation.

    the model provides the following member functions: 

    __init__(parameters)
    create()
    connect()
    simulate(t_sim)

    In addition, each model may implement other model-specific member functions.
    """

    def __init__(self, params, sequences, vocabulary):
        """Initialize model and simulation instance, including

        1) parameter setting,
        2) generate sequence data,
        3) configuration of the NEST kernel,
        4) setting random-number generator seed, and

        Parameters
        ----------
        params:    dict
                   Parameter dictionary
        """

        print('\nInitialising model and simulation...')

        # set parameters derived from base parameters
        self.params = helper.derived_parameters(params)

        # data directory
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

        # set network size parameters
        self.num_exc_neurons = params['num_exc_neurons']
        self.num_inh_neurons = params['num_inh_neurons']
        self.num_exc_clusters = params['num_exc_clusters']

        # initialize RNG        
        np.random.seed(self.params['seed'])
        random.seed(self.params['seed'])

        # input stream: sequence data
        self.sequences = sequences
        self.vocabulary = vocabulary
        self.length_sequence = len(self.sequences[0]) # TODO: possible that sequences do not have the same length
        self.num_sequences = len(self.sequences)

        # initialize the NEST kernel
        self.__setup_nest()

        # get time constant for dendriticAP rate
        #self.params['exhibit_params']['tau_h'] = self.__get_time_constant_dendritic_rate(
        #    calibration=self.params['calibration'])

    def __setup_nest(self):
        """Initializes the NEST kernel.
        """

        nest.ResetKernel()
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

        # create excitatory and inhibitory population of RNN
        self.__create_RNN_populations()

        # create recording devices
        self.__create_recording_devices()

        # create spike generators
        self.__create_spike_generators()

        #TODO: Can this be deleted?
        # compute timing of the external inputs and recording devices
        # TODO: this function should probably not be part of the model
        # excitation_times, excitation_times_dict = self.__compute_timing_external_inputs(self.params['DeltaT'], 
        #                                                                                 self.params['DeltaT_seq'], 
        #                                                                                 self.params['DeltaT_cue'], 
        #                                                                                 self.params['excitation_start'], 
        #                                                                                 self.params['time_dend_to_somatic'])

    def connect(self):
        """Connects network and devices
        """

        print('\nConnecting network and devices...')

        # connect excitatory population (EE)
        if self.params['load_connections']:
            self.__load_connections(label='ee_connections')
        else:
            self.__connect_excitatory_neurons()

        # connect inhibitory population (II, EI, IE)
        self.__connect_RNN_neurons()

        # connect external input
        self.__connect_external_inputs_to_clusters()

        # connect neurons to the spike recorder
        self.__connect_neurons_to_spike_recorders()
        
        print('All nodes connected!')

    def simulate(self):
        """Run simulation.
        """

        sim_time = self.params['sim_time']

        # the simulation time is set during the creation of the network  
        if nest.Rank() == 0:
            print('\nSimulating {} ms.'.format(sim_time))

        #TODO: set actual loop, current is only for testing
        #nest.Prepare()
        for i in tqdm(range(1)):
            assert (i*sim_time) == nest.biological_time

            nest.Simulate(sim_time)
            #nest.Run(sim_time)

            # for generator_exc in self.external_node_to_exc_neuron_dict.values():
            #     generator_exc.stop += sim_time
            #     generator_exc.start += sim_time
            #     print(generator_exc.origin)

            # for generator_inh in self.external_node_to_inh_neuron_list:
            #     generator_inh.stop += sim_time
            #     generator_inh.start += sim_time

            for generators_to_exc in self.external_node_to_exc_neuron_dict.values():
                generators_to_exc[0].origin += sim_time
                generators_to_exc[1].origin += sim_time

            for generator_to_inh in self.external_node_to_inh_neuron_list:
                generator_to_inh.origin += sim_time
                
        #nest.Cleanup()

        #voltage_trace.from_device(self.voltmeter)
        #plt.show()
        

    def __create_RNN_populations(self):
        """'Create neuronal populations
        """

        # create excitatory population
        self.exc_neurons = nest.Create(self.params['exhibit_model'],
                                       self.num_exc_neurons,
                                       params=self.params['exhibit_params'])
        print(f"Create {self.num_exc_neurons=} excitatory neurons")
        print(nest.network_size)

        # create inhibitory population
        self.inh_neurons = nest.Create(self.params['inhibit_model'],
                                       self.num_inh_neurons,
                                       params=self.params['inhibit_params'])
        print(f"Create {self.num_inh_neurons=} inhibitory neurons")
        print(nest.network_size)

    def __create_spike_generators(self):
        """Create spike generators
        """
        self.external_node_to_exc_neuron_dict = {}
        self.external_node_to_inh_neuron_list = []
        cluster_stimulation_time = self.params['cluster_stimulation_time']
        stimulation_gap = self.params['stimulation_gap']

        #TODO: At the moment there is one Poisson generator for all exc_clusters per stimulation -> Constraint: rate is for all 4.5 -> paper shows contrdiction
        for stimulation_step in range(self.num_exc_clusters):
            external_input_per_step_list = []
            start = stimulation_step * (cluster_stimulation_time + stimulation_gap)
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time, rate=self.params['exh_rate_ex'])))
            external_input_per_step_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time, rate=self.params['inh_rate_ex'])))
            self.external_node_to_exc_neuron_dict[stimulation_step] = external_input_per_step_list
            self.external_node_to_inh_neuron_list.append(nest.Create('poisson_generator', params=dict(start=start, stop=start+cluster_stimulation_time, rate=self.params['exh_rate_ix'])))

        # TODO: set spike generator status with the above computed excitation times (see Younes code above)
        
    def __create_recording_devices(self):
        """Create recording devices
        """

        #TODO: Should the params dictionary also be in parameters_space?
        # create a spike recorder for exc neurons
        self.spike_recorder_soma = nest.Create('spike_recorder', params={'record_to': 'ascii',
                                                                         'label': 'exh_spikes'})
        print('Create 1 spike recorder for soma')
        print(nest.network_size)

        # create a spike recorder for inh neurons
        self.spike_recorder_inh = nest.Create('spike_recorder', params={'record_to': 'ascii',
                                                               'label': 'inh_spikes'})
        print('Create 1 spike recorder for inh')
        print(nest.network_size)

        self.spike_recorder_generator = nest.Create('spike_recorder', params={'record_to': 'ascii',
                                                               'label': 'generator_spikes'})

        self.voltmeter = nest.Create('voltmeter')

    def __connect_excitatory_neurons(self):
        """Connect excitatory neurons
        """

        nest.Connect(self.exc_neurons, self.exc_neurons, conn_spec=self.params['conn_dict_ee'],
                     syn_spec=self.params['syn_dict_ee'])

    def __connect_RNN_neurons(self):
        """Create II, EI, IE connections
        """
        # II connection
        nest.Connect(self.inh_neurons, self.inh_neurons, conn_spec=self.params['conn_dict_ii'], syn_spec=self.params['syn_dict_ii'])

        # EI connection
        nest.Connect(self.inh_neurons, self.exc_neurons, conn_spec=self.params['conn_dict_ei'], syn_spec=self.params['syn_dict_ei'])

        # IE connection
        nest.Connect(self.exc_neurons, self.inh_neurons, conn_spec=self.params['conn_dict_ie'], syn_spec=self.params['syn_dict_ie'])

    def __connect_external_inputs_to_clusters(self):
        """Connect external inputs to subpopulations
        """

        exc_cluster_size = self.params['exc_cluster_size']

        # connect to excitatory neurons
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
        
        # connect to inhibitory neurons
        for external_node in self.external_node_to_inh_neuron_list:
            nest.Connect(external_node, self.inh_neurons, conn_spec=self.params['conn_dict_ix'], syn_spec=self.params['syn_dict_ix'])

    def __connect_neurons_to_spike_recorders(self):
        """Connect excitatory neurons to spike recorders
        """
        nest.Connect(self.exc_neurons, self.spike_recorder_soma)
        nest.Connect(self.inh_neurons, self.spike_recorder_inh)
        #import pdb; pdb.set_trace()
        for i in range(self.num_exc_clusters):
            nest.Connect(self.external_node_to_exc_neuron_dict[i][0], self.spike_recorder_generator)
            nest.Connect(self.external_node_to_exc_neuron_dict[i][1], self.spike_recorder_generator)
            nest.Connect(self.external_node_to_inh_neuron_list[i], self.spike_recorder_generator)
        nest.Connect(self.voltmeter, self.exc_neurons)

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

        #connections_all = nest.GetConnections(synapse_model=self.params['syn_dict_ee']['synapse_model'])
        connections_all = nest.GetConnections(synapse_model=synapse_model)

    #TODO replace stdsp_synapse by clopath_synapse
        if synapse_model== 'stdsp_synapse':
            connections = nest.GetStatus(connections_all, ['target', 'source', 'weight', 'permanence'])
        else:
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

        if self.params['syn_dict_ee']['synapse_model'] == 'stdsp_synapse':
            conns_perms = [conn[3] for conn in conns]

        if self.params['evaluate_replay']:
            syn_dict = {'receptor_type': 2,
                        'delay': [self.params['syn_dict_ee']['delay']] * len(conns_weights),
                        'weight': conns_weights}
            nest.Connect(conns_src, conns_tg, 'one_to_one', syn_dict)
        else:
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
                            'weight': conns_weights,
                            'permanence': conns_perms}
            else:
                syn_dict = {'synapse_model': 'stdsp_synapse',
                            'receptor_type': 2,
                            'weight': conns_weights}

            nest.Connect(conns_src, conns_tg, 'one_to_one', syn_dict)

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