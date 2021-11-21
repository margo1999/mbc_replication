"""PyNEST Clock Network: Model Class [1]
----------------------------------------

Main file of the Clock Network defining ``Model`` class with function
to build, connect and simulate the network.

Authors
~~~~~~~
Jette Oberländer, Younes Bouhadjar
"""

import random
import nest
import copy
import numpy as np
from collections import defaultdict

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

        if nest.Rank() == 0:
            if self.data_path.is_dir():
                message = "Directory already existed."
                if self.params['overwrite_files']:
                    message += "Old data will be overwritten."
            else:
                self.data_path.mkdir(parents=True, exist_ok=True)
                message = "Directory has been created."
            print("Data will be written to: {}\n{}\n".format(self.data_path, message))

        # set network size
        self.num_subpopulations = params['M']
        self.num_exc_neurons = params['n_E'] * self.num_subpopulations

        # initialize RNG        
        np.random.seed(self.params['seed'])
        random.seed(self.params['seed'])

        # input stream: sequence data
        self.sequences = sequences
        self.vocabulary = vocabulary
        self.length_sequence = len(self.sequences[0])
        self.num_sequences = len(self.sequences)

        # initialize the NEST kernel
        self.__setup_nest()

        # get time constant for dendriticAP rate
        self.params['soma_params']['tau_h'] = self.__get_time_constant_dendritic_rate(
            calibration=self.params['calibration'])

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

        # create excitatory population
        self.__create_neuronal_populations()

        # compute timing of the external inputs and recording devices
        # TODO: this function should probably not be part of the model
        excitation_times, excitation_times_dict = self.__compute_timing_external_inputs(self.params['DeltaT'], 
                                                                                        self.params['DeltaT_seq'], 
                                                                                        self.params['DeltaT_cue'], 
                                                                                        self.params['excitation_start'], 
                                                                                        self.params['time_dend_to_somatic'])

        # create spike generators
        self.__create_spike_generators(excitation_times_dict)

        # create recording devices
        self.__create_recording_devices(excitation_times)

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
        self.__connect_inhibitory_neurons()

        # connect external input
        self.__connect_external_inputs_to_subpopulations()

        # connect neurons to the spike recorder
        nest.Connect(self.exc_neurons, self.spike_recorder_soma)
        nest.Connect(self.inh_neurons, self.spike_recorder_inh)

        # set min synaptic strength
        #self.__set_min_synaptic_strength()

    def simulate(self):
        """Run simulation.
        """

        # the simulation time is set during the creation of the network  
        if nest.Rank() == 0:
            print('\nSimulating {} ms.'.format(self.sim_time))

        nest.Simulate(self.sim_time)

    def __create_neuronal_populations(self):
        """'Create neuronal populations
        """

        # create excitatory population
        self.exc_neurons = nest.Create(self.params['soma_model'],
                                       self.num_exc_neurons,
                                       params=self.params['soma_params'])

        # create inhibitory population
        self.inh_neurons = nest.Create(self.params['inhibit_model'],
                                       self.params['n_I'] * self.num_subpopulations,
                                       params=self.params['inhibit_params'])

    def __create_spike_generators(self, excitation_times_dict):
        """Create spike generators
        """

        self.input_excitation_dict = {}
        for char in self.vocabulary:
            self.input_excitation_dict[char] = nest.Create('spike_generator')

        # set spike generator status with the above computed excitation times
        for char in self.vocabulary:
            nest.SetStatus(self.input_excitation_dict[char], {'spike_times': excitation_times_dict[char]})

    def __create_recording_devices(self, excitation_times):
        """Create recording devices
        """

        # create a spike recorder for exc neurons
        self.spike_recorder_soma = nest.Create('spike_recorder', params={'record_to': 'ascii',
                                                                         'label': 'somatic_spikes'})

        # create a spike recorder for inh neurons
        self.spike_recorder_inh = nest.Create('spike_recorder', params={'record_to': 'ascii',
                                                               'label': 'inh_spikes'})

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

    def __get_subpopulation_neurons(self, index_subpopulation):
        """Get neuron's indices (NEST NodeCollection) belonging to a subpopulation
        
        Parameters
        ---------
        index_subpopulation: int

        Returns
        -------
        NEST NodeCollection
        """

        neurons_indices = [int(index_subpopulation) * self.params['n_E'] + i for i in
                           range(self.params['n_E'])]

        return self.exc_neurons[neurons_indices]

    def __connect_excitatory_neurons(self):
        """Connect excitatory neurons
        """

        nest.Connect(self.exc_neurons, self.exc_neurons, conn_spec=self.params['conn_dict_ee'],
                     syn_spec=self.params['syn_dict_ee'])

    def __connect_inhibitory_neurons(self):
        """Connect inhibitory neurons
        """

        for k, subpopulation_index in enumerate(range(self.num_subpopulations)):
            # connect inhibitory population 
            subpopulation_neurons = self.__get_subpopulation_neurons(subpopulation_index)

            # connect neurons within the same mini-subpopulation to the inhibitory population
            nest.Connect(subpopulation_neurons, self.inh_neurons[k], syn_spec=self.params['syn_dict_ie'])

            # connect the inhibitory neurons to the neurons within the same mini-subpopulation
            nest.Connect(self.inh_neurons[k], subpopulation_neurons, syn_spec=self.params['syn_dict_ei'])

    def __connect_external_inputs_to_subpopulations(self):
        """Connect external inputs to subpopulations
        """

        # get input encoding
        self.characters_to_subpopulations = self.__stimulus_preference(fname='characters_to_subpopulations')

        # save characters_to_subpopulations for evaluation
        if self.params['evaluate_performance'] or self.params['evaluate_replay']:
            fname = 'characters_to_subpopulations'
            np.save('%s/%s' % (self.data_path, fname), self.characters_to_subpopulations)

        for char in self.vocabulary:
            subpopulations_indices = self.characters_to_subpopulations[char]

            # receptor type 1 correspond to the feedforward synapse of the 'iaf_psc_exp_multisynapse' model
            for subpopulation_index in subpopulations_indices:
                subpopulation_neurons = self.__get_subpopulation_neurons(subpopulation_index)
                nest.Connect(self.input_excitation_soma[char], subpopulation_neurons,
                             self.params['conn_dict_ex'], syn_spec=self.params['syn_dict_ex'])

    def __stimulus_preference(self, fname='characters_to_subpopulations'):
        """Assign a subset of subpopulations to a each element in the vocabulary.

        Parameters
        ----------
        fname : str

        Returns
        -------
        characters_to_subpopulations: dict
        """

        if len(self.vocabulary) * self.params['L'] > self.num_subpopulations:
            raise ValueError(
                "num_subpopulations needs to be large than length_user_characters*num_subpopulations_per_character")

        characters_to_subpopulations = defaultdict(list)  # a dictionary that assigns mini-subpopulation to characters

        subpopulation_indices = np.arange(self.num_subpopulations)
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

    def save_connections(self, fname='ee_connections'):
        """Save connection matrix

        Parameters
        ----------
        label: str
            name of the stored file
        """

        print('\nSave connections ...')
        connections_all = nest.GetConnections(synapse_model=self.params['syn_dict_ee']['synapse_model'])

        if self.params['syn_dict_ee']['synapse_model'] == 'stdsp_synapse':
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
