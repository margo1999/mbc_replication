import parameters as para
import numpy as np

DELAY = 0.1
TEST_NETWORK = True # The test network has fewer nodes than the orginal network, therefore it makes it faster to simulate and easier to debug

p = para.ParameterSpace({})

###################################################### general parameters for RNN ######################################################
if(not TEST_NETWORK):
    p['num_exc_neurons'] = 2400                                             # Number of recurrent E neurons
    p['num_inh_neurons'] = 600                                              # Number of recurrent I neurons
    p['num_exc_clusters'] = 80                                              # Number of exciatatory clusters
    p['exc_cluster_size'] = p['num_exc_neurons'] // p['num_exc_clusters']   # Number of excitatory neurons in one cluster
    p['connection_p'] = 0.2                                                 # Recurrent network connection probability
    print('The orginal size of the network is used!') 
else:    
    p['num_exc_neurons'] = 240                                              # Number of recurrent E neurons
    p['num_inh_neurons'] = 60                                               # Number of recurrent I neurons
    p['num_exc_clusters'] = 8                                               # Number of exciatatory clusters
    p['exc_cluster_size'] = p['num_exc_neurons'] // p['num_exc_clusters']   # Number of excitatory neurons in one cluster
    p['connection_p'] = 0.2                                                 # Recurrent network connection probability
    print('A smaller size of the network is used!')                                                 

###################################################### all node models ######################################################

#TODO: Replace model by actual model -> aeif_cond_alpha_clopath + different kernel
# Parameters of excitatory neurons
p['exhibit_model'] = 'aeif_psc_delta_clopath'
p['exhibit_params'] = {'A_LTD': 0.0014,         # LTD amplitude (pa/mV)
                    'A_LTP': 0.0008,            # LTP amplitude (pa/mV^2)
                    'C_m': 300.0,               # Capacitance of the membrane (pF)
                    'V_reset': -60.0,           # Reset potential (for all neurons the same) (mV)
                    'V_peak': -52.0,            # Spike detection threshold (mV)
                    'V_th_max': -52.0 + 10.0,   # Maximum threshold that can be reached (mV) - V_th_max = V_th_rest + A_T, where A_T is adaptive threshold increase constant          
                    'V_th_rest': -52.0,         # Threshold V_th is relaxing back to V_th_rest (mV) 
                    't_ref': 5.0,               # Absolute refactory period (ms)
                    'a': 0.0,                   # Subthreshold adaption (nS) - keep zero so it matches equation (3) in Maes et al. (2020)
                    'b':1000.0,                 # Adaption current increase constant (pA)
                    'Delta_T': 2.0,             # Exponential slope (mV)
                    'tau_w': 100.0,             # Adaption current time constant (ms)
                    'I_sp': 0.0,                # Depolarizing spike afterpotential current magnitude (pA)
                    'z': 0.0,                   # Spike-adaptation current (pA)
                    'I_e': 0.0,                 # Constant external input current (pA)
                    'E_L': -70.0,               # Leak reversal potential aka resting potential (mV)
                    'g_L': 300.0 / 20.0,        # Leak conductance (mV) - g_L = C_m / tau_E, where tau_E is membrane potential time constant
                    'tau_syn_ex': 1.0,          # Rise time of the excitatory synaptic alpha function aka rise time constant (ms)
                    'tau_syn_in': 6.0,          # Rise time of the inhibitory synaptic alpha function aka decay time constant(ms)
                    # 'V_m':                    # Membrane potential (mV) - TODO: Should V_m also be set?
                    }
assert(p['exhibit_params']['V_th_max'] == (p['exhibit_params']['V_th_rest']+ 10.0))
assert(p['exhibit_params']['g_L'] == (p['exhibit_params']['C_m'] / 20.0))

#TODO: Replace model by actual model -> iaf_cond_alpha + different kernel
# Parameters of inhibitory neurons
p['inhibit_model'] = 'iaf_cond_alpha'
p['inhibit_params'] = {'C_m': 300.0,            # Capacitance of the membrane (pF)
                    'E_L': -62.0,               # Leak reversal potential aka resting potential (mV)
                    'E_ex': 0.0,                # Excitatory reversal potential (mV)
                    'E_in': -75.0,              # Inhibitory reversal potential (mV)
                    'g_L': 300.0 / 20.0,        # Leak conductance (mV) - g_L = C_m / tau_E, where tau_I is membrane potential time constant
                    'I_e': 0.0,                 # Constant external input current (pA)          
                    'V_reset': -60.0,           # Reset potential (for all neurons the same) (mV)
                    'V_th': -52.0,              # Spike threshold (mV)
                    't_ref': 5.0,               # Absolute refactory period (ms)
                    'tau_syn_ex': 0.5,          # Rise time of the excitatory synaptic alpha function aka rise time constant (ms)
                    'tau_syn_in': 2.0,          # Rise time of the inhibitory synaptic alpha function aka decay time constant (ms)
                    # 'V_m':                    # Membrane potential (mV) - TODO: Should V_m also be set?
                    }
assert(p['inhibit_params']['g_L'] == (p['inhibit_params']['C_m'] / 20.0))

# TODO: External poisson generator for excitatory neurons
# Parameters of poisson generator to excitatory neurons
p['exh_rate_ex'] = 0.0                          # = 18000.0 Rate of external excitatory input to excitatory neurons (spikes/s)
p['inh_rate_ex'] = 0.0                          # = 4500.0 Rate of external inhibitory input to excitatory neurons (spikes/s)

# TODO: External poisson generator for inhibitory neurons
# Parameters of poisson generator to excitatory neurons
p['exh_rate_ix'] = 0.0                          # = 2250.0 Rate of external excitatory input to inhibitory neurons

###################################################### connection and synapse dictionaries ######################################################

# General connection dictionary
general_RNN_conn_dict = {'rule': 'pairwise_bernoulli',              # Connection rule
                        'p': p['connection_p'],                     # Connection probability of neurons in RNN
                        'allow_autapses': False,                    # If False then no self-connections are allowed
                        'allow_multapses': False                    # If False then only one connection between the neurons is allowed - TODO: check if this also disallows cycle between two nodes
                        }

# Parameters of excitatory EX synapses (external to E neurons)
p['syn_dict_ex_exc'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                        'weight': 1.6,                              # Synaptic weight (pF) - TODO: check if Farad is correct
                        }
p['conn_dict_ex_exc'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of inhibitory EX synapses (external to E neurons)
p['syn_dict_ex_inh'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                        'weight': -2.4                              # Synaptic weight (pF)
                        }
p['conn_dict_ex_inh'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of IX synapses (external to I neurons)
p['syn_dict_ix'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                        'weight': 1.52                              # Synaptic weight (pF)
                        }
p['conn_dict_ix'] = {'rule': 'all_to_all'}                          # Connection rule

# Parameters of EE synapses (voltage-based STDP)
p['syn_dict_ee'] = {'synapse_model': 'clopath_synapse',             # Name of synapse model - TODO: In the MATLAB code the weights might be randomized
                    'weight': 2.83,                                 # Initial synaptic weight (pF)
                    'Wmax': 32.68,                                  # Maximum allowed weight (pF)
                    'Wmin': 1.45,                                   # Minimum allowed weight (pF)
                    'delay': 1000                                   # Synaptic delay (ms)
                    }
p['conn_dict_ee'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

# parameters for II synapses
p['syn_dict_ii'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                    'weight': 20.91                                 # Synaptic weight (pF)
                    }
p['conn_dict_ii'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

# parameters for IE synapses 
p['syn_dict_ie'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                    'weight': 1.96                                  # Synpatic weight (pF)
                   }
p['conn_dict_ie'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

#TODO: check if synapse model is correct + need to add Wmin but for synpase implementation there is no Wmin key in dict
# parameters for EI synapses 
p['syn_dict_ei'] = {'synapse_model': 'vogels_sprekeler_synapse',    # Name of synapse model
                    'weight': 62.87,                                # Initial synpatic weight (pF)
                    'Wmax': 243.0,                                  # Maximum allowed weight (pF)
#                    'Wmin': 48.7                                   # Minimum allowed weight (pF)
                   }
p['conn_dict_ei'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

###################################################### simulation parameters ######################################################

# simulation parameters 
p['dt'] = 0.1                                                                                   # Simulation time resolution (ms)
p['overwrite_files'] = True                                                                     # True: data will be overwritten; False: a NESTError is raised if the files already exist
p['seed'] = para.ParameterRange([1])                                                            # Seed for NEST
p['print_simulation_progress'] = False                                                          # Print the time progress
p['n_threads'] = 2                                                                              # Number of threads per MPI process 
p['pad_time'] = 5.                                                                              # TODO: What is this?
p['idend_recording_interval'] = 10 * p['dt']                                                    # Dendritic current recording resolution TODO: Do we need this?
p['idend_record_time'] = 8.                                                                     # Time interval after the external stimulation at which the dendritic current is recorded TODO: Do we need this?
p['evaluate_performance'] = True                                                                # True: we monitor the dendritic current at a certain time steps during the simulation. This then is used for the prediction performance assessment
p['evaluate_replay'] = False                                                                    # TODO: What is this?  
p['record_idend_last_episode'] = True                                                           # Used for debugging, if turned on we record the dendritic current of all neurons this can consume too much memory TODO: Does this take too much time?
p['store_connections'] = True                                                                   # Stores connection in a seperate file
p['load_connections'] = False                                                                   # Loads connection from existing file
#p['sparse_first_char'] = False                                                                 # If turned on, the dAP of a subset of neurons in the subpopulation representing 
                                                                                                # First sequence elements is activated externally 
p['active_weight_recorder'] = False                                                             # True: the weights are recorded every presynaptic spike
p['cluster_stimulation_time'] = 10.0                                                            # Stimulation time from external input to excitatory cluster
p['stimulation_gap'] = 5.0                                                                      # Gap between to stimulations of excitatory clusters  
p['sim_time'] = p['num_exc_clusters'] * (p['cluster_stimulation_time'] + p['stimulation_gap'])  #simulation time for one round
#p['sim_time'] = p['num_exc_clusters'] * p['cluster_stimulation_time'] + (p['num_exc_clusters'] - 1) * p['stimulation_gap'] 

###################################################### data path dict ######################################################

# Simulation results such as spike times and connection weights are stored in clock_net/data/sequence_learning_performance/sequence_learning_and_prediction
p['data_path'] = {}
p['data_path']['data_root_path'] = 'data'
p['data_path']['project_name'] = 'sequence_learning_performance'
p['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction'

###################################################### task parameters ######################################################

p['task'] = {}
p['task']['task_name'] = 'hard_coded'          # Name of the task
p['task']['task_type'] = 1                     # This chooses between three hard coded sequence sets (see ./utils.py)
p['task']['vocab_size'] = 6                    # Vocabulary size
p['task']['seed'] = 111                        # Seed number
p['task']['store_training_data'] = True        # If turned on, the sequence set is stored in directory defined in dict data_path
if p['task']['task_name'] != 'hard_coded':
    p['task']['num_sequences'] = 2             # Number of sequences per sequence set
    p['task']['num_sub_seq'] = 2               # If task_name == 'high_order', 
                                               # It sets the number of sequences with same shared subsequence
    p['task']['length_sequence'] = 6           # Number of elements per sequence
    p['task']['replace'] = False               # Random choice of characters with replacement

###################################################### REST ######################################################

# # setup the training loop  
# p['learning_episodes'] = 85                     # Total number of training episodes ('repetitions of the sequence sets')
# p['episodes_to_testing'] = 1                    # Number of episodes after which we measure the prediction perfomance

#TODO: Do I need this?
# Stimulus parameters
p['DeltaT'] = 40.                     # Inter-stimulus interval
p['excitation_start'] = 30.           # Time at which the external stimulation begins
p['time_dend_to_somatic'] = 20.       # Time between the dAP activation and the somatic activation (only used if sparse_first_char is True)   
p['DeltaT_cue'] = 80.                 # Inter-cue interval during replay
