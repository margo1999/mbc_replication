""" TODO
    Here all parameters are set that are necessary for learning sequences and for replaying. This corresponds to the parameters for neuron and synapse
    models within the read-out layer, its spike recorders and spike generators. In addition, the connection between recurrent network and read-out
    layer is specified and details of the sequence to be learned are set.Creates a set of multiple points in parameter space by using ParameterRange
    objects. Creating ParameterRange objects allows to run the simulation with various combinations of a parameter.
"""
import parameters as para
import numpy as np

RESOLUTION = 0.1
TEST_NETWORK = False  # Test network has fewer nodes than the orginal network, therefore it makes it faster to simulate and easier to debug
SEED = 1

np.random.seed(SEED)

# ==============================================================================================================================
#                                   Parameterspace for recurrent network
# ==============================================================================================================================

param_recurrent = para.ParameterSpace({})

param_recurrent['read_out_off'] = False
# ========================================== general parameters for RNN ==========================================
if not TEST_NETWORK:
    param_recurrent['num_exc_neurons'] = 2400                                             # Number of recurrent E neurons
    param_recurrent['num_inh_neurons'] = 600                                              # Number of recurrent I neurons
    param_recurrent['num_exc_clusters'] = 30                                              # Number of exciatatory clusters
    param_recurrent['exc_cluster_size'] = param_recurrent['num_exc_neurons'] // param_recurrent['num_exc_clusters']   # Number of excitatory neurons in one cluster
    param_recurrent['connection_p'] = 0.2                                                 # Recurrent network connection probability
    print('\nThe orginal size of the network is used!')
else:
    param_recurrent['num_exc_neurons'] = 240                                              # Number of recurrent E neurons
    param_recurrent['num_inh_neurons'] = 60                                               # Number of recurrent I neurons
    param_recurrent['num_exc_clusters'] = 8                                               # Number of exciatatory clusters
    param_recurrent['exc_cluster_size'] = param_recurrent['num_exc_neurons'] // param_recurrent['num_exc_clusters']   # Number of excitatory neurons in one cluster
    param_recurrent['connection_p'] = 0.2                                                 # Recurrent network connection probability
    print('\nA smaller size of the network is used!')

# ========================================== all node models ==========================================

# Parameters of excitatory neurons
V_m_exc = np.random.uniform(low=-60.0, high=-52, size=240)  # TODO make this generalized and add to dictionary
param_recurrent['exhibit_model'] = 'aeif_cond_diff_exp_clopath'
param_recurrent['exhibit_params'] = {'A_LTD': 0.00014,        # LTD amplitude (pa/mV)
                                     'A_LTP': 0.0008,            # LTP amplitude (pa/mV^2)
                                     'C_m': 300.0,               # Capacitance of the membrane (pF)
                                     'V_reset': -60.0,           # Reset potential (for all neurons the same) (mV)
                                     'V_peak': 20.0,             # Spike detection threshold (mV)
                                     'V_th_max': -52.0 + 10.0,   # Maximum threshold that can be reached (mV) - V_th_max = V_th_rest + A_T, where A_T is adaptive threshold increase constant
                                     'V_th_rest': -52.0,         # Threshold V_th is relaxing back to V_th_rest (mV)
                                     't_ref': 5.0,               # Absolute refactory period (ms)
                                     'a': 0.0,                   # Subthreshold adaption (nS) - keep zero so it matches equation (3) in Maes et al. (2020)
                                     'b': 1000.0,                # Adaption current increase constant (pA)
                                     'Delta_T': 2.0,             # Exponential slope (mV)
                                     'tau_w': 100.0,             # Adaption current time constant (ms)
                                     'I_e': 0.0,                 # Constant external input current (pA)
                                     'E_L': -70.0,               # Leak reversal potential aka resting potential (mV)
                                     'g_L': 300.0 / 20.0,        # Leak conductance (mV) - g_L = C_m / tau_E, where tau_E is membrane potential time constant
                                     'tau_syn_ex_decay': 6.0,    # Decay time of the excitatory synaptic difference of exp functions (ms)
                                     'tau_syn_ex_rise': 1.0,     # Rise time of the excitatory synaptic difference of exp functions (ms)
                                     'tau_syn_in_decay': 2.0,    # Decay time of the inhibitory synaptic difference of exp functions (ms)
                                     'tau_syn_in_rise': 0.5,     # Rise time of the inhibitory synaptic difference of exp functions (ms)
                                     'E_ex': 0.0,                # Excitatory reversal potential (mV)
                                     'E_in': -75.0,              # Inhibitory reversal potential (mV)
                                     'tau_V_th': 30.0,           # Apaptive Threshold time constant (ms)
                                     'theta_minus': -70.0,       # LTD threshold (mV)
                                     'theta_plus': -49.0,        # LTP threshold (mV)
                                     'tau_u_bar_minus': 10.0,    # Time constant of low pass filtered postsynaptic membrane potential (LTD) (ms)
                                     'tau_u_bar_plus': 7.0,      # Time constant of low pass filtered postsynaptic membrane potential (LTP) (ms)
                                     'V_m': -60.0,               # Initial Membrane potential (mV)
                                     'u_bar_minus': 0.0,         # Initial low-pass filtered membrane potential
                                     'u_bar_plus': 0.0,          # Initial low-pass filtered membrane potential
                                     'delay_u_bars': RESOLUTION  # Delay with which u_bar_[plus/minus] are processed to compute the synaptic weights
                                     }
assert param_recurrent['exhibit_params']['V_th_max'] == (param_recurrent['exhibit_params']['V_th_rest'] + 10.0)
assert param_recurrent['exhibit_params']['g_L'] == (param_recurrent['exhibit_params']['C_m'] / 20.0)

# Parameters of inhibitory neurons
V_m_inh = np.random.uniform(low=-60.0, high=-52, size=60)  # TODO make this generalized and add to dictionary
param_recurrent['inhibit_model'] = 'iaf_cond_diff_exp'
param_recurrent['inhibit_params'] = {'C_m': 300.0,            # Capacitance of the membrane (pF)
                                     'E_L': -62.0,               # Leak reversal potential aka resting potential (mV)
                                     'E_ex': 0.0,                # Excitatory reversal potential (mV)
                                     'E_in': -75.0,              # Inhibitory reversal potential (mV)
                                     'g_L': 300.0 / 20.0,        # Leak conductance (mV) - g_L = C_m / tau_E, where tau_I is membrane potential time constant
                                     'I_e': 0.0,                 # Constant external input current (pA)
                                     'V_reset': -60.0,           # Reset potential (for all neurons the same) (mV)
                                     'V_th': -52.0,              # Spike threshold (mV)
                                     't_ref': 5.0,               # Absolute refactory period (ms)
                                     'tau_syn_ex_decay': 6.0,    # Decay time of the excitatory synaptic difference of exp functions (ms)
                                     'tau_syn_ex_rise': 1.0,     # Rise time of the excitatory synaptic difference of exp functions (ms)
                                     'tau_syn_in_decay': 2.0,    # Decay time of the inhibitory synaptic difference of exp functions (ms)
                                     'tau_syn_in_rise': 0.5,     # Rise time of the inhibitory synaptic difference of exp functions (ms)
                                     'V_m': -60.0                # Membrane potential (mV)
                                     }
assert param_recurrent['inhibit_params']['g_L'] == (param_recurrent['inhibit_params']['C_m'] / 20.0)

# ========================================== poisson generator rates ==========================================

# Parameters of poisson generator to excitatory neurons
param_recurrent['exh_rate_ex'] = 18000.0 + 4500.0                 # Rate of external excitatory input to excitatory neurons (spikes/s)
param_recurrent['inh_rate_ex'] = 4500.0                           # Rate of external inhibitory input to excitatory neurons (spikes/s)

# Rates discribed in the paper but differ from original implementation
param_recurrent['exh_rate_ex_old'] = 18000.0                      # Rate of external excitatory input to excitatory neurons (spikes/s)
param_recurrent['inh_rate_ex_old'] = 4500.0                       # Rate of external inhibitory input to excitatory neurons (spikes/s)

# Parameters of poisson generator to excitatory neurons
param_recurrent['exh_rate_ix'] = 2250.0                           # Rate of external excitatory input to inhibitory neurons (spikes/s)

# Random dynamics for excitatory and inhibitory neurons
param_recurrent['random_dynamics_ex'] = 4500.0                    # Rate of external excitatory input to excitatory neurons (spikes/s)
param_recurrent['random_dynamics_ix'] = 2250.0                    # Rate of external excitatory input to inhibitory neurons (spikes/s)

# ========================================== connection and synapse dictionaries ==========================================

# General connection dictionary
general_RNN_conn_dict = {'rule': 'pairwise_bernoulli',              # Connection rule
                         'p': param_recurrent['connection_p'],                     # Connection probability of neurons in RNN
                         'allow_autapses': False,                    # If False then no self-connections are allowed
                         'allow_multapses': False                    # If False then only one connection between the neurons is allowed
                         }

# Parameters of excitatory EX synapses (external to E neurons)
param_recurrent['syn_dict_ex_exc'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                      'weight': 1.6,                              # Synaptic weight (pF)
                                      'delay': RESOLUTION                         # Synaptic delay (ms)
                                      }
param_recurrent['conn_dict_ex_exc'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of inhibitory EX synapses (external to E neurons)
param_recurrent['syn_dict_ex_inh'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                      'weight': -0.8,                             # Synaptic weight (pF)
                                      'delay': RESOLUTION                         # Synaptic delay (ms)
                                      }
param_recurrent['conn_dict_ex_inh'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of excitatory EX synapses (external to E neurons) during random dynamics
param_recurrent['syn_dict_ex_random'] = {'synapse_model': 'static_synapse',       # Name of synapse model
                                         'weight': 1.6,                              # Synaptic weight (pF)
                                         'delay': RESOLUTION                         # Synaptic delay (ms)
                                         }
param_recurrent['conn_dict_ex_random'] = {'rule': 'all_to_all'}                   # Connection rule

# Parameters of IX synapses (external to I neurons)
param_recurrent['syn_dict_ix'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                                  'weight': 1.52,                             # Synaptic weight (pF)
                                  'delay': RESOLUTION                         # Synaptic delay (ms)
                                  }
param_recurrent['conn_dict_ix'] = {'rule': 'all_to_all'}                          # Connection rule

# Parameters of IX synapses (external to I neurons)
param_recurrent['syn_dict_ix_random'] = {'synapse_model': 'static_synapse',       # Name of synapse model
                                         'weight': 1.52,                             # Synaptic weight (pF)
                                         'delay': RESOLUTION                         # Synaptic delay (ms)
                                         }
param_recurrent['conn_dict_ix_random'] = {'rule': 'all_to_all'}                   # Connection rule

# Parameters of EE synapses (voltage-based STDP)
param_recurrent['syn_dict_ee'] = {'synapse_model': 'clopath_synapse',             # Name of synapse model
                                  'weight': 2.83,                                 # Initial synaptic weight (pF)
                                  'Wmax': 32.68,                                  # Maximum allowed weight (pF)
                                  'Wmin': 1.45,                                   # Minimum allowed weight (pF)
                                  'tau_x': 3.5,                                   # Time constant of low pass filtered presynaptic spike train in recurrent network (ms)
                                  'delay': RESOLUTION                             # Synaptic delay (ms)
                                  }
param_recurrent['conn_dict_ee'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

# parameters for II synapses
param_recurrent['syn_dict_ii'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                                  'weight': - 20.91,                              # Synaptic weight (pF)
                                  'delay': RESOLUTION                             # Synaptic delay (ms)
                                  }
param_recurrent['conn_dict_ii'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

# parameters for IE synapses
param_recurrent['syn_dict_ie'] = {'synapse_model': 'static_synapse',              # Name of synapse model
                                  'weight': 1.96,                                 # Synpatic weight (pF)
                                  'delay': RESOLUTION                             # Synaptic delay (ms)
                                  }
param_recurrent['conn_dict_ie'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

# parameters for EI synapses
# p['syn_dict_ei'] = {'synapse_model': 'static_synapse',              # Name of synapse model
#                     'weight': 0.0                                 # Synaptic weight (pF)
#                     }
param_recurrent['syn_dict_ei'] = {'synapse_model': 'vogels_sprekeler_synapse',    # Name of synapse model
                                  'weight': - 62.87,                              # Initial synpatic weight (pF)
                                  'eta': 1.0,                                     # Learning rate/ Amplitude of inhibitory plasticity - INCONSISTENCY: orginal implementation: eta=1.0 & paper: eta=Ainh=10^-5
                                  'alpha': 2.0 * 0.003 * 20.0,                    # Constant depression set (= 2 * r_0 * tau_y = 2 * target_firing_rate * tau) # TODO: Unit of target_firing_rate might be wrong!
                                  'Wmax': -243.0,                                 # Maximum allowed weight (pF)
                                  'Wmin': -48.7,                                  # Minimum allowed weight (pF)
                                  'delay': RESOLUTION,                            # Synaptic delay (ms)
                                  'tau': 20.0                                     # Time constant of low pass filtered spike train
                                  }
param_recurrent['conn_dict_ei'] = general_RNN_conn_dict                           # Connection dictionary for RNN neurons

# ========================================== simulation parameters ==========================================

# simulation parameters
param_recurrent['dt'] = RESOLUTION                                                                                # Simulation time resolution (ms)
param_recurrent['overwrite_files'] = True                                                                         # True: data will be overwritten; False: a NESTError is raised if the files already exist
param_recurrent['seed'] = SEED  # para.ParameterRange([1])                                                          # Seed for NEST
param_recurrent['print_simulation_progress'] = False                                                              # Print the time progress
param_recurrent['n_threads'] = 1                                                                                  # Number of threads per MPI process
param_recurrent['idend_record_time'] = 8.                                                                         # Time interval after the external stimulation at which the dendritic current is recorded TODO: Do we need this?
param_recurrent['evaluate_performance'] = True                                                                    # True: we monitor the dendritic current at a certain time steps during the simulation. This then is used for the prediction performance assessment
param_recurrent['evaluate_replay'] = False                                                                        # TODO: What is this?
param_recurrent['store_connections'] = True                                                                       # Stores connection in a seperate file (bool)
param_recurrent['load_connections'] = False                                                                        # Loads connection from existing file (bool)
param_recurrent['cluster_stimulation_time'] = 10.0                                                                # Stimulation time from external input to excitatory cluster (ms)
param_recurrent['stimulation_gap'] = 5.0                                                                          # Gap between to stimulations of excitatory clusters (ms)
param_recurrent['round_time'] = param_recurrent['num_exc_clusters'] * (param_recurrent['cluster_stimulation_time'] + param_recurrent['stimulation_gap'])    # Simulation time for one round (ms)
param_recurrent['training_iterations'] = 30                                                                       # Indicates how many iterations there are during the training phase. One iteration corresponds to approximately 2 minutes. (int)
param_recurrent['normalization_time'] = para.ParameterRange([20.0, 50.0, 100.0, 200.0, 225.0, 450.0])             # Time after normalization is necessary (ms)
param_recurrent['random_dynamics'] = True                                                                         # If turned on, a phase of spontaneous dynamics follows after training phase (bool)
param_recurrent['random_dynamics_time'] = 1.0 * 60.0 * 60.0 * 1000.0                                              # Time of spontaneous dynamics (ms)

# ========================================== data path dict ==========================================

# Simulation results such as spike times and connection weights are stored in clock_net/data/sequence_learning_performance/sequence_learning_and_prediction
param_recurrent['data_path'] = {}
param_recurrent['data_path']['data_root_path'] = 'data'
param_recurrent['data_path']['project_name'] = 'sequence_learning_performance'
param_recurrent['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction'

# ========================================== task parameters ==========================================

param_recurrent['task'] = {}
param_recurrent['task']['task_name'] = 'hard_coded'          # Name of the task ['high_order', 'random', 'structure', 'hard_coded']
param_recurrent['task']['task_type'] = 1                     # This chooses between three hard coded sequence sets (see ./utils.py)
param_recurrent['task']['vocab_size'] = 6                    # Vocabulary size
param_recurrent['task']['seed'] = 111                        # Seed number
param_recurrent['task']['store_training_data'] = True        # If turned on, the sequence set is stored in directory defined in dict data_path
if param_recurrent['task']['task_name'] != 'hard_coded':
    param_recurrent['task']['num_sequences'] = 2             # Number of sequences per sequence set
    param_recurrent['task']['num_sub_seq'] = 2               # If task_name == 'high_order', It sets the number of sequences with same shared subsequence
    param_recurrent['task']['length_sequence'] = 6           # Number of elements per sequence
    param_recurrent['task']['replace'] = False               # Random choice of characters with replacement

# ========================================== REST ==========================================

# # setup the training loop
# p['learning_episodes'] = 85                     # Total number of training episodes ('repetitions of the sequence sets')
# p['episodes_to_testing'] = 1                    # Number of episodes after which we measure the prediction perfomance

# TODO: Do I need this?
# Stimulus parameters
param_recurrent['DeltaT'] = 40.                     # Inter-stimulus interval
param_recurrent['excitation_start'] = 30.           # Time at which the external stimulation begins
param_recurrent['time_dend_to_somatic'] = 20.       # Time between the dAP activation and the somatic activation (only used if sparse_first_char is True)
param_recurrent['DeltaT_cue'] = 80.                 # Inter-cue interval during replay

# ==============================================================================================================================
#                                   Parameterspace for read-out layer
# ==============================================================================================================================

param_readout = para.ParameterSpace({})
param_readout['param_recurrent'] = param_recurrent

# ========================================== all node models ==========================================

# Parameters of read-out neurons
V_m_exc = np.random.uniform(low=-60.0, high=-52, size=240)  # TODO make this generalized and add to dictionary
param_readout['read_out_model'] = 'aeif_cond_diff_exp_clopath'
param_readout['read_out_params'] = {'A_LTD': 0.00014,           # LTD amplitude (pa/mV)
                                    'A_LTP': 0.0008,            # LTP amplitude (pa/mV^2)
                                    'C_m': 300.0,               # Capacitance of the membrane (pF)
                                    'V_reset': -60.0,           # Reset potential (for all neurons the same) (mV)
                                    'V_peak': 20.0,             # Spike detection threshold (mV)
                                    'V_th_max': -52.0 + 10.0,   # Maximum threshold that can be reached (mV) - V_th_max = V_th_rest + A_T, where A_T is adaptive threshold increase constant
                                    'V_th_rest': -52.0,         # Threshold V_th is relaxing back to V_th_rest (mV)
                                    't_ref': 1.0,               # Absolute refactory period for R neurons(ms)
                                    'a': 0.0,                   # Subthreshold adaption (nS) - keep zero so it matches equation (3) in Maes et al. (2020)
                                    'b': 0.0,                   # Adaption current increase constant (pA)
                                    'Delta_T': 2.0,             # Exponential slope (mV)
                                    'tau_w': 10.0,              # Adaption current time constant (ms) (does not affect the dynamics, since the neuron does not have an adaption current)
                                    'I_e': 0.0,                 # Constant external input current (pA)
                                    'E_L': -70.0,               # Leak reversal potential aka resting potential (mV)
                                    'g_L': 300.0 / 20.0,        # Leak conductance (mV) - g_L = C_m / tau_E, where tau_E is membrane potential time constant
                                    'tau_syn_ex_decay': 6.0,    # Decay time of the excitatory synaptic difference of exp functions (ms)
                                    'tau_syn_ex_rise': 1.0,     # Rise time of the excitatory synaptic difference of exp functions (ms)
                                    'tau_syn_in_decay': 2.0,    # Decay time of the inhibitory synaptic difference of exp functions (ms)
                                    'tau_syn_in_rise': 0.5,     # Rise time of the inhibitory synaptic difference of exp functions (ms)
                                    'E_ex': 0.0,                # Excitatory reversal potential (mV)
                                    'E_in': -75.0,              # Inhibitory reversal potential (mV)
                                    'tau_V_th': 30.0,           # Apaptive Threshold time constant (ms)
                                    'theta_minus': -70.0,       # LTD threshold (mV)
                                    'theta_plus': -49.0,        # LTP threshold (mV)
                                    'tau_u_bar_minus': 10.0,    # Time constant of low pass filtered postsynaptic membrane potential (LTD) (ms)
                                    'tau_u_bar_plus': 7.0,      # Time constant of low pass filtered postsynaptic membrane potential (LTP) (ms)
                                    'V_m': -60.0,               # Initial Membrane potential (mV)
                                    'u_bar_minus': -60.0,       # Initial low-pass filtered membrane potential
                                    'u_bar_plus': -60.0,        # Initial low-pass filtered membrane potential
                                    'delay_u_bars': RESOLUTION  # Delay with which u_bar_[plus/minus] are processed to compute the synaptic weights
                                    }
assert param_readout['read_out_params']['V_th_max'] == (param_readout['read_out_params']['V_th_rest'] + 10.0)
assert param_readout['read_out_params']['g_L'] == (param_readout['read_out_params']['C_m'] / 20.0)

# Parameters of supervisor neurons
V_m_exc = np.random.uniform(low=-60.0, high=-52, size=240)  # TODO make this generalized and add to dictionary
param_readout['supervisor_model'] = 'aeif_cond_diff_exp_clopath'
param_readout['supervisor_params'] = {'A_LTD': 0.00014,           # LTD amplitude (pa/mV)
                                      'A_LTP': 0.0008,            # LTP amplitude (pa/mV^2)
                                      'C_m': 300.0,               # Capacitance of the membrane (pF)
                                      'V_reset': -60.0,           # Reset potential (for all neurons the same) (mV)
                                      'V_peak': 20.0,             # Spike detection threshold (mV)
                                      'V_th_max': -52.0 + 10.0,   # Maximum threshold that can be reached (mV) - V_th_max = V_th_rest + A_T, where A_T is adaptive threshold increase constant
                                      'V_th_rest': -52.0,         # Threshold V_th is relaxing back to V_th_rest (mV)
                                      't_ref': 1.0,               # Absolute refactory period for S neurons(ms)
                                      'a': 0.0,                   # Subthreshold adaption (nS) - keep zero so it matches equation (3) in Maes et al. (2020)
                                      'b': 0.0,                   # Adaption current increase constant (pA)
                                      'Delta_T': 2.0,             # Exponential slope (mV)
                                      'tau_w': 10.0,              # Adaption current time constant (ms) (does not affect the dynamics, since the neuron does not have an adaption current)
                                      'I_e': 0.0,                 # Constant external input current (pA)
                                      'E_L': -70.0,               # Leak reversal potential aka resting potential (mV)
                                      'g_L': 300.0 / 20.0,        # Leak conductance (mV) - g_L = C_m / tau_E, where tau_E is membrane potential time constant
                                      'tau_syn_ex_decay': 6.0,    # Decay time of the excitatory synaptic difference of exp functions (ms)
                                      'tau_syn_ex_rise': 1.0,     # Rise time of the excitatory synaptic difference of exp functions (ms)
                                      'tau_syn_in_decay': 2.0,    # Decay time of the inhibitory synaptic difference of exp functions (ms)
                                      'tau_syn_in_rise': 0.5,     # Rise time of the inhibitory synaptic difference of exp functions (ms)
                                      'E_ex': 0.0,                # Excitatory reversal potential (mV)
                                      'E_in': -75.0,              # Inhibitory reversal potential (mV)
                                      'tau_V_th': 30.0,           # Apaptive Threshold time constant (ms)
                                      'theta_minus': -70.0,       # LTD threshold (mV)
                                      'theta_plus': -49.0,        # LTP threshold (mV)
                                      'tau_u_bar_minus': 10.0,    # Time constant of low pass filtered postsynaptic membrane potential (LTD) (ms)
                                      'tau_u_bar_plus': 7.0,      # Time constant of low pass filtered postsynaptic membrane potential (LTP) (ms)
                                      'V_m': -60.0,               # Initial Membrane potential (mV)
                                      'u_bar_minus': -60.0,       # Initial low-pass filtered membrane potential
                                      'u_bar_plus': -60.0,        # Initial low-pass filtered membrane potential
                                      'delay_u_bars': RESOLUTION  # Delay with which u_bar_[plus/minus] are processed to compute the synaptic weights
                                      }
assert param_readout['supervisor_params']['V_th_max'] == (param_readout['supervisor_params']['V_th_rest'] + 10.0)
assert param_readout['supervisor_params']['g_L'] == (param_readout['supervisor_params']['C_m'] / 20.0)

# Parameters of interneurons
V_m_inh = np.random.uniform(low=-60.0, high=-52, size=60)  # TODO make this generalized and add to dictionary
param_readout['interneuron_model'] = 'iaf_cond_diff_exp'
param_readout['interneuron_params'] = {'C_m': 300.0,               # Capacitance of the membrane (pF)
                                       'E_L': -62.0,               # Leak reversal potential aka resting potential (mV)
                                       'E_ex': 0.0,                # Excitatory reversal potential (mV)
                                       'E_in': -75.0,              # Inhibitory reversal potential (mV)
                                       'g_L': 300.0 / 20.0,        # Leak conductance (mV) - g_L = C_m / tau_E, where tau_I is membrane potential time constant
                                       'I_e': 0.0,                 # Constant external input current (pA)
                                       'V_reset': -60.0,           # Reset potential (for all neurons the same) (mV)
                                       'V_th': -52.0,              # Spike threshold (mV)
                                       't_ref': 1.0,               # Absolute refactory period (ms)
                                       'tau_syn_ex_decay': 6.0,    # Decay time of the excitatory synaptic difference of exp functions (ms)
                                       'tau_syn_ex_rise': 1.0,     # Rise time of the excitatory synaptic difference of exp functions (ms)
                                       'tau_syn_in_decay': 2.0,    # Decay time of the inhibitory synaptic difference of exp functions (ms)
                                       'tau_syn_in_rise': 0.5,     # Rise time of the inhibitory synaptic difference of exp functions (ms)
                                       'V_m': -60.0                # Membrane potential (mV)
                                       }
assert param_readout['interneuron_params']['g_L'] == (param_readout['interneuron_params']['C_m'] / 20.0)

# ========================================== poisson generator rates ==========================================

# Parameters of poisson generator to supervisor neurons
param_readout['exh_rate_sx'] = 10000.0                          # Training rate of external excitatory input to supervisor neurons (spikes/s)
param_readout['baseline_rate_sx'] = 1000.0                      # Basline rate of external inhibitory input to supervisor neurons (spikes/s)

# Parameters of poisson generator to interneurons
param_readout['exh_rate_hx'] = 1000.0                           # Rate of external excitatory input to interneurons (spikes/s)

# Parameters of poisson generator to read-out neurons to select a certain sequence while replay
param_readout['inh_rate_rx'] = 1000.0

# ========================================== connection and synapse dictionaries ==========================================

# Parameters of RS synapses (S neuron to E neurons)
param_readout['syn_dict_rs'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                'weight': 200.0,                            # Synaptic weight (pF)
                                'delay': RESOLUTION                         # Synaptic delay (ms)
                                }
param_readout['conn_dict_rs'] = {'rule': 'one_to_one'}                      # Connection rule

# Parameters of RH synapses (H neuron to R neurons)
param_readout['syn_dict_rh'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                'weight': -200.0,                           # Synaptic weight (pF)
                                'delay': RESOLUTION                         # Synaptic delay (ms)
                                }
param_readout['conn_dict_rh'] = {'rule': 'one_to_one'}                      # Connection rule

# Parameters of HR synapses (R neuron to H neurons)
param_readout['syn_dict_hr'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                'weight': 200.0,                            # Synaptic weight (pF)
                                'delay': RESOLUTION                         # Synaptic delay (ms)
                                }
param_readout['conn_dict_hr'] = {'rule': 'one_to_one'}                      # Connection rule

# Parameters of SX synapses (external to S neurons)
param_readout['syn_dict_sx'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                'weight': 1.78,                             # Synaptic weight (pF)
                                'delay': RESOLUTION                         # Synaptic delay (ms)
                                }
param_readout['conn_dict_sx'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of HX synapses (external to I neurons)
param_readout['syn_dict_hx'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                'weight': 1.78,                             # Synaptic weight (pF)
                                'delay': RESOLUTION                         # Synaptic delay (ms)
                                }
param_readout['conn_dict_hx'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of RX synapses (external to R neurons) (only relevant when replaying multiple sequences)
param_readout['syn_dict_rx'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                                'weight': -200.0,                             # Synaptic weight (pF)
                                'delay': RESOLUTION                         # Synaptic delay (ms)
                                }
param_readout['conn_dict_rx'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of RE synapses (voltage-based STDP for read-out synapses)
param_readout['syn_dict_re'] = {'synapse_model': 'clopath_synapse',         # Name of synapse model
                                'weight': 0.0,                              # Initial synaptic weight (pF)
                                'Wmax': 25.0,                               # Maximum allowed weight (pF)
                                'Wmin': 0.0,                                # Minimum allowed weight (pF)
                                'tau_x': 5.0,                               # Time constant of low pass filtered presynaptic spike train for read-out synapses (ms)
                                'delay': RESOLUTION                         # Synaptic delay (ms)
                                }
param_readout['conn_dict_re'] = {'rule': 'all_to_all'}                      # Connection rule

# ========================================== data path dict ==========================================
# TODO
# Simulation results such as spike times and connection weights are stored in clock_net/data/sequence_learning_performance/sequence_learning_and_prediction
param_readout['data_path'] = {}
param_readout['data_path']['data_root_path'] = 'data'
param_readout['data_path']['project_name'] = 'sequence_learning_performance'
param_readout['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction'

# ========================================== task parameters ==========================================
# TODO
param_readout['sequences'] = ['ABCBA', 'DEDED']            # Hard coded sequences to be learned
param_readout['task'] = {}
param_readout['task']['task_name'] = 'hard_coded'          # Name of the task ['high_order', 'random', 'structure', 'hard_coded']
param_readout['task']['task_type'] = 1                     # This chooses between three hard coded sequence sets (see ./utils.py)
param_readout['task']['vocab_size'] = 6                    # Vocabulary size
param_readout['task']['seed'] = 111                        # Seed number
param_readout['task']['store_training_data'] = True        # If turned on, the sequence set is stored in directory defined in dict data_path
if param_readout['task']['task_name'] != 'hard_coded':
    param_readout['task']['num_sequences'] = 2             # Number of sequences per sequence set
    param_readout['task']['num_sub_seq'] = 2               # If task_name == 'high_order', It sets the number of sequences with same shared subsequence
    param_readout['task']['length_sequence'] = 6           # Number of elements per sequence
    param_readout['task']['replace'] = False               # Random choice of characters with replacement

# ========================================== simulation parameters ==========================================
param_readout['sim_time'] = 12000 * len(param_readout['sequences'])                     # Trainings time, indicates how long the learning of a sequence takes in total (ms)
param_readout['switching_time'] = 2000                                                  # After this time there is a switch in sequence presentation (ms)
param_readout['stimulation_time'] = 75                                                  # Stimulation time for the presentation of one element, determines how long the supervisor generator sends signals to the corresponding S neuron (ms)
param_readout['lead_time'] = 50                                                         # Lead time to allow the recurrent network to exhibit sequential dynamics (ms)
param_readout['recording_setup'] = 'disconnect_readout_generators'                      # Determines which nodes influence the dynamics of the network during the replay phase ('disconnect_readout_generators', 'disconnect_readout_population', 'all_nodes')
param_readout['replay_tuples'] = [('ABCBA', 1000), ('DEDED', 1000)]                        # List of tuples where the tuple determines what sequence should be replayed for how long ((str, ms))
param_readout['recording_time'] = sum([i for _, i in param_readout['replay_tuples']])   # Duration of the recording of the spike behavior (ms)
