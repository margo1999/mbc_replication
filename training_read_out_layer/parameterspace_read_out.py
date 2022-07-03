import numpy as np
import parameters as para

RESOLUTION = 0.1
SEED = 1

np.random.seed(SEED)

p = para.ParameterSpace({})

# ========================================== all node models ==========================================

# Parameters of read-out neurons
V_m_exc = np.random.uniform(low=-60.0, high=-52, size=240)  # TODO make this generalized and add to dictionary
p['read_out_model'] = 'aeif_cond_diff_exp_clopath'
p['read_out_params'] = {'A_LTD': 0.00014,           # LTD amplitude (pa/mV)
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
assert p['read_out_params']['V_th_max'] == (p['read_out_params']['V_th_rest'] + 10.0)
assert p['read_out_params']['g_L'] == (p['read_out_params']['C_m'] / 20.0)

# Parameters of supervisor neurons
V_m_exc = np.random.uniform(low=-60.0, high=-52, size=240)  # TODO make this generalized and add to dictionary
p['supervisor_model'] = 'aeif_cond_diff_exp_clopath'
p['supervisor_params'] = {'A_LTD': 0.00014,           # LTD amplitude (pa/mV)
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
assert p['supervisor_params']['V_th_max'] == (p['supervisor_params']['V_th_rest'] + 10.0)
assert p['supervisor_params']['g_L'] == (p['supervisor_params']['C_m'] / 20.0)

# Parameters of interneurons
V_m_inh = np.random.uniform(low=-60.0, high=-52, size=60)  # TODO make this generalized and add to dictionary
p['interneuron_model'] = 'iaf_cond_diff_exp'
p['interneuron_params'] = {'C_m': 300.0,               # Capacitance of the membrane (pF)
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
assert p['interneuron_params']['g_L'] == (p['interneuron_params']['C_m'] / 20.0)

# ========================================== poisson generator rates ==========================================

# Parameters of poisson generator to supervisor neurons
p['exh_rate_sx'] = 10000.0                          # Training rate of external excitatory input to supervisor neurons (spikes/s)
p['baseline_rate_sx'] = 1000.0                      # Basline rate of external inhibitory input to supervisor neurons (spikes/s)

# Parameters of poisson generator to interneurons
p['exh_rate_hx'] = 1000.0                           # Rate of external excitatory input to interneurons (spikes/s)

# ========================================== connection and synapse dictionaries ==========================================

# Parameters of RS synapses (S neuron to E neurons)
p['syn_dict_rs'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                    'weight': 200.0,                            # Synaptic weight (pF)
                    'delay': RESOLUTION                         # Synaptic delay (ms)
                    }
p['conn_dict_rs'] = {'rule': 'one_to_one'}                      # Connection rule

# Parameters of RH synapses (H neuron to R neurons)
p['syn_dict_rh'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                    'weight': -200.0,                           # Synaptic weight (pF)
                    'delay': RESOLUTION                         # Synaptic delay (ms)
                    }
p['conn_dict_rh'] = {'rule': 'one_to_one'}                      # Connection rule

# Parameters of HR synapses (R neuron to H neurons)
p['syn_dict_hr'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                    'weight': 200.0,                            # Synaptic weight (pF)
                    'delay': RESOLUTION                         # Synaptic delay (ms)
                    }
p['conn_dict_hr'] = {'rule': 'one_to_one'}                      # Connection rule

# Parameters of SX synapses (external to S neurons)
p['syn_dict_sx'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                    'weight': 1.78,                             # Synaptic weight (pF)
                    'delay': RESOLUTION                         # Synaptic delay (ms)
                    }
p['conn_dict_sx'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of HX synapses (external to I neurons)
p['syn_dict_hx'] = {'synapse_model': 'static_synapse',          # Name of synapse model
                    'weight': 1.78,                             # Synaptic weight (pF)
                    'delay': RESOLUTION                         # Synaptic delay (ms)
                    }
p['conn_dict_hx'] = {'rule': 'all_to_all'}                      # Connection rule

# Parameters of RE synapses (voltage-based STDP for read-out synapses)
p['syn_dict_re'] = {'synapse_model': 'clopath_synapse',         # Name of synapse model
                    'weight': 0.0,                              # Initial synaptic weight (pF)
                    'Wmax': 25.0,                               # Maximum allowed weight (pF)
                    'Wmin': 0.0,                                # Minimum allowed weight (pF)
                    'tau_x': 5.0,                               # Time constant of low pass filtered presynaptic spike train for read-out synapses (ms)
                    'delay': RESOLUTION                         # Synaptic delay (ms)
                    }
p['conn_dict_re'] = {'rule': 'all_to_all'}                      # Connection rule

# ========================================== simulation parameters ==========================================
# TODO


# ========================================== data path dict ==========================================
# TODO
# Simulation results such as spike times and connection weights are stored in clock_net/data/sequence_learning_performance/sequence_learning_and_prediction
p['data_path'] = {}
p['data_path']['data_root_path'] = 'data'
p['data_path']['project_name'] = 'sequence_learning_performance'
p['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction'

# ========================================== task parameters ==========================================
# TODO
p['task'] = {}
p['task']['task_name'] = 'hard_coded'          # Name of the task ['high_order', 'random', 'structure', 'hard_coded']
p['task']['task_type'] = 1                     # This chooses between three hard coded sequence sets (see ./utils.py)
p['task']['vocab_size'] = 6                    # Vocabulary size
p['task']['seed'] = 111                        # Seed number
p['task']['store_training_data'] = True        # If turned on, the sequence set is stored in directory defined in dict data_path
if p['task']['task_name'] != 'hard_coded':
    p['task']['num_sequences'] = 2             # Number of sequences per sequence set
    p['task']['num_sub_seq'] = 2               # If task_name == 'high_order', It sets the number of sequences with same shared subsequence
    p['task']['length_sequence'] = 6           # Number of elements per sequence
    p['task']['replace'] = False               # Random choice of characters with replacement
