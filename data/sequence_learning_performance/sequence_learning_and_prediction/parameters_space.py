import parameters as para
import numpy as np

DELAY = 0.1

p = para.ParameterSpace({})

###################################################### general parameters for RNN ######################################################
p['num_exc_neurons'] = 240                                             # Number of recurrent E neurons
p['num_inh_neurons'] = 60                                              # Number of recurrent I neurons
p['num_exc_clusters'] = 8                                              # Number of exciatatory clusters
p['exc_cluster_size'] = p['num_exc_neurons'] // p['num_exc_clusters']   # Number of excitatory neurons in one cluster
p['connection_p'] = 0.2                                                 # Recurrent network connection probability

###################################################### all node models ######################################################

#TODO: replace model by actual model
# parameters of excitatory neurons
p['exhibit_model'] = 'aeif_psc_delta_clopath'
p['exhibit_params'] = {'A_LTD': 0.0014,        #! check if this is equal to 0.0014 pA mV^-2 LTD amplitude for voltage-based STDP  !not the same
                    'A_LTP': 0.0008,        #! check if this is equal to 0.0008 pA mV^-1 LTP amplitude  !not the same
                    'C_m': 300.0,           # Capacitance of the membrane 
                    'V_reset': -60.0,       # Reset potential (for all neurons the same) default value: -60
                   #'V_th': -52.0,          # Adaptive spike initiation threshold
                   'V_peak': 1000.0,
                   'V_th_max': 1000.0,         
                   'V_th_rest': 1000.0,         
                    't_ref': 5.0,            # absolute refactory period
                    'V_m': -60.0
                    }
print(p['exhibit_params'])

#TODO: replace model by actual model
# parameters of inhibitory neurons
p['inhibit_model'] = 'iaf_psc_exp'
p['inhibit_params'] = {}
p['inhibit_params']['C_m'] = 300.0          # membrane capacitance (pF)
p['inhibit_params']['E_L'] = 0.0            # resting membrane potential (mV)
p['inhibit_params']['I_e'] = 0.0            # external DC currents (pA)
p['inhibit_params']['V_reset'] = -60.0        # reset potential (mV)
p['inhibit_params']['V_m'] = p['inhibit_params']['V_reset']            # initial potential (mV)
p['inhibit_params']['V_th'] = 1000.0          # spike threshold (mV)
p['inhibit_params']['t_ref'] = 5.0          # refractory period
p['inhibit_params']['tau_m'] = 20.0          # membrane time constant (ms)
p['inhibit_params']['tau_syn_ex'] = 0.5     # synaptic time constant of an excitatory input (ms) 
p['inhibit_params']['tau_syn_in'] = 1.65    # synaptic time constant of an inhibitory input (ms)

# TODO: external poisson generator for excitatory neurons
p['exh_rate_ex'] = 0.0 #18000.0
p['inh_rate_ex'] = 0.0 #4500.0

# TODO: external poisson generator for inhibitory neurons
p['exh_rate_ix'] = 0.0 #2250.0

###################################################### connection and synapse dictionaries ######################################################

#general connection dictionary
general_RNN_conn_dict = {'rule': 'pairwise_bernoulli',              # connection rule
                        'p': p['connection_p'],
                        'allow_autapses': False,
                        'allow_multapses': False
                        }

# parameters of excitatory EX synapses (external to E neurons)
p['syn_dict_ex_exc'] = {'synapse_model': 'static_synapse',          # synapse model
                        'weight': 1.6,                              # synaptic weight
                        }
p['conn_dict_ex_exc'] = {'rule': 'all_to_all'}                      # connection rule

# parameters of inhibitory EX synapses (external to E neurons)
p['syn_dict_ex_inh'] = {'synapse_model': 'static_synapse',          # synapse model
                        'weight': -2.4                              # synaptic weight
                        }
p['conn_dict_ex_inh'] = {'rule': 'all_to_all'}                      #connection rule

# parameters of IX synapses (external to I neurons)
p['syn_dict_ix'] = {'synapse_model': 'static_synapse',              # synapse model
                        'weight': 1.52                              # synaptic weight
                        }
p['conn_dict_ix'] = {'rule': 'all_to_all'}                          #connection rule

# parameters of EE synapses (voltage-based STDP)
p['syn_dict_ee'] = {'weight': 2.83,                                 # synaptic weight
                    'synapse_model': 'clopath_synapse',             # synapse model
                    'Wmax': 32.68,                                  # maximum E to E weight
                    'Wmin': 1.45,                                   # minimum E to E weight
                    'delay': 1000
                    }
p['conn_dict_ee'] = general_RNN_conn_dict

# parameters for II synapses
p['syn_dict_ii'] = {'synapse_model': 'static_synapse',              # synapse model
                    'weight': 20.91                                 # synaptic weight
                    }
p['conn_dict_ii'] = general_RNN_conn_dict

# parameters for IE synapses 
p['syn_dict_ie'] = {'synapse_model': 'static_synapse',              # synapse model
                    'weight': 1.96                                  # synpatic weight
                   }
p['conn_dict_ie'] = general_RNN_conn_dict

#TODO: check if synapse model is correct + need to add Wmin but for synpase implementation there is no Wmin key in dict
# parameters for EI synapses 
p['syn_dict_ei'] = {'synapse_model': 'vogels_sprekeler_synapse',    # synapse model
                    'weight': 62.87,                                # initial synpatic weight
                    'Wmax': 243.0,
#                    'Wmin': 48.7
                   }
p['conn_dict_ei'] = general_RNN_conn_dict

###################################################### simulation parameters ######################################################

# simulation parameters 
p['dt'] = 0.1                                                           # simulation time resolution (ms)
p['overwrite_files'] = True                                             # True: data will be overwritten; False: a NESTError is raised if the files already exist
p['seed'] = para.ParameterRange([1])                                    # seed for NEST
p['print_simulation_progress'] = False                                  # print the time progress.
p['n_threads'] = 2                                                      # number of threads per MPI process 
p['pad_time'] = 5.                                                      # TODO: What is this?
p['idend_recording_interval'] = 10 * p['dt']                            # dendritic current recording resolution TODO: Do we need this?
p['idend_record_time'] = 8.                                             # time interval after the external stimulation at which the dendritic current is recorded TODO: Do we need this?
p['evaluate_performance'] = True                                        # if True, we monitor the dendritic current at a certain time steps during the simulation. This then is used for the prediction performance assessment
p['evaluate_replay'] = False                                            # TODO: What is this?  
p['record_idend_last_episode'] = True                                   # used for debugging, if turned on we record the dendritic current of all neurons this can consume too much memory TODO: Does this take too much time?
p['store_connections'] = True                                           # stores connection in a seperate file
p['load_connections'] = False                                           # loads connection from existing file
#p['sparse_first_char'] = False                                         # if turned on, the dAP of a subset of neurons in the subpopulation representing 
                                                                        # first sequence elements is activated externally 
p['active_weight_recorder'] = False                                     # if True, the weights are recorded every presynaptic spike
p['cluster_stimulation_time'] = 10.0                            # stimulation time from external input to excitatory cluster
p['stimulation_gap'] = 5.0                                              # gap between to stimulations of excitatory clusters
#p['sim_time'] = p['num_exc_clusters'] * p['cluster_stimulation_time'] + (p['num_exc_clusters'] - 1) * p['stimulation_gap']    #simulation time for one round TODO: better formatation
p['sim_time'] = p['num_exc_clusters'] * (p['cluster_stimulation_time'] + p['stimulation_gap'])
###################################################### data path dict ######################################################
p['data_path'] = {}
p['data_path']['data_root_path'] = 'data'
p['data_path']['project_name'] = 'sequence_learning_performance'
p['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction'

###################################################### task parameters ######################################################

# task parameters
p['task'] = {}
p['task']['task_name'] = 'hard_coded'          # name of the task
p['task']['task_type'] = 1                     # this chooses between three hard coded sequence sets (see ./utils.py)
p['task']['vocab_size'] = 6                   # vocabulary size
p['task']['seed'] = 111                        # seed number
p['task']['store_training_data'] = True        # if turned on, the sequence set is stored in directory defined in dict data_path
if p['task']['task_name'] != 'hard_coded':
    p['task']['num_sequences'] = 2             # number of sequences per sequence set
    p['task']['num_sub_seq'] = 2               # if task_name == 'high_order', 
                                               # it sets the number of sequences with same shared subsequence
    p['task']['length_sequence'] = 6           # number of elements per sequence
    p['task']['replace'] = False               # random choice of characters with replacement

###################################################### REST ######################################################

# # setup the training loop  
# p['learning_episodes'] = 85                     # total number of training episodes ('repetitions of the sequence sets')
# p['episodes_to_testing'] = 1                   # number of episodes after which we measure the prediction perfomance

#TODO: Do I need this?
# stimulus parameters
p['DeltaT'] = 40.                     # inter-stimulus interval
p['excitation_start'] = 30.           # time at which the external stimulation begins
p['time_dend_to_somatic'] = 20.       # time between the dAP activation and the somatic activation (only used if sparse_first_char is True)   
p['DeltaT_cue'] = 80.                 # inter-cue interval during replay
