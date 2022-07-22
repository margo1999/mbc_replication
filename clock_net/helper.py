"""
Helper functions of the Clock Network model

Authors
~~~~~~~
Jette Oberlaender, Younes Bouhadjar
"""

import copy
import hashlib
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from parameters import ParameterSpace, ParameterSet

import numpy as np


##########################################
def generate_sequences(params, data_path, fname):
    """Generate sequence of elements using three methods:
    1. Higher order sequences: sequences with shared subsequences 'high_oder'
    2. Randomly drawn elements from a vocabulary 'random'
    3. Sequences with transition matrix 'structure'
    4. Hard coded sequences 'hard_coded'

    Parameters
    ----------
    params : dict
        dictionary contains task parameters
    data_path   : dict
    fname       : str

    Returns
    -------
    sequences: list
    test_sequences: list
    vocabulary: list
    """

    task_name = params['task_name']
    task_type = params['task_type']
    # length_seq = params['length_sequence']

    # set of characters used to build the sequences
    vocabulary = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z'][:params['vocab_size']]
    sequences = []

    # create high order sequences, characters are drawn without replacement
    if task_name == "high_order":

        if params['num_sequences'] % params['num_sub_seq'] != 0:
            raise ZeroDivisionError(
                'for high order sequences number of sequences needs ("num_sequences") to be divisible by num_sub_seq')

        num_sequences_high_order = int(params['num_sequences'] / params['num_sub_seq'])
        for _ in range(num_sequences_high_order):
            characters_sub_seq = copy.copy(vocabulary)
            sub_seq = random.sample(characters_sub_seq, params["length_sequence"] - 2)
            for char in sub_seq:
                characters_sub_seq.remove(char)

            for _ in range(params['num_sub_seq']):
                # remove the characters that were chosen for the end and the start of the sequence
                # this is to avoid sequences with adjacent letters of the same kind
                # we will add this feature to the code asap
                end_char = random.sample(characters_sub_seq, 1)
                characters_sub_seq.remove(end_char[0])

                start_char = random.sample(characters_sub_seq, 1)
                characters_sub_seq.remove(start_char[0])

                sequence = start_char + sub_seq + end_char
                sequences.append(sequence)

    elif task_name == "random":
        sequences = [random.sample(vocabulary, params['length_sequence']) for _ in range(params['num_sequences'])]

    elif task_name == "structure":
        matrix_transition = defaultdict(list)
        for char in vocabulary:
            samples = np.random.choice(2, len(vocabulary), p=[0.2, 0.8])
            matrix_transition[char] = samples / sum(samples)

        for _ in range(params['num_sequences']):
            sequence = random.sample(vocabulary, 1)
            last_char = sequence[-1]
            for _ in range(params['length_sequence'] - 1):
                sequence += np.random.choice(vocabulary, 1, p=matrix_transition[last_char])[0]
                last_char = sequence[-1]

            sequences += [sequence]

    elif task_name == "hard_coded":

        # hard coded sequences
        if task_type == 1:
            sequences = [['A', 'D', 'B', 'E'], ['F', 'D', 'B', 'C']]
        elif task_type == 2:
            sequences = [['E', 'N', 'D', 'I', 'J'], ['L', 'N', 'D', 'I', 'K'], ['G', 'J', 'M', 'C', 'N'],
                         ['F', 'J', 'M', 'C', 'I'], ['B', 'C', 'K', 'H', 'I'], ['A', 'C', 'K', 'H', 'F']]
        elif task_type == 3:
            sequences = [['E', 'N', 'D', 'I', 'J'], ['L', 'N', 'D', 'I', 'K'], ['G', 'J', 'M', 'E', 'N'],
                         ['F', 'J', 'M', 'E', 'I'], ['B', 'C', 'K', 'B', 'I'], ['A', 'C', 'K', 'B', 'F']]
        else:
            sequences = [['A', 'D', 'B', 'G', 'H', 'E'], ['F', 'D', 'B', 'G', 'H', 'C']]
    else:
        raise Exception(f"{task_name=} does not exist! Choose between: 'high_oder', 'random', 'structure', 'hard_coded'")

    # Test sequences used to measure the accuracy TODO: What is this used for?
    test_sequences = sequences

    if params['store_training_data']:
        fname = 'training_data'
        fname_voc = 'vocabulary'
        data_path = get_data_path(data_path)
        print(f"\nSave training data to {data_path}/{fname}")
        np.save(os.path.join(data_path, fname), sequences)
        np.save(os.path.join(data_path, fname_voc), vocabulary)

    return sequences, test_sequences, vocabulary


###############################################################################
def derived_parameters(params):
    """Set additional parameters derived from base parameters.

    A dictionary containing all (base and derived) parameters is stored as model attribute params

    Parameters
    ----------
    params:    dict
               Parameter dictionary
    """

    params = copy.deepcopy(params)

    # connection rules for EE connections
    # params['conn_dict_ee'] = {}
    # params['conn_dict_ee']['rule'] = params['rule']
    # params['conn_dict_ee']['indegree'] = int(params['connection_prob'] *
    #                                               params['M'] *
    #                                               params['n_E'])
    # params['conn_dict_ee']['allow_autapses'] = False
    # params['conn_dict_ee']['allow_multapses'] = False

    # compute neuron's membrane resistance
    # params['R_m_soma'] = params['exhibit_params']['tau_m'] / params['exhibit_params']['C_m']
    # params['R_m_inhibit'] = params['inhibit_params']['tau_m'] / params['inhibit_params']['C_m']

    # compute psc max from the psp max
    # params['J_IE_psp'] = 1.2 * params['inhibit_params']['V_th']         # inhibitory PSP as a response to an input from E neuron

    # if params['evaluate_replay']:
    #     params['J_IE_psp'] /= params['n_E']
    # else:
    #     params['J_IE_psp'] /= params['pattern_size']

    # params['syn_dict_ex']['weight'] = psp_max_2_psc_max(params['J_EX_psp'], params['exhibit_params']['tau_m'],
    #                                                params['exhibit_params']['tau_syn1'], params['R_m_soma'])
    # params['syn_dict_ie']['weight'] = psp_max_2_psc_max(params['J_IE_psp'], params['inhibit_params']['tau_m'],
    #                                                params['inhibit_params']['tau_syn_ex'],
    #                                                params['R_m_inhibit'])
    # params['syn_dict_ei']['weight'] = psp_max_2_psc_max(params['J_EI_psp'], params['exhibit_params']['tau_m'],
    #                                                params['exhibit_params']['tau_syn3'], params['R_m_soma'])

    # set initial weights (or permanences in the case of the structural synapse)
    # import nest
    # if params['syn_dict_ee']['synapse_model'] == 'stdsp_synapse':
    #     params['syn_dict_ee']['permanence'] = nest.random.uniform(min=params['p_min'], max=params['p_max'])
    # else:
    #     params['syn_dict_ee']['weight'] = nest.random.uniform(min=params['w_min'], max=params['w_max'])

    # params['syn_dict_ee']['dt_max'] = -2.*params['DeltaT']              # maximum time lag for the STDP window
    # params['DeltaT_seq'] = 2.5*params['DeltaT']                         # inter-sequence interval

    # clamp DeltaT_seq if it exceeds the duration of the dAP
    # if params['DeltaT_seq'] < params['exhibit_params']['tau_dAP']:
    #     params['DeltaT_seq'] = params['exhibit_params']['tau_dAP']

    # print('\n#### postsynaptic potential ####')
    # print('PSP maximum J_EX psp:  %f mV' % params['J_EX_psp'])
    # print('PSP maximum J_IE psp:  %f mV' % params['J_IE_psp'])
    # print('PSP maximum J_EI psp:  %f mV' % params['J_EI_psp'])

    # print('\n#### postsynaptic current ####')
    # print('PSC maximum J_EX:  %f pA' % params['syn_dict_ex']['weight'])
    # print('PSC maximum J_IE:  %f pA' % params['syn_dict_ie']['weight'])
    # print('PSC maximum J_EI:  %f pA' % params['syn_dict_ei']['weight'])

    return params


# ###############################################################################
# def get_parameter_set(analysis_pars):
#     """ Get parameter set from data directory at location specified in analysis_pars.

#     Parameters
#     ----------
#     analysis_pars: dict

#     Returns
#     -------
#     P: dict
#        ParameterSpace
#     """

#     params_path = get_data_path(analysis_pars)
#     sys.path.insert(0, str(params_path))

#     import parameters_space as data_pars

#     P = data_pars.p

#     return P, params_path


###############################################################################
def parameter_set_list(parameterspace: ParameterSpace):
    """ Generate list of parameters sets

    Parameters
    ----------
    parameterspace : dict
        parameter space

    Returns
    -------
    param_set_list : list
        list of parameter sets
    """

    param_set_list = []
    for param_set in parameterspace.iter_inner():
        param_set_copy = copy.deepcopy((param_set))  # copy.deepcopy(dict(param_set))
        param_set_copy['label'] = compute_parameter_set_hash(param_set_copy)  # add md5 checksum as label of parameter set (used e.g. for data file names)
        param_set_list.append(param_set_copy)

    return param_set_list


###############################################################################
def compute_parameter_set_hash(parameterset: ParameterSet) -> str:
    """_summary_ TODO documentation

    Args:
        parameterset (ParameterSet): _description_

    Returns:
        str: _description_
    """
    return hashlib.md5(pformat(parameterset).encode('utf-8')).hexdigest()


###############################################################################
def get_data_path(pars, ps_label='', add_to_path=''):
    """ Construct the path to the data directory

    Parameters
    ----------
    pars : dict
           path parameters
    ps_label : string
    add_to_path : string

    Returns
    -------
    data_path : Pathlib instantiation
    """

    if "home" in pars:
        home = pars['home']
    else:
        home = '../..'

    data_path = Path(home, pars['data_root_path'],
                     pars['project_name'],
                     pars['parameterspace_label'],
                     ps_label, add_to_path)

    return data_path


###############################################################################
def copy_scripts(pars, fname):
    """ Copying Python scripts to data folder

    Parameters
    ----------
    pars  : dict
    fname : string

    """

    print("\tCopying Python scripts to data folder ...")
    data_path = get_data_path(pars)
    os.system(f'mkdir -p {data_path:s}')
    # os.system('cp -r --backup=t  %s %s/%s' % (fname, data_path, 'parameters_space.py'))
    os.system(f'cp -r {fname} {data_path}/parameters_space.py')
    # os.system('mv %s/%s %s/%s' % (dat_path,fname,dat_path,'sim_'+fname))


##############################################
def load_spike_data(path, label, skip_rows=3):
    """Load spike data from files.

    Parameters
    ---------
    path:           str
                    Path containing spike files.

    label:          str
                    Spike file label (file name root).

    skip_rows:      int, optional
                    Number of rows to be skipped while reading spike files (to remove file headers). The default is 3.

    Returns
    -------
    spikes:   numpy.ndarray
              Lx2 array of spike senders spikes[:,0] and spike times spikes[:,1] (L = number of spikes).
    """

    # get list of files names
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith('.dat') and file_name.startswith(label):
            files += [file_name]
    files.sort()

    assert len(files) > 0, f"No files of type '{label:s}*.dat' found in path '{path:s}'."

    # open spike files and read data
    spikes = []
    for file_name in files:
        try:
            spikes += [np.loadtxt(os.path.join(path, file_name), skiprows=skip_rows)]  # load spike file while skipping the header
        except Exception:
            print(f'Error: {sys.exc_info()[1]}')
            print(f'Remove non-numeric entries from file {file_name} (e.g. in file header) by specifying (optional) parameter "skip_rows".\n')

    try:
        spikes = np.concatenate([spike for spike in spikes if spike.size > 0])
    except ValueError:
        print("All files are empty")

#    # open spike files and read data
#    spikes = []
#    for file_name in files:
#        try:
#            spikes += [np.loadtxt('%s/%s' % (path, file_name),
#                                  skiprows=skip_rows)]  ## load spike file while skipping the header
#            print(spikes)
#        except:
#            print("Error: %s" % sys.exc_info()[1])
#            print(
#                "Remove non-numeric entries from file %s (e.g. in file header) by specifying (optional) parameter 'skip_rows'.\n" % (
#                    file_name))
#
#    spikes = np.concatenate(spikes)
#
    return spikes


###############################################################################
def load_data(path, fname):
    """Load data

    Parameters
    ----------
    path: str
    fname: str

    Returns
    -------
    data: ndarray
    """

    # TODO: this is temporary hack!
    try:
        data = np.load(f'{path}/{fname}.npy', allow_pickle=True).item()
    except ValueError:
        data = np.load(f'{path}/{fname}.npy', allow_pickle=True)

    return data
