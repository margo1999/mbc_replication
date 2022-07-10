"""Learns and replays sequences based on the Maes network (2021). The network is created through the Model class, whereby connections (EE, EI, IE, II) in the RNN
    are loaded and kept fixed. Functions to create neurons and connection for the read-out layer.
    TODO documentation / add all functionalities after implemented
"""
import os
import sys
from pickle import dump
from typing import Tuple

import matplotlib.pyplot as plt
import nest
import numpy as np
from parameters import ParameterSet
from tqdm import tqdm
from clock_net import helper, plot_helper
from clock_net.model import Model
from experiments.sequential_dynamics.parameters_space import param_readout as paramspace_readout


OFFSET = 30                         # Time interval between the activation of the first cluster and the presentation of the first element of the sequence (ms)
PROBE_INTERVAL_LAST = 5             # Probe interval in which the activity of the last cluster is measured in order to detect the start of a sequential run in the RNN (ms)
PROBE_INTERVAL_FIRST = 2            # Probe interval in which the activity of the first cluster is measured in order to detect the start of a sequential run in the RNN (ms)
CLUSTER_ACTIVITY_THRESHOLD = 5      # required amount of spikes within a cluster so that its activity can be inferred (ms)
ASSERTION_TIME_THRESHOLD = 700      # Limited time interval to prevent infinite loops if the parameters for detecting the start of a sequential run are not set appropriately (ms)


class SequenceModel:
    """ TODO documentation
    """
    rnn_model = None
    sr_first = None
    sr_last = None
    gen_h = None
    gen_s_baseline = None
    gen_s = None
    sequence_selector = None
    paramset_readout = {}
    created_global_members = False

    @classmethod
    def setup(cls, paramset_readout: ParameterSet):
        """ TODO documentation
        """
        cls.paramset_readout = paramset_readout
        cls.create_rnn_model(paramset_recurrent=paramset_readout['param_recurrent'])
        cls.create_generators()
        cls.created_global_members = True

    @classmethod
    def create_rnn_model(cls, paramset_recurrent: ParameterSet):
        """ TODO documentation
        """
        sequences, _, vocabulary = helper.generate_sequences(paramset_recurrent['task'], paramset_recurrent['data_path'], paramset_recurrent['label'])  # TODO check if this works

        cls.rnn_model = Model(paramset_recurrent, sequences, vocabulary)
        cls.rnn_model.create_rnn_populations()
        cls.rnn_model.load_connections(label='all_connections')

        # # Plot loaded weights
        # conns = nest.GetConnections()
        # conns = nest.GetStatus(conns, ['target', 'source', 'weight'])
        # plot_helper.plot_weight_matrix(ax=None, connections=conns, title='initial trained weight matrix')
        # plt.show()

        # Create Poisson Generators and connect them to achive spontaneous dynamics in RNN
        cls.rnn_model.set_up_spontaneous_dynamics(sim_time=sys.maxsize)  # RNN receives spontaneous input for an unlimited amount of time

        # Create and connect spike recorders to identify beginning of sequential dynamics
        cls.sr_first, cls.sr_last = create_spike_recorders_for_detection()
        connect_spike_recorders_for_detection(model_instance=cls.rnn_model, sr_first=cls.sr_first, sr_last=cls.sr_last)

        print(f"{nest.network_size=} after setting up RNN + random input generators + spike recorders for detection")
        assert nest.network_size == (cls.rnn_model.num_exc_neurons + cls.rnn_model.num_inh_neurons + 2 + 2)

    @classmethod
    def create_generators(cls):
        """ TODO documentation
        """
        cls.gen_h, cls.gen_s_baseline, cls.gen_s = cls._create_read_out_layer_generators(params_ro=cls.paramset_readout)
        cls.sequence_selector = cls._create_sequence_selector(params_ro=cls.paramset_readout)

    def __init__(self, sequence: str):
        """ TODO documentation
        """
        if not self.created_global_members:
            raise Exception("Must call setup before creating Sequence objects!")

        self.sequence = sequence
        self.alphabet = sorted(set(self.sequence))

        # Create neurons and connections to get the final model architecture
        self._create_read_out_layer_population()
        self._connect_recurrent_to_readout()
        self._connect_readout_population()
        self._connect_readout_generators()
        self._connect_sequence_selectors()

        self.read_out_dict = dict(zip(self.alphabet, self.s_neurons))

    def _create_read_out_layer_population(self):
        """Creates all required neurons in the read-out component of the network to set up learning of
        a sequence. For each distinct element of the sequence a read-out neuron (R), a supervisor
        neuron (S) and an interneuron (H) are created. This function should only be called once per sequence.
        """
        params_ro = self.paramset_readout
        neuron_num = len(self.alphabet)
        self.r_neurons = nest.Create(params_ro['read_out_model'], neuron_num, params_ro['read_out_params'])
        self.s_neurons = nest.Create(params_ro['supervisor_model'], neuron_num, params_ro['supervisor_params'])
        self.h_neurons = nest.Create(params_ro['interneuron_model'], neuron_num, params_ro['interneuron_params'])

    @staticmethod
    def _create_read_out_layer_generators(*, params_ro: ParameterSet) -> Tuple[nest.NodeCollection, nest.NodeCollection, nest.NodeCollection]:
        """Creates three types of Poisson spike generators to enable learning of a sequence.
        One constantly active generator for all interneurons (H). One baseline and one supervisor
        generator for all supervisor neurons (S), whereby these alternate. The time of alternation
        is given by the sequence.

        Args:
            params_ro (ParameterSet): Dictionary of read-out parameters including parameters of generators

        Returns:
            Tuple[nest.NodeCollection, nest.NodeCollection, nest.NodeCollection]: Poisson generator for H neurons,
            basline Poisson generator and supervisor Poisson generator for S neurons
        """
        gen_h = nest.Create('poisson_generator', params=dict(rate=params_ro['exh_rate_hx']))
        gen_s_baseline = nest.Create('poisson_generator', params=dict(rate=params_ro['baseline_rate_sx']))
        gen_s = nest.Create('poisson_generator', params=dict(rate=params_ro['exh_rate_sx']))

        return gen_h, gen_s_baseline, gen_s

    @staticmethod
    def _create_sequence_selector(*, params_ro: ParameterSet):
        """_summary_ TODO documentation

        Args:
            params_ro (ParameterSet): _description_

        Returns:
            _type_: _description_
        """
        sequence_selector = nest.Create('poisson_generator', params=dict(origin=sys.maxsize, rate=params_ro['inh_rate_rx']))

        return sequence_selector

    def _connect_recurrent_to_readout(self):
        """Connects excitatory neurons (E) of the recurrent network (RNN) all-to-all to read-out neurons (R)
        in the read-out layer.
        """
        nest.Connect(self.rnn_model.exc_neurons, self.r_neurons, self.paramset_readout['conn_dict_re'], self.paramset_readout['syn_dict_re'])

    def _connect_readout_population(self):
        """Connects the neurons in the read-out layer accordingly. Read-out neurons (R) and Interneurons (H) are bidirectionally connected.
        Supervisor neurons (S) are connected to Read-out neurons (R).

        Args:
            params_ro (ParameterSet): Dictionary of read-out parameters including connection parameters in the read-out layer (synapse type, weight,...)
        """
        nest.Connect(self.h_neurons, self.r_neurons, self.paramset_readout['conn_dict_rh'], self.paramset_readout['syn_dict_rh'])
        nest.Connect(self.r_neurons, self.h_neurons, self.paramset_readout['conn_dict_hr'], self.paramset_readout['syn_dict_hr'])
        nest.Connect(self.s_neurons, self.r_neurons, self.paramset_readout['conn_dict_rs'], self.paramset_readout['syn_dict_rs'])

    def _connect_readout_generators(self):
        """Connects Poisson generators to supervisor (S) and interneurons (H) in the read-out layer to enable learning of a sequence. S neurons are connected to a
        baseline generator and a supervisor generator. The two generators can only be active alternately, therefore the synaptic weight from the supervisor
        generator to the S neuron is set to zero first (≈inactive). H neurons are only connected to one generator.

        Args:
            params_ro (ParameterSet): Dictionary of read-out parameters including connection parameters for generators (weight,...)
        """
        nest.Connect(self.gen_h, self.h_neurons, conn_spec=self.paramset_readout['conn_dict_hx'], syn_spec=self.paramset_readout['syn_dict_hx'])
        nest.Connect(self.gen_s_baseline, self.s_neurons, conn_spec=self.paramset_readout['conn_dict_sx'], syn_spec=self.paramset_readout['syn_dict_sx'])
        nest.Connect(self.gen_s, self.s_neurons, conn_spec=self.paramset_readout['conn_dict_sx'], syn_spec=self.paramset_readout['syn_dict_sx'])
        conn = nest.GetConnections(source=self.gen_s, target=self.s_neurons)
        conn.weight = 0.0

    def _connect_sequence_selectors(self):
        nest.Connect(self.sequence_selector, self.r_neurons, conn_spec=self.paramset_readout['conn_dict_rx'], syn_spec=self.paramset_readout['syn_dict_rx'])


def create_spike_recorders_for_detection() -> Tuple[nest.NodeCollection, nest.NodeCollection]:
    """Creates two spike recorders to identify the start time of a sequential run in the recurrent
       network (clock period). One spike recorder is needed to check for activity in
       the last cluster. If spikes are recorded, a second spike recorder will observe the activity in
       the first cluster to eventually determine the start time. The recorders for the first cluster is
       inactive when created and should be activated by setting the time as soon as the recorder for the
       last cluster detects sufficient spikes. All spike times and neuron ids are written to 'memory',
       data has to be reseted manually after successful detection.

    Returns:
        Tuple[nest.NodeCollection, nest.NodeCollection]: spike recorder used to detect if the first cluster is being active,
        spike recorder used to detect if the last cluster is being active
    """
    sr_first = nest.Create("spike_recorder", params={'record_to': 'memory', 'stop': 0.0})
    sr_last = nest.Create("spike_recorder", params={'record_to': 'memory'})

    return sr_first, sr_last


def connect_spike_recorders_for_detection(*, model_instance: Model, sr_first: nest.NodeCollection, sr_last: nest.NodeCollection):
    """Connects the spike recorder used for detection of the start time of a sequential run
       to the first and last cluster in the recurrent network. The recorders are connected to all neurons
       in the corresponding cluster.

    Args:
        model_instance (Model): _description_
        sr_first (nest.NodeCollection): spike recorder used to detect if the first cluster is being active
        sr_last (nest.NodeCollection): spike recorder used to detect if the last cluster is being active
    """
    num_exc_neurons = model_instance.num_exc_neurons
    num_exc_clusters = model_instance.num_exc_clusters
    cluster_size = num_exc_neurons // num_exc_clusters
    nest.Connect(model_instance.exc_neurons[:cluster_size], sr_first)
    nest.Connect(model_instance.exc_neurons[(num_exc_neurons - 30):num_exc_neurons], sr_last)


def create_spike_recorders() -> Tuple[nest.NodeCollection, nest.NodeCollection, nest.NodeCollection]:
    """Creates three spike recorders. One each for inhibitory (I) and excitatory (E) neurons in the recurrent
       neurons in the recurrent network, as well as one for the read-out neurons. The recorders are
       necessary for plotting the spike behavior. The recorders are inactive when created and must be activated later by setting the times.
       All spike times and neuron ids are written to 'memory' meaning that they have to be saved manually otherwise the data will be
       lost after simulation.

    Returns:
        Tuple[nest.NodeCollection, nest.NodeCollection, nest.NodeCollection]: recorder for E neuron, recorder for I neuron, recorder for R neuron
    """
    sr_e = nest.Create("spike_recorder", params={'record_to': 'memory', 'stop': 0.0})
    sr_i = nest.Create("spike_recorder", params={'record_to': 'memory', 'stop': 0.0})
    sr_r = nest.Create("spike_recorder", params={'record_to': 'memory', 'stop': 0.0})

    return sr_e, sr_i, sr_r


def connect_spike_recorders_to_rnn(*, model_instance: Model, sr_e: nest.NodeCollection, sr_i: nest.NodeCollection):
    """Connects the RNN spike recorders to their respective neuron type (inhibitory (I) and excitatory (E)) of the
       recurrent network. This is necessary for plotting the spike behavior.

    Args:
        model_instance (Model): _description_
        sr_e (nest.NodeCollection): spike recorder for excitatory neurons (E)
        sr_i (nest.NodeCollection): spike recorder for inhibitory neurons (I)
    """
    nest.Connect(model_instance.exc_neurons, sr_e)
    nest.Connect(model_instance.inh_neurons, sr_i)


def connect_spike_recorders_to_readout(*, model_instance: SequenceModel, sr_r: nest.NodeCollection):
    """Connects the read-out spike recorders to the read-out neurons of the sequence model.
       This is necessary for plotting the spike behavior.

    Args:
        model_instance (SequenceModel): _description_
        sr_r (nest.NodeCollection): spike recorder for read-out neurons (R)
    """
    nest.Connect(model_instance.r_neurons, sr_r)


def save_spikes_after_sim(*, sr_e: nest.NodeCollection, sr_i: nest.NodeCollection, sr_r: nest.NodeCollection, params_ro: ParameterSet):
    """Saves all spikes of inhibitory (I), excitatory (E) and read-out (R) neurons that occurred during the
       recording time in a pickle file. The data can be used to plot the spike behavior.

    Args:
        sr_e (nest.NodeCollection): spike recorder for excitatory neurons (E)
        sr_i (nest.NodeCollection): spike recorder for inhibitory neurons (I)
        sr_r (nest.NodeCollection): spike recorder for read-out neurons (R)
        params_ro (ParameterSet): Dictionary of read-out parameters including recording setup
    """
    sr_times_e = sr_e.events['times']
    sr_senders_e = sr_e.events['senders']

    sr_times_i = sr_i.events['times']
    sr_senders_i = sr_i.events['senders']

    sr_times_r = sr_r.events['times']
    sr_senders_r = sr_r.events['senders']

    spikes = dict(sr_times_e=sr_times_e, sr_senders_e=sr_senders_e, sr_times_i=sr_times_i, sr_senders_i=sr_senders_i, sr_times_r=sr_times_r, sr_senders_r=sr_senders_r)
    spikefilepath = os.path.join('/Users/Jette/Desktop/results/NEST/job_3822329/1787e7674087ddd1d5749039f947a2cd/', 'spikes_after_learning_' + params_ro['recording_setup'] + '.pickle')  # TODO
    dump(spikes, open(spikefilepath, "wb"))


def disconnect_readout_generators(*, gen_h: nest.NodeCollection, gen_s_baseline: nest.NodeCollection, gen_s: nest.NodeCollection):
    """_summary_

    Args:
        gen_h (nest.NodeCollection): _description_
        gen_s_baseline (nest.NodeCollection): _description_
        gen_s (nest.NodeCollection): _description_
        h_neurons (nest.NodeCollection): _description_
        s_neurons (nest.NodeCollection): _description_
    """
    sg_baseline_conns = nest.GetConnections(source=gen_s_baseline)
    sg_conns = nest.GetConnections(source=gen_s)
    hg_conns = nest.GetConnections(source=gen_h)
    sg_baseline_conns.weight = 0.0
    sg_conns.weight = 0.0
    hg_conns.weight = 0.0


def disconnect_readout_population(*, r_neurons: nest.NodeCollection, s_neurons: nest.NodeCollection, h_neurons: nest.NodeCollection):
    """_summary_

    Args:
        r_neurons (nest.NodeCollection): _description_
        s_neurons (nest.NodeCollection): _description_
        h_neurons (nest.NodeCollection): _description_
    """
    rs_conns = nest.GetConnections(source=s_neurons, target=r_neurons)
    rh_conns = nest.GetConnections(source=h_neurons, target=r_neurons)
    hr_conns = nest.GetConnections(source=r_neurons, target=h_neurons)
    rs_conns.weight = 0.0
    rh_conns.weight = 0.0
    hr_conns.weight = 0.0


def simulate_sequence_learning(*, params_ro: ParameterSet, sequence_model: SequenceModel):
    """_summary_ TODO documentation

    Args:
        sequence_model (SequenceModel): _description_
        params_ro (ParameterSet): Dictionary of read-out parameters
    """
    training_sequence = list(sequence_model.sequence)
    alphabet = sequence_model.alphabet
    assert len(sequence_model.r_neurons) == len(alphabet)
    assert len(sequence_model.s_neurons) == len(alphabet)
    assert len(sequence_model.h_neurons) == len(alphabet)

    stimulation_time = params_ro['stimulation_time']
    lead_time = params_ro['lead_time']
    if len(params_ro['sequences']) > 1:
        sim_time = params_ro['switching_time']
    else:
        sim_time = params_ro['sim_time']

    # Simulate
    sr_last = sequence_model.sr_last
    sr_first = sequence_model.sr_first
    cur_time = nest.biological_time
    print(f"At {cur_time=} learn {sequence_model.sequence=}")

    nest.Simulate(lead_time)  # needed to exhibit sequential dynamics in recurrent network

    while nest.biological_time - cur_time < sim_time:  # start of sequence learning
        first_cluster_active = False
        sr_last.stop = sys.maxsize

        while not first_cluster_active:
            start = nest.biological_time
            nest.Simulate(PROBE_INTERVAL_LAST)

            if sr_last.n_events > CLUSTER_ACTIVITY_THRESHOLD:
                sr_last.stop = nest.biological_time
                sr_first.stop = sys.maxsize

                while not first_cluster_active:
                    nest.Simulate(PROBE_INTERVAL_FIRST)

                    if sr_first.n_events > CLUSTER_ACTIVITY_THRESHOLD:
                        first_cluster_active = True
                    sr_first.n_events = 0
                    assert nest.biological_time <= start + ASSERTION_TIME_THRESHOLD
            sr_last.n_events = 0
            assert nest.biological_time <= start + ASSERTION_TIME_THRESHOLD

        sr_last.stop = nest.biological_time
        sr_first.stop = nest.biological_time

        nest.Simulate(OFFSET)
        nest.Prepare()
        for item in tqdm(training_sequence):
            conn = nest.GetConnections(target=sequence_model.read_out_dict[item])
            conn.weight = list(reversed(conn.weight))
            nest.Run(stimulation_time)
            conn.weight = list(reversed(conn.weight))
        nest.Cleanup()


def replay_sequence(*, sequence_model: SequenceModel, replay_time: int):
    """ TODO documentation
    """
    sequence_model.sequence_selector.origin = nest.biological_time
    sequence_model.sequence_selector.stop = replay_time

    conns = nest.GetConnections(sequence_model.sequence_selector, sequence_model.r_neurons)
    prev_weights = conns.weight
    conns.weight = 0.0  # All other read-out populations are inhibited to suppress replaying multiple sequences at the same time

    nest.Simulate(replay_time)

    conns.weight = prev_weights


def main():
    """_summary_ TODO documentation
    """

    # ===============================================================
    # get all parameter sets
    # ===============================================================

    batch_id = 0
    task_id = 0
    job_max = 0

    if len(sys.argv) > 1:
        batch_id = sys.argv[1]

    if len(sys.argv) > 2:
        task_id = sys.argv[2]

    if len(sys.argv) > 3:
        job_max = sys.argv[3]

    parameterset_idx = batch_id * job_max + task_id

    # parameter-set id from command line (submission script)
    paramset_readout = helper.parameter_set_list(paramspace_readout)[parameterset_idx]
    paramset_readout['param_recurrent']['label'] = paramset_readout['label']

    # ===============================================================
    # specify simulation times + sequences
    # ===============================================================

    sim_time = paramset_readout['sim_time']
    recording_time = paramset_readout['recording_time']

    sequences = paramset_readout['sequences']

    # ===============================================================
    # create recurrent network + random input generators
    # ===============================================================
    # Create the recurrent network where the weights are already learned to exhibit sequential dynamics

    # ===============================================================
    # create sequence learning models
    # ===============================================================

    SequenceModel.setup(paramset_readout)

    sequence_models = {}
    for seq in sequences:
        sequence_models[seq] = SequenceModel(sequence=seq)

    # ===============================================================
    # learn all sequences
    # ===============================================================

    # Simulate training
    while nest.biological_time < sim_time:
        for sequence_model in sequence_models.values():
            simulate_sequence_learning(params_ro=paramset_readout, sequence_model=sequence_model)

    # ===============================================================
    # record spike behavior after learning sequences
    # ===============================================================

    # Record spike behavior after learning
    recording_setup = paramset_readout['recording_setup']
    sr_e, sr_i, sr_r = create_spike_recorders()
    connect_spike_recorders_to_rnn(model_instance=SequenceModel.rnn_model, sr_e=sr_e, sr_i=sr_i)

    for sequence_model in sequence_models.values():
        connect_spike_recorders_to_readout(model_instance=sequence_model, sr_r=sr_r)

        if recording_setup == 'disconnect_readout_generators':
            disconnect_readout_generators(gen_h=SequenceModel.gen_h, gen_s_baseline=SequenceModel.gen_s_baseline, gen_s=SequenceModel.gen_s)
        elif recording_setup == 'disconnect_readout_population':
            disconnect_readout_population(r_neurons=sequence_model.r_neurons, s_neurons=sequence_model.s_neurons, h_neurons=sequence_model.h_neurons)
        else:
            assert recording_setup == 'all_nodes'

    sr_r.origin = nest.biological_time
    sr_r.stop = recording_time
    sr_e.origin = nest.biological_time
    sr_e.stop = recording_time
    sr_i.origin = nest.biological_time
    sr_i.stop = recording_time

    # TODO after training, should the weights be frozen?

    for seq, replay_time in paramset_readout['replay_tuples']:
        replay_sequence(sequence_model=sequence_models[seq], replay_time=replay_time)

    save_spikes_after_sim(sr_e=sr_e, sr_i=sr_i, sr_r=sr_r, params_ro=paramset_readout)

    # ===============================================================
    # plot read-out weights + spike behavior
    # ===============================================================

    # results  # TODO instead of neuron ids rather plot their corresponding sequence element
    conns = nest.GetConnections(source=SequenceModel.rnn_model.exc_neurons, target=sequence_models['DEDED'].r_neurons)  # TODO include all read-out weights in plott
    conns = nest.GetStatus(conns, ['target', 'source', 'weight'])
    np.save(os.path.join('/Users/Jette/Desktop/results/NEST/job_3822329/1787e7674087ddd1d5749039f947a2cd/', 'readout_weights'), conns)  # TODO no absolute paths
    exc_spikes, inh_spikes, readout_spikes = plot_helper.load_spikes(filepath='/Users/Jette/Desktop/results/NEST/job_3822329/1787e7674087ddd1d5749039f947a2cd/', filename='spikes_after_learning_' + paramset_readout['recording_setup'] + '.pickle')
    figure, axes = plt.subplots(1, 3)
    figure.set_size_inches(17, 9)
    plot_helper.plot_weight_matrix(ax=axes[0], connections=conns, title='trained read-out synapses', cmap='viridis')
    if len(readout_spikes) > 0:
        plot_helper.plot_spikes(ax=axes[1], R_spikes=readout_spikes)
    plot_helper.plot_spikes(ax=axes[2], exh_spikes=exc_spikes, inh_spikes=inh_spikes)

    # TODO Save plot for later
    plt.show()


if __name__ == "__main__":
    main()
