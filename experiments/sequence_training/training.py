"""
This script executes a single "sequence training" experiment and is usually called by one of the submission cripts
(run_local.py, run_cluster).
The script instantiates the complete MBC model, creates the recurrent network (ClockModel) and read-out layer (SequenceModel),
connects them, simulates the training of sequences as well as the replay and produces the corresponding figures.

Training protocol:
During learning of the read-out synapses, external input drives both the supervisor neuron (S) as well as interneurons (H).
Each element of the sequence is assigned to one of the read-out neurons (R) and its supervisor neuron. Thus, an element
of the sequence is learned by producing and sending a spike train to the associated S neuron for a certain time interval.
Meanwhile all other supervisor neurons receive a baseline input.
Before the sequence is learned, the RNN must first exhibit sequential dynamics, which is ensured by stimulating the network
using spontaneous input. After the sequential dynamics has stabilized, the sequence is presented to the network by sequentially
stimulating the associated supervisor neuron for each element. Throughout the learning, each interneuron receives constant input. 
The sequence is repeatedly shown to the network, with each element always appearing at the same time relative to the activation
of the first cluster in the RNN.
After training has been completed, the supervisor neurons and interneurons no longer receive external input. Replay of the
learned sequence is triggered by external spontaneous input to the recurrent network alone.

Authors
~~~~~~~
Jette Oberlaender
"""

import os
import sys

import nest
import numpy as np

from mbc_network.models.sequence_model import SequenceModel
import mbc_network.helper.training_helper as training_helper
import mbc_network.helper.plot_helper as plot_helper
from experiments.parameters_space import param_readout as paramspace_readout


def train_and_replay_sequences():  # TODO add timing stats
    """
    This function controls the complete process from learning the sequences by modifying read-out weights,
    and sequence replay to plotting the results. First, the read-out parameter set corresponding to the job
    is created, including the recurrent parameter set. Then the sequences to be learned and simulation times
    are read out to create a new SequenceModel object for each sequence. Subsequently, the sequences are
    learned one after the other but with several repetitions. This is followed by a replay of the sequences
    and the plotting of the weights and the spike behavior.
    """

    # ===============================================================
    # get parameter set
    # ===============================================================

    task_id = 0

    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])

    parameterset_idx = task_id

    # parameter-set id from command line (submission script)
    paramset_readout = training_helper.parameter_set_list(paramspace_readout)[parameterset_idx]
    paramset_readout['param_recurrent']['label'] = training_helper.compute_parameter_set_hash(paramset_readout['param_recurrent'])
    print(f"{paramset_readout['param_recurrent']['label']} is the generated hash based on the recurrent parameter set...")
    print(f"{paramset_readout['label']=} is the generated hash based on the read-out parameter set...")

    data_path = training_helper.get_data_path(paramset_readout['data_path'], paramset_readout['label'])

    # ===============================================================
    # specify simulation times + sequences
    # ===============================================================

    sim_time = paramset_readout['sim_time']
    recording_time = paramset_readout['recording_time']

    sequences = paramset_readout['sequences']

    if len(sequences) == 0:
        sequences, _, _ = training_helper.generate_sequences(paramset_readout['task'], paramset_readout['data_path'])  # TODO check if this works

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
            SequenceModel.simulate_sequence_learning(params_ro=paramset_readout, sequence_model=sequence_model)

    # ===============================================================
    # replay sequences and record spike behavior
    # ===============================================================

    # Record spike behavior after learning
    recording_setup = paramset_readout['recording_setup']
    sr_e, sr_i, sr_r = SequenceModel.create_spike_recorders()
    SequenceModel.connect_spike_recorders_to_rnn(model_instance=SequenceModel.rnn_model, sr_e=sr_e, sr_i=sr_i)

    for sequence_model in sequence_models.values():
        SequenceModel.connect_spike_recorders_to_readout(model_instance=sequence_model, sr_r=sr_r)

        if recording_setup == 'disconnect_readout_generators':
            SequenceModel.disconnect_readout_generators(gen_h=SequenceModel.gen_h, gen_s_baseline=SequenceModel.gen_s_baseline, gen_s=SequenceModel.gen_s)
        elif recording_setup == 'disconnect_readout_population':
            SequenceModel.disconnect_readout_population(r_neurons=sequence_model.r_neurons, s_neurons=sequence_model.s_neurons, h_neurons=sequence_model.h_neurons)
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
        SequenceModel.replay_sequence(sequence_model=sequence_models[seq], replay_time=replay_time)

    SequenceModel.save_spikes_after_sim(sr_e=sr_e, sr_i=sr_i, sr_r=sr_r, params_ro=paramset_readout, data_path=data_path)

    # ===============================================================
    # plot read-out weights + spike behavior
    # ===============================================================

    # results  # TODO instead of neuron ids rather plot their corresponding sequence element
    # r_neurons = nest.NodeCollection()
    # for sequence_model in sequence_models.values():
    #     r_neurons += sequence_model.r_neurons

    conns = nest.GetConnections(source=SequenceModel.rnn_model.exc_neurons, target=SequenceModel.all_r_neurons())
    conns = nest.GetStatus(conns, ['target', 'source', 'weight'])
    np.save(os.path.join(data_path, 'readout_weights'), conns)
    exc_spikes, inh_spikes, readout_spikes = plot_helper.load_spikes(filepath=data_path, filename='spikes_after_learning_' + paramset_readout['recording_setup'] + '.pickle')
    figure, axes = plt.subplots(1, 3)
    figure.set_size_inches(17, 9)
    plot_helper.plot_weight_matrix(ax=axes[0], connections=conns, title='trained read-out synapses', cmap='viridis')
    if len(readout_spikes) > 0:
        plot_helper.plot_spikes(ax=axes[1], R_spikes=readout_spikes)
    plot_helper.plot_spikes(ax=axes[2], exh_spikes=exc_spikes, inh_spikes=inh_spikes)

    # TODO Save plot for later


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from mbc_network.plotting import plot_results
    # matplotlib.use("Agg")

    train_and_replay_sequences()
    plt.show()
