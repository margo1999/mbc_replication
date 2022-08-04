# Running "sequence training" experiments

This folder contains two scripts that can be executed directly:

* `run_local.py`
* `run_cluster.py`

The script `run_cluster.py` needs to be run on a computer cluster supporting [Slurm](https://slurm.schedmd.com/overview.html). 

A third script, `training.py`, is usually not called directly.

What the scripts actually do is governed by the settings in [parameters_space.py](../parameters_space.py) in the parent directory. This file contains the parameters for both, clock training (`param_recurrent`) and sequence training (`param_readout`). However, based on the settings in `param_recurrent`, the corresponding learned RNN weights are used for the sequence training. If these learned weights do not yet exist, then either the [clock training experiment](../clock_training/) must run first or a path to already learned weights must be set in [parameter_space.py](../parameters_space.py).

## Prerequisites

See [here](../../README.md).

## Running an experiment

1. On your local machine:
   ```bash
   cd <mbc_replication>
   python ./experiments/sequence_training/run_local.py
   ```
2. On computer cluster:
   ```bash 
   cd <mbc_replication>
   python ./experiments/sequence_training/run_cluster.py
   ```

Results are stored in the [data directory](../data/) under the corresponding hash.
To visualize the results execute the scripts in [plotting](../mbc_network/plotting/).

## What is the "sequence training" experiment?

This experiment corresponds to training the read-out layer presented in [Maes et al. 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007606). After training, the clock represented by the RNN drives the read-out layer such that the sequences can be replayed by the spike behavior of read-out neurons. Each read-out neuron is thereby associated with a distinct element of one sequence.
