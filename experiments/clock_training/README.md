# Running "clock training" experiments

This folder contains two scripts that can be executed directly:

* `run_local.py`
* `run_cluster.py`

The script `run_cluster.py` needs to be run on a computer cluster supporting [Slurm](https://slurm.schedmd.com/overview.html). 

A third script, `training.py`, is usually not called directly.

What the scripts actually do is governed by the settings in [parameters_space.py](../parameters_space.py) in the parent directory. This file contains the parameters for both, clock and read-out training. However, only clock parameters are considered for "clock training" experiments.

## Prerequisites

See [here](../../README.md).


## Running an experiment

1. On your local machine:
   ```bash
   cd <mbc_replication>
   python ./experiments/clock_training/run_local.py
   ```
2. On computer cluster:
   ```bash 
   cd <mbc_replication>
   python ./experiments/clock_training/run_cluster.py
   ```

Results are stored in the [data directory](../data/) under the corresponding hash.
To visualize the results execute the scripts in [plotting](../mbc_network/plotting/).

## What is the "clock training" experiment?

This experiment corresponds to training the recurrent network (RNN) presented in [Maes et al. 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007606). After training, the recurrent network develops precise switching dynamics by successively activating and deactivating small groups of neurons. Effectively, a neural clock is created.
