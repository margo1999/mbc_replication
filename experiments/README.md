Inkludiere README files von allen Experimenten

# Running experiments

This folder contains all experiments regarding the model of [Maes et al. 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007606):

* `clock_training`
* `sequence_training`

The file `parameters_space.py` govers the experiments and contains the parameters for both, clock training (`param_recurrent`) and sequence training (`param_readout`).

The `data` directory stores the simulation results under the corresponding hash value, which is calculated from the values in `parameters_space.py`.

## Prerequisites

1. Check that all [requiremts](../README.md#software-dependencies) are fullfilled.
1. Activate conda environment `mbc_replication`
```bash
   conda activate mbc_replication
   ``` 


## Running an experiment

For running `clock_training` see [here](clock_training/README.md).

For running `sequence_training` see [here](sequence_training/README.md).

To visualize data execute the scripts in [plotting](../mbc_network/plotting/).
