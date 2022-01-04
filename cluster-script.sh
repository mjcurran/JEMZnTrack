#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -N jemzn-epoch-10
#$ -wd /afs/crc.nd.edu/user/m/mcurran2/python/JEMZnTrack/

# Force exit on error
set -e

# Setup Modules System
if [ -r /opt/crc/Modules/current/init/bash ]; then
    source /opt/crc/Modules/current/init/bash
fi

module use /afs/crc.nd.edu/user/j/jsweet/Public/module/file

# Run job
module load pdm
$(pdm info --package)/bin/dvc exp run --name jemzn-epoch-10 -S epochs=10

# Requires ssh deploy key without password
# $(pdm info --package)/bin/dvc push origin jem-epoch-10
