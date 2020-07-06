#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh
conda activate learning-module

# Use custom port, or else this might clash with host redis server when container uses host network
redis-server --daemonize yes --port 12345

python create_yaml_file.py config.yml /tmp /models "$@"
python learning_module.py config.yml
