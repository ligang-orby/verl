#!/bin/bash

# Check if exactly one argument is given
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_python_trainer>"
    exit 1
fi

# Path to the Python program from the argument
PYTHON_PROGRAM=$1

# Check the NODE_RANK environment variable
if [ "$NODE_RANK" = "0" ]; then
    # First node will be Ray head
    ray start --head --dashboard-host=0.0.0.0 --node-ip-address=$(hostname -I | awk '{print $1}') --gcs-port 6379

    # python "$PYTHON_PROGRAM" /mnt/config/parameters.yaml --ray-address $MASTER_ADDR:$MASTER_PORT
    ray stop
else
    # wait for the head node to start
    sleep 60
    ray start --address $(hostname -I | awk '{print $1}'):6379 --block
fi