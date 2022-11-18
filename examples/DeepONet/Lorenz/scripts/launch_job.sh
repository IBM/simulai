#!/bin/bash

# simulai configuration
export engine=pytorch

# Environment configuration
export CUDA_HOME=/opt/share/cuda-10.1/x86_64
export CUDA_BIN_PATH=$CUDA_HOME/bin
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64

# Basic choices
device=$1
echo "Device selected: $device"

log_path=/dccstor/jlsa931/log
queue=x86_12h
gpu=v100
cores=20
mem_limit=100G
data_path=/dccstor/jlsa931/Lorenz_data
device=gpu

# Selecting the execution type
if [ $device == cpu ]; then
        echo "Using CPU."
        exec_string="jbsub -e $log_path/job.err -o $log_path/job.out -q $queue -mem $mem_limit -cores $cores"
elif [ $device == gpu ]; then
        echo "Using GPU."
        exec_string="jbsub -e $log_path/job.err -o $log_path/job.out -q $queue -mem $mem_limit -r $gpu -cores 1x8+1"
else
        echo "The device $device is not available."
fi

$exec_string python lorenz_63_integrator_deeponet.py --data_path $data_path --device $device

