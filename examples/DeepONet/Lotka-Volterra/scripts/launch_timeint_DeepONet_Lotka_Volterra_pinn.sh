#!/bin/bash 

# simulai configuration
export engine=pytorch
export PYTHONPATH=$HOME/kansas

device=gpu

# Environment configuration
export CUDA_HOME=/opt/share/cuda-10.1/x86_64
export CUDA_BIN_PATH=$CUDA_HOME/bin
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64

if [[ $device -eq gpu ]]; then
    bsub_string="jbsub -e ~/log/lotka_volterra.err -o ~/log/lotka_volterra.out -cores 1x20+1 -r v100"
else
    bsub_string="jbsub -e ~/log/lotka_volterra.err -o ~/log/lotka_volterra.out -cores 1x20"
fi

$bsub_string python timeint_DeepONet_Lotka_Volterra_pinn.py --save_path /dccstor/jlsa931/Lotka-Volterra_data/ 
