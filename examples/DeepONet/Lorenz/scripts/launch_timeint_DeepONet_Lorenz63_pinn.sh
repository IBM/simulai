#!/bin/bash 

# simulai configuration
export engine=pytorch
export PYTHONPATH=$HOME/kansas

# Environment configuration
export CUDA_HOME=/opt/share/cuda-10.1/x86_64
export CUDA_BIN_PATH=$CUDA_HOME/bin
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64

#jbsub -e ~/log/job.err -o ~/log/job.out -cores 1x20+1 -r v100  python timeint_DeepONet_Lotka_Volterra_pinn.py

jbsub -e ~/log/job.err -o ~/log/job.out -cores 1x20 python timeint_DeepONet_Lorenz63_pinn.py
