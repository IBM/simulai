# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch

from simulai.models import DeepONet
from simulai.optimization import Optimizer
from simulai.regression import DenseNetwork

# In[2]:

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--data_path", type=str, help="Save path", default="/tmp")
args = parser.parse_args()

data_path = args.data_path


# In[3]:


datasets = np.load(data_path)


# In[4]:
# Reading the datasets

input_dataset_raw = datasets["input_dataset"]
output_dataset_raw = datasets["output_dataset"]
input_dataset_parameter = datasets["parameter_dataset"]
time_raw = datasets["time"]


# In[5]:
# Defining global parameters

time_interval = [0, 120]
n_cases = 80
n_cases_test = 20
n_sensors = 100
n_time_samples = 100
latent_dim = 100
n_vars = 2
activation = "tanh"
trunk_layers_units = [50, 50, 50]
branch_layers_units = [50, 50, 50]
n_inputs = 1
lr = 5e-4
lambda_1 = 0.0
lambda_2 = 1e-5
n_epochs = 500
device = "gpu"

# In[6]:

time_ = time_raw[time_raw <= time_interval[-1]]
time_indices = sorted(np.random.choice(time_.shape[0], n_time_samples))


# In[7]:


time = time_[time_indices]


# In[8]:
# Separating train and test datasets

output_dataset_train = output_dataset_raw[:, :, :n_cases]
output_dataset_test = output_dataset_raw[:, :, n_cases:]

input_dataset_train = input_dataset_parameter[:n_cases]
input_dataset_test = input_dataset_parameter[n_cases:]


# In[9]:
# Sampling over time (we will not use all the available samples, since the 
# final datasets can became large to much for our purposes)

output_dataset_time_sampled = output_dataset_train[time_indices, ...]
input_dataset_sensor_sampled = input_dataset_train


# In[10]:
# The input dataset for branch is the repetition of each
# parameters scenario for all the time samples
input_dataset = input_dataset_train[:, None, :]
input_dataset = np.tile(input_dataset, (1, n_time_samples, 1))
input_dataset = input_dataset.reshape((n_time_samples*n_cases, -1))
input_branch = torch.from_numpy(input_dataset.astype("float32"))

# The input dataset for trunk is the repetition of the time array (our
# coordinates) for each parameters scenario (the branch inputs)
input_trunk = np.tile(time[:, None], (n_cases, 1))

# The target dataset is organized in the way that we have the 
# correspondency: target_data(parameters, t) = G(parameters)(t)
target_data = output_dataset_time_sampled.transpose((2,0,1))
target_data = target_data.reshape((n_time_samples*n_cases, -1))

# In[11]:
# Verifying the dimensions of the final datasets
print("Final datasets:\n")
print(target_data.shape)
print(input_branch.shape)
print(input_trunk.shape)


# In[12]:

# Instantiating the neural network model

# Configuration for the fully-connected network
config_trunk = {
    "layers_units": trunk_layers_units,  # Hidden layers
    "activations": activation,
    "input_size": n_inputs,
    "output_size": latent_dim,
    "name": "trunk_net",
}

# Configuration for the fully-connected network
config_branch = {
    "layers_units": branch_layers_units,  # Hidden layers
    "activations": activation,
    "input_size": 2,
    "output_size": latent_dim,
    "name": "branch_net",
}

# Instantiating and training the surrogate model
trunk_net = DenseNetwork(**config_trunk)

# Instantiating and training the surrogate model
branch_net = DenseNetwork(**config_branch)

trunk_net.summary()
branch_net.summary()

optimizer_config = {"lr": lr}

# Maximum magnitudes to be used as loss weights
maximum_values = (1 / np.linalg.norm(target_data, 2, axis=0)).tolist()

params = {"lambda_1": lambda_1, "lambda_2": lambda_2, "weights": maximum_values}

input_data = {"input_branch": input_branch, "input_trunk": input_trunk}

# The DeepONet receives the two instances in order to construct
# the trunk and the branch components
op_net = DeepONet(
    trunk_network=trunk_net,
    branch_network=branch_net,
    var_dim=2,
    model_id="LotkaVolterra",
    devices="gpu",
)

# In[13]:

# Instantiating the optimizer and training
optimizer = Optimizer("adam", params=optimizer_config)

optimizer.fit(
    op=op_net,
    input_data=input_data,
    target_data=target_data,
    n_epochs=n_epochs,
    loss="wrmse",
    params=params,
    device="gpu",
)


# In[30]:


n_tests_choices = 100
test_indices = np.random.choice(n_cases_test, n_tests_choices)
time_test = np.linspace(0, time_interval[-1], 2000)[:, None]

for index in test_indices:
    target_test = output_dataset_test[:, :, index]
    input_test_ = input_dataset_test[index, :]
    input_test = np.tile(input_test_, (2000, 1))
    evaluation = op_net.eval(trunk_data=time_test, branch_data=input_test)

    plt.plot(time_raw, target_test[:, 0], label="Exact")
    plt.plot(time_test, evaluation[:, 0], label="Approximated")
    plt.savefig(f"evaluation_case_{index}.png")

    plt.legend()
    plt.xlim(0, 120)
    plt.show()
