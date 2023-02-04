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

from argparse import ArgumentParser

import numpy as np

from simulai.math.differentiation import CollocationDerivative
from simulai.math.integration import RK4
from simulai.metrics import L2Norm
from simulai.simulation import Surrogate

"""
 We intend to use some specific
 dataset U and fit a surrogate model for it

 The surrogate construction is achieved with the support of
 three entities, namely:  Data (or a Simulation object in a physically-based approach),
                          Dimensionality Reduction Model and
                          Machine-Learning model
"""

# Reading command-line arguments
parser = ArgumentParser(description="Argument parsers")

parser.add_argument("--data_path", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--model_name", type=str)

args = parser.parse_args()

# The file format is not important
# at this moment. Let us to use simple Numpy files
data_path = args.data_path
save_path = args.save_path
model_name = args.model_name

# Constructing data
data = np.load(data_path)

variables_list = list(data.dtype.names)

n_batches = data.shape[0]
# Training data
train_data = data[: int(n_batches / 2), :, :, :]
# Testing data
test_data = data[int(n_batches / 2) :, :, :, :]

N_t = data.shape[0]
dt = 1 / N_t
dt_ = dt / 10

N_epochs = int((dt / dt_) * N_t / 2)

# The initial state is used to execute a time-integrator
# as will be seen below
initial_state = train_data[-1:, :, :, :]

# Machine learning model configuration
model_config = {
    "architecture": [50, 50, 50, 50, 50],  # Hidden layers only
    "dropouts_rates_list": [0, 0, 0, 0, 0],
    "learning_rate": 1e-05,
    "l2_reg": 1e-05,
    "activation_function": "elu",
    "loss_function": "mse",
    "optimizer": "adam",
}

# Fitting process configuration
fit_config = {
    "n_epochs": 10000,  # Just for testing purposes
    "use_second_order_opt": True,  # Default is True
}

rom_config = {"n_components": 5}

data_generator_config = {"step": dt}

# Instantiating the class Surrogate
surrogate = Surrogate(
    input_var_names=variables_list,
    data_preparer="reshaper",
    rom="pod",
    model="dense",
    data_generator=CollocationDerivative,
    rom_config=rom_config,
    model_config=model_config,
    data_generator_config=data_generator_config,
)

# Fitting
surrogate.fit(data=train_data, fit_config=fit_config)

# Saving surrogate to a file
surrogate.save(save_path=save_path, model_name=model_name)

print("Model fitting concluded.")

# Testing to execute an operation which iteratively calls the surrogate model
initial_state = surrogate.project_data(initial_state, variables_list)

rk4 = RK4(surrogate.eval)
extrapolation = rk4(initial_state=initial_state, epochs=N_epochs, dt=dt_, resolution=dt)

# Reconstructing the output.py of the machine learning model using the ROM
# reconstruction method
reconstructed_extrapolation = surrogate.reconstruct_data(extrapolation)

# Providing a metric and executing a test with the trained model
l2_norm = L2Norm()
error = surrogate.test(
    metric=l2_norm, data=test_data, reference_data=reconstructed_extrapolation
)
print("L2 error norm of the extrapolation {}".format(error))

print("Evaluation using the model concluded.")
