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

import matplotlib.pyplot as plt
import numpy as np

from simulai.io import Reshaper
from simulai.math.integration import RK4
from simulai.metrics import L2Norm
from simulai.normalization import UnitaryNormalization
from simulai.regression import DenseNetwork
from simulai.rom import POD
from simulai.simulation import Pipeline

# Reading command-line arguments
parser = ArgumentParser(description="Argument parsers")

parser.add_argument("--save_path", type=str)
parser.add_argument("--model_name", type=str)

args = parser.parse_args()

# The file format is not important
# at this moment. Let us to use simple Numpy files
save_path = args.save_path
model_name = args.model_name

# Constructing data
N = 100
Nt = 5000
lambd = 10
omega_t = 10 * np.pi

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
t = np.linspace(0, 1, Nt)

dt = 1.0 / Nt
dt_ = dt
N_epochs = int(dt / dt_) * int(Nt / 2)

X, T, Y = np.meshgrid(x, t, y)

# Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
U = np.exp(-lambd * T) * (X**2 * np.cos(Y) + X * Y)
# Time derivative of U
U_t = -lambd * U

data = np.core.records.fromarrays([U, U_t], names="U, U_t", formats="f8, f8")[:, None]

# Data rescaling
rescaler = UnitaryNormalization()
data = rescaler.rescale(data=data)

n_batches = data.shape[0]
# Training data
train_data = data[: int(n_batches / 2), :, :, :]
# Testing data
test_data = data[int(n_batches / 2) :, :, :, :]

input_data = np.core.records.fromarrays([train_data["U"]], names="U", formats="f8")[
    :, None
]

target_data = np.core.records.fromarrays(
    [train_data["U_t"]], names="U_t", formats="f8"
)[:, None]

# The initial state is used to execute a time-integrator
# as will be seen below
initial_state = train_data[-1:, :, :, :]

# Configurations
# Machine learning model configuration
architecture = [50, 50, 50, 50, 50]  # Hidden layers only

model_config = {
    "dropouts_rates_list": [0, 0, 0, 0, 0],
    "learning_rate": 1e-05,
    "l2_reg": 1e-05,
    "activation_function": "elu",
    "loss_function": "mse",
    "optimizer": "adam",
}

# Fitting process configuration
fit_config = {
    "n_epochs": 20000,  # Just for testing purposes
    "use_second_order_opt": True,  # Default is True
}

rom_config = {"n_components": 5}

extra_kwargs = {
    "initial_state": initial_state,
    "epochs": N_epochs,
    "dt": dt_,
    "resolution": dt,
}

l2_norm = L2Norm()

model = DenseNetwork(architecture=architecture, config=model_config)
model.restore(save_path, model_name)

pipeline = Pipeline(
    [("data_preparer", Reshaper()), ("rom", POD(config=rom_config)), ("model", model)]
)

pipeline.exec(
    data=data,
    input_data=input_data,
    target_data=target_data,
    reference_data=test_data,
    fit_kwargs=fit_config,
)

output = pipeline.predict(post_process_op=RK4, extra_kwargs=extra_kwargs)

error = l2_norm(data=output, reference_data=test_data["U"], relative_norm=True)

print("Evaluation error: {} %".format(error * 100))

plt.imshow(output[-1, 0, :, :])
plt.savefig(save_path + "solution_estimated.png")

plt.imshow(test_data["U"][-1, 0, :, :])
plt.savefig(save_path + "solution_expected.png")
