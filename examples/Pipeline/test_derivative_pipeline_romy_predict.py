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

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from simulai.io import Reshaper
from simulai.math.differentiation import CollocationDerivative
from simulai.math.integration import RK4
from simulai.metrics import L2Norm
from simulai.normalization import UnitaryNormalization
from simulai.regression import DenseNetwork
from simulai.rom import POD
from simulai.simulation import Pipeline


def time_nonlinear_system(X, Y, T):
    omega_t = 3 * np.pi

    # Generic function U = (x**2)*cos(omega_t*t*y) + (t**2)*x*y
    U = (X**2) * np.cos(omega_t * T * Y) + (T**2) * X * Y
    # Time derivative of U
    U_t = -omega_t * Y * np.sin(omega_t * T * Y) * (X**2) + 2 * T * X * Y

    U_nl = -omega_t * Y * np.sin(omega_t * T * Y) * (X**2)
    U_l = 2 * T * X * Y

    u_nl_intg = np.linalg.norm(U_nl.flatten(), 2)
    u_intg = np.linalg.norm((U_l + U_nl).flatten(), 2)

    print(
        "Nonlinear contribution ratio to the derivatives: {}".format(u_nl_intg / u_intg)
    )

    return U, U_t


def time_linear_system(X, Y, T):
    lambd = 1

    # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
    U = np.exp(-lambd * T) * (X**2 * np.cos(Y) + X * Y)
    # Time derivative of U
    U_t = -lambd * U

    return U, U_t


function_switcher = {"linear": time_linear_system, "nonlinear": time_nonlinear_system}

# Reading command-line arguments
parser = ArgumentParser(description="Argument parsers")

parser.add_argument("--save_path", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--problem", type=str)
parser.add_argument("--case", type=str)

args = parser.parse_args()

# The file format is not important
# at this moment. Let us to use simple Numpy files
save_path = args.save_path
model_name = args.model_name
problem = args.problem
case = args.case

save_path = save_path + "/" + model_name + "/"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# Constructing data
N = 100
Nt = 5000

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
t = np.linspace(0, 1, Nt)

X, T, Y = np.meshgrid(x, t, y)

system = function_switcher.get(problem)

U, U_t = system(X, Y, T)

data = np.core.records.fromarrays([U, U_t], names="U, U_t", formats="f8, f8")[:, None]

dt = 1.0 / Nt
dt_ = dt
N_epochs = int(dt / dt_) * int(Nt / 2)

n_batches = data.shape[0]
# Training data
train_data = data[: int(n_batches / 2), :, :, :]
# Testing data
test_data = data[int(n_batches / 2) :, :, :, :]

# Preparing multiple data arrays
input_data = np.core.records.fromarrays([train_data["U"]], names="U", formats="f8")[
    :, None
]

target_data = np.core.records.fromarrays(
    [train_data["U_t"]], names="U_t", formats="f8"
)[:, None]

test_input_data = np.core.records.fromarrays([test_data["U"]], names="U", formats="f8")[
    :, None
]

test_target_data = np.core.records.fromarrays(
    [test_data["U_t"]], names="U_t", formats="f8"
)[:, None]

# The initial state is used to execute a time-integrator
# as will be seen below
initial_state = train_data[-1:, :, :, :]

# Configurations
# Machine learning model configuration
# Hidden layers only
if problem == "linear":
    architecture = [100, 100, 100, 100]

    # Machine learning configuration model
    model_config = {
        "dropouts_rates_list": [0, 0],
        "learning_rate": 1e-05,
        "l2_reg": 1e-06,
        "activation_function": "elu",
        "loss_function": "mse",
        "optimizer": "adam",
    }

    # Fitting process configuration
    fit_config = {
        "n_epochs": 20000,  # Just for testing purposes
        "use_second_order_opt": True,  # Default is True
    }

    # ROM config (in this case, POD)
    rom_config = {"n_components": 2}

if problem == "nonlinear":
    architecture = [100, 100]

    # Machine learning configuration model
    model_config = {
        "dropouts_rates_list": [0, 0],
        "learning_rate": 1e-05,
        "l2_reg": 1e-05,
        "activation_function": "tanh",
        "loss_function": "mse",
        "optimizer": "adam",
    }

    # Fitting process configuration
    fit_config = {
        "n_epochs": 20000,  # Just for testing purposes
        "use_second_order_opt": True,  # Default is True
    }

    # ROM config (in this case, POD)
    rom_config = {"n_components": 10}

# Time-integration configuration
extra_kwargs = {
    "initial_state": initial_state,
    "epochs": N_epochs,
    "dt": dt_,
    "resolution": dt,
}

if case == "fit":
    model = DenseNetwork(architecture=architecture, config=model_config)

elif case == "restore":
    model = DenseNetwork(architecture=architecture, config=model_config)
    model.restore(save_path, model_name)

else:
    raise Exception(
        "It is necessary to provide the way" "for obtaining the machine learning model"
    )

# Derivative operator
diff_config = {"step": dt}
diff_op = CollocationDerivative(config=diff_config)

pipeline = Pipeline(
    stages=[
        ("data_preparer", Reshaper()),
        ("rom", POD(config=rom_config)),
        ("model", model),
    ]
)

pipeline.exec(
    data=train_data,
    input_data=input_data,
    reference_data=test_data,
    data_generator=diff_op,
    fit_kwargs=fit_config,
)

if case == "fit":
    pipeline.save(save_path=save_path, model_name=model_name)

estimated_target_data = pipeline.eval(data=test_input_data)

output = pipeline.predict(post_process_op=RK4, extra_kwargs=extra_kwargs)

l2_norm = L2Norm()
error_target = l2_norm(
    data=estimated_target_data,
    reference_data=test_target_data["U_t"],
    relative_norm=True,
)

error_nominal_list = list()
for ii in range(output.shape[0]):
    error_nominal = l2_norm(
        data=output[ii, :, :, :],
        reference_data=test_input_data["U"][ii, :, :, :],
        relative_norm=True,
    )

    error_nominal_list.append(error_nominal)

error_nominal = l2_norm(
    data=output, reference_data=test_input_data["U"], relative_norm=True
)

print("Evaluation error: {} %".format(error_target * 100))
print("Evaluation error: {} %".format(error_nominal * 100))

plt.plot(list(range(len(error_nominal_list))), error_nominal_list)
plt.savefig(save_path + "error_curve.png")
plt.close()

plt.imshow(output[-1, 0, :, :])
plt.colorbar()
plt.savefig(save_path + "solution_estimated.png")
plt.close()

plt.imshow(test_data["U"][-1, 0, :, :])
plt.colorbar()
plt.savefig(save_path + "solution_expected.png")
plt.close()

plt.imshow(estimated_target_data[-1, 0, 0, :, :])
plt.colorbar()
plt.savefig(save_path + "derivative_estimated.png")
plt.close()

plt.imshow(test_target_data["U_t"][-1, 0, 0, :, :])
plt.colorbar()
plt.savefig(save_path + "derivative_expected.png")
plt.close()
