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
from simulai.math.differentiation import CollocationDerivative
from simulai.math.integration import RK4
from simulai.metrics import L2Norm
from simulai.normalization import UnitaryNormalization
from simulai.regression import DenseNetwork
from simulai.rom import POD
from simulai.simulation import Pipeline

# Reading command-line arguments
parser = ArgumentParser(description="Argument parsers")

parser.add_argument("--data_path", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--case", type=str)

args = parser.parse_args()

# The file format is not important
# at this moment. Let us to use simple Numpy files
data_path = args.data_path
save_path = args.save_path
model_name = args.model_name
case = args.case

# Constructing data
# Loading simulation data
data = np.load(data_path)
# Correcting the shape of the input data.
# It must be (n_batches, n_variables, **(other dimensions))

data = data.transpose((0, 2, 1))
rescaler = UnitaryNormalization()
data_dict = rescaler.rescale(map_dict={"data": data})
data = data_dict["data"]

# Problem variables names
variables_list = list(data.dtype.names)

# Convert the simulation data to simple NumPy arrays (it is originally a structured array)
input_data_ = np.concatenate([data[var] for var in variables_list], 1)

n_batches = data.shape[0]
frac = 0.9

# Training data
input_data = data[: int(frac * n_batches), :, :]
test_data = data[int(frac * n_batches) :, :, :]

dt = 1
dt_ = dt
T_max = test_data.shape[0]
N_epochs = int(dt / dt_) * T_max

# Configurations
rom_config = {"n_components": 2}

# Machine learning configuration model
initial_state = input_data[-1, :, :]

architecture = [100, 100, 100]

model_config = {
    "dropouts_rates_list": [0, 0, 0],
    "learning_rate": 1e-05,
    "l2_reg": 1e-05,
    "activation_function": "tanh",
    "loss_function": "mse",
    "optimizer": "adam",
}

# Fitting process configuration
fit_config = {
    "n_epochs": 40000,  # Just for testing purposes
    "use_second_order_opt": True,  # Default is True
}

# Time-integration configuration
extra_kwargs = {
    "initial_state": initial_state,
    "epochs": N_epochs,
    "dt": dt_,
    "resolution": dt,
}

# Condition for defining the kind of execution.
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

l2_norm = L2Norm()

# Pipeline instantiation
pipeline = Pipeline(
    stages=[
        ("data_preparer", Reshaper()),
        ("rom", POD(config=rom_config)),
        ("model", model),
    ]
)

# It executes the sequence defined in the list stages
pipeline.exec(
    data=input_data,
    input_data=input_data,
    reference_data=test_data,
    data_generator=diff_op,
    fit_kwargs=fit_config,
)

if case == "fit":
    pipeline.save(save_path=save_path, model_name=model_name)

projected = pipeline.project_data(data=input_data, variables_list=variables_list)
reconstructed = pipeline.reconstruct_data(data=projected)

projected_test = pipeline.project_data(data=test_data, variables_list=variables_list)
reconstructed_test = pipeline.reconstruct_data(data=projected_test)

input_data_numeric = np.hstack([input_data[name] for name in input_data.dtype.names])
test_data_numeric = np.hstack([test_data[name] for name in test_data.dtype.names])

error = l2_norm(
    data=reconstructed, reference_data=input_data_numeric, relative_norm=True
)
error_test = l2_norm(
    data=reconstructed_test, reference_data=test_data_numeric, relative_norm=True
)

print("Projection error for the training data: {} %".format(100 * error))
print("Projection error for the testing data: {} %".format(100 * error_test))

output = pipeline.predict(post_process_op=RK4, extra_kwargs=extra_kwargs)

error_nominal = l2_norm(
    data=output, reference_data=test_data_numeric, relative_norm=True
)

print("Extrapolation error: {} % ".format(100 * error_nominal))

# Post-processing the outputs for visualization purposes
grid_shape = (15, 25, 24)
final_shape = output.shape[:-1] + grid_shape

output_reshaped = output.reshape(final_shape)
test_data_reshaped = test_data_numeric.reshape(final_shape)

plt.imshow(output_reshaped[-1, -1, :, 0, :])
plt.colorbar()
plt.savefig(save_path + "pressure_estimated.png")
plt.close()

plt.imshow(test_data_reshaped[-1, -1, :, 0, :])
plt.colorbar()
plt.savefig(save_path + "pressure_exact.png")
plt.close()

plt.imshow(output_reshaped[-1, 0, :, 0, :])
plt.colorbar()
plt.savefig(save_path + "saturation_estimated.png")
plt.close()

plt.imshow(test_data_reshaped[-1, 0, :, 0, :])
plt.colorbar()
plt.savefig(save_path + "saturation_exact.png")
plt.close()
