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
from simulai.normalization import StandardNormalization, UnitaryNormalization
from simulai.regression import DenseNetwork
from simulai.rom import POD
from simulai.simulation import Pipeline


def evaluate_reduced_var_and_time_derivatives(
    pipeline, test_data, input_data, variables_list
):
    test_data_reduced = pipeline.project_data(
        data=test_data, variables_list=variables_list, mean_component=True
    )
    input_data_reduced = pipeline.project_data(
        data=input_data, variables_list=variables_list, mean_component=True
    )

    config = {"step": dt}
    diff_op_raw = CollocationDerivative(config=config)
    derivatives_test_data_reduced = diff_op_raw(data=test_data_reduced)
    derivatives_input_data_reduced = diff_op_raw(data=input_data_reduced)

    return (
        test_data_reduced,
        input_data_reduced,
        derivatives_test_data_reduced,
        derivatives_input_data_reduced,
    )


def test_projection_error(pipeline, input_data, test_data, variables_list, l2_norm):
    projected = pipeline.project_data(data=input_data, variables_list=variables_list)
    reconstructed = pipeline.reconstruct_data(data=projected)

    projected_test = pipeline.project_data(
        data=test_data, variables_list=variables_list
    )
    reconstructed_test = pipeline.reconstruct_data(data=projected_test)

    input_data_numeric = np.hstack(
        [input_data[name] for name in input_data.dtype.names]
    )
    test_data_numeric = np.hstack([test_data[name] for name in test_data.dtype.names])

    error = l2_norm(
        data=reconstructed, reference_data=input_data_numeric, relative_norm=True
    )
    error_test = l2_norm(
        data=reconstructed_test, reference_data=test_data_numeric, relative_norm=True
    )

    print("Projection error for the training data: {} %".format(100 * error))
    print("Projection error for the testing data: {} %".format(100 * error_test))


def test_derivatives_error(pipeline, test_data_reduced, input_data_reduced, l2_norm):
    derivatives_output_test = pipeline.eval(
        data=test_data_reduced, with_projection=False, with_reconstruction=False
    )
    derivatives_output = pipeline.eval(
        data=input_data_reduced, with_projection=False, with_reconstruction=False
    )

    derivatives_error_test = l2_norm(
        data=derivatives_output_test,
        reference_data=derivatives_test_data_reduced,
        relative_norm=True,
    )

    derivatives_error = l2_norm(
        data=derivatives_output,
        reference_data=derivatives_input_data_reduced,
        relative_norm=True,
    )

    print("Evaluation error: {} %".format(derivatives_error_test * 100))
    print("Evaluation error: {} %".format(derivatives_error * 100))

    return derivatives_output_test, derivatives_output


def test_extrapolation_error(pipeline, test_data, RK4, extra_kwargs):
    output = pipeline.predict(post_process_op=RK4, extra_kwargs=extra_kwargs)

    error = l2_norm(data=output, reference_data=test_data, relative_norm=True)

    print("Evaluation error: {} %".format(error * 100))

    errors_list = list()
    for tt in range(output.shape[0]):
        error = l2_norm(
            data=output[tt, :, :, :],
            reference_data=test_data[tt, :, :, :],
            relative_norm=True,
        )
        errors_list.append(error)

    return output, errors_list


# Reading command-line arguments
parser = ArgumentParser(description="Argument parsers")

parser.add_argument("--data_path", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--case", type=str)

args = parser.parse_args()

# The file format is not important
# at this moment. Let us using simple Numpy files
data_path = args.data_path
save_path = args.save_path
model_name = args.model_name
case = args.case

save_path = os.path.join(save_path, model_name)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# Constructing data
# Loading simulation data
data = np.load(data_path)

# Correcting the shape of the input data.
# It must be (n_batches, n_variables, **(other dimensions))
data = data.transpose(0, 3, 1, 2)

rescaler = UnitaryNormalization()

# Number of timesteps
N_t = data.shape[0]

# This values are just for testing
# purposes and considers a previous unitary
# transformation
t_max = 1
t_min = 0

# Size of the results timestep
dt = (t_max - t_min) / N_t
# Size of the time-integration timestep
dt_ = dt
frac = 0.9

# Number of the time iterations
N_epochs = round((1 - frac) * (dt / dt_) * N_t)

# Problem variables names
variables_list = list(data.dtype.names)

# Convert the simulation data to simple NumPy arrays (it is originally a structured array)
input_data_ = np.concatenate([data[var] for var in variables_list], 1)

# Generating the time-derivatives of simulation data
config = {"step": dt}
diff_op = CollocationDerivative(config=config)

# The number of training batches corresponds to the number of timesteps
n_batches = input_data_.shape[0]

# Training data
input_data = data[: int(frac * n_batches), :, :, :]
test_data_ = data[int(frac * n_batches) :, :, :, :]

# Training data rescaling
data_dict = rescaler.rescale(map_dict={"input_data": input_data})
input_data = data_dict["input_data"]

# Testing data rescaling
data_dict = rescaler.rescale(map_dict={"test_data": test_data_})
test_data_ = data_dict["test_data"]

# Testing data
test_data = np.concatenate([test_data_[var] for var in variables_list], 1)

# The initial state is used to execute a time-integrator
# as will be seen below
initial_state = input_data[-1:, :, :, :]

# Configurations

# The configurations of the operation classes are made via
# Python dictionaries

# Machine learning model configuration
architecture = [50, 50]  # Hidden layers only

adam_config = {"learning_rate": 1e-05, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-08}
scipy_config = {
    "method": "L-BFGS-B",
    "options": {
        "maxiter": 50000,
        "maxfun": 50000,
        "maxcor": 50,
        "maxls": 50,
        "ftol": 1.0 * np.finfo(float).eps,
    },
}

# Other setup choices for the machine learning model
model_config = {
    "dropouts_rates_list": [0.0, 0.0],
    "l2_reg": 1e-06,
    "l1_reg": 1e-06,
    "activation_function": "tanh",
    "loss_function": "mse",
    "optimizers": {
        "adam": {"n_epochs": 2000, "config": adam_config},
        "ScipyOptimizerInterface": {"config": scipy_config},
    },
}

# There is only one configuration variable for the linear POD,
# the number of principal components
rom_config = {"n_components": 25}

# Extra kwargs for the time-integrator
extra_kwargs = {
    "initial_state": initial_state,
    "epochs": N_epochs,
    "dt": dt_,
    "resolution": dt,
}

# Condition for defining the kind of execution,
if case == "fit":
    model = DenseNetwork(architecture=architecture, config=model_config)

elif case == "restore":
    model = DenseNetwork(architecture=architecture, config=model_config)
    model.restore(save_path, model_name)

else:
    raise Exception(
        "It is necessary to provide the way" "for obtaining the machine learning model"
    )

l2_norm = L2Norm()

# Instantiating the class Pipeline
pipeline = Pipeline(
    stages=[
        ("data_preparer", Reshaper()),
        ("rom", POD(config=rom_config)),
        ("normalization", StandardNormalization()),
        ("model", model),
    ]
)

# Executing the workflow shown in the list stages
pipeline.exec(
    data=input_data,
    input_data=input_data,
    reference_data=test_data,
    data_generator=diff_op,
)

# Saving the model to a binary meta file
if case == "fit":
    pipeline.save(save_path=save_path, model_name=model_name)

# Time derivatives of the target data
(
    test_data_reduced,
    input_data_reduced,
    derivatives_test_data_reduced,
    derivatives_input_data_reduced,
) = evaluate_reduced_var_and_time_derivatives(
    pipeline, test_data_, input_data, variables_list
)

### Testing ###

## Test 0: evaluating the ROM projection error
test_projection_error(pipeline, input_data, test_data_, variables_list, l2_norm)

## Test 1: estimating the target data (time derivatives) using the trained model
derivatives_output_test, derivatives_output = test_derivatives_error(
    pipeline, test_data_reduced, input_data_reduced, l2_norm
)

## Test 2: time-extrapolating using the trained model and a time-integration algorithm
# for the original problem dimensions
output, errors_list = test_extrapolation_error(pipeline, test_data, RK4, extra_kwargs)

## Test 3: time-extrapolating using the trained model and a time-integration algorithm
# for the reduced problem dimensions
output_reduced = pipeline.predict(
    post_process_op=RK4, extra_kwargs=extra_kwargs, with_reconstruction=False
)
input_vars_list = list(initial_state.dtype.names)
test_data_reduced = pipeline.project_data(
    test_data_, input_vars_list, mean_component=True
)

modes_ = pipeline.rom.modes
modes = pipeline.data_preparer.prepare_output_data(modes_)

### Post-processing ###

# Post-processing for visualization purposes
for vv in range(output.shape[1]):
    plt.imshow(output[-1, vv, :, :])
    plt.colorbar()
    plt.savefig(save_path + "solution_estimated_{}.png".format(vv))
    plt.close()

    plt.imshow(test_data[-1, vv, :, :])
    plt.colorbar()
    plt.savefig(save_path + "solution_expected_{}.png".format(vv))
    plt.close()

plt.plot(errors_list)
plt.savefig(save_path + "error_series.png")
plt.close()

for ss in range(output_reduced.shape[1]):
    plt.plot(output_reduced[:, ss], label="Estimated")
    plt.plot(test_data_reduced[:, ss], label="Expected")
    plt.legend()
    plt.savefig(save_path + "coefficients_series_{}.png".format(ss))
    plt.close()

    plt.plot(derivatives_output_test[:, ss], label="Estimated")
    plt.plot(derivatives_test_data_reduced[:, ss], label="Expected")
    plt.legend()
    plt.savefig(save_path + "derivatives_coefficients_series_{}.png".format(ss))
    plt.close()

    plt.plot(derivatives_output[:, ss], label="Estimated")
    plt.plot(derivatives_input_data_reduced[:, ss], label="Expected")
    plt.legend()
    plt.savefig(save_path + "train_derivatives_coefficients_series_{}.png".format(ss))
    plt.close()

n_modes = modes.shape[0]
n_variables = modes.shape[1]

for mm in range(n_modes):
    for vv in range(n_variables):
        plt.imshow(modes[mm, vv, :, :])
        plt.colorbar()
        plt.savefig(save_path + "mode_{}_var{}.png".format(mm, vv))
        plt.close()
