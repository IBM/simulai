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

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from simulai.io import BatchCopy, MapValid
from simulai.metrics import L2Norm, MemorySizeEval
from simulai.models import ModelPool
from simulai.normalization import (
    BatchNormalization,
    UnitaryNormalization,
    UnitarySymmetricalNormalization,
)
from simulai.rom import IPOD
from simulai.simulation import Pipeline


# Testing the projection and reconstruction pipeline for
# each dataset
def construct_and_test_pipeline(
    dataset_, fraction=None, n_components=None, batch_size=None, dataset_name=None
):
    n_samples = int(fraction * dataset_.shape[0])
    variables_names = list(dataset_.dtype.names)
    n_variables = len(variables_names)

    # Data rescaling
    # Let us to rescale the data before entering in the Pipeline execution in order to avoid
    # problem with the mini-batch loop
    if norm_usage:
        rescaler = BatchNormalization(norm=UnitaryNormalization())
        nldas_dataset_norm = rescaler.transform(
            data=dataset_,
            data_interval=[0, n_samples],
            batch_size=batch_size,
            dump_path=os.path.join(data_path, dataset_name + "_normalization.h5"),
        )
        dataset = nldas_dataset_norm
    else:
        dataset = dataset_

    data_preparer_config = {}

    rom_config = {"n_components": n_components, "mean_component": True}

    pipeline = Pipeline(
        stages=[
            ("data_preparer", MapValid(config=data_preparer_config)),
            ("rom", IPOD(config=rom_config)),
        ]
    )

    pipeline.exec(
        input_data=dataset, data_interval=[0, n_samples], batch_size=batch_size
    )

    # The projected data usually can be allocated in memory
    projected = pipeline.project_data(
        data=dataset,
        data_interval=[0, n_samples],
        variables_list=variables_names,
        batch_size=batch_size,
    )

    reconstructed = pipeline.reconstruct_data(
        data=projected,
        data_interval=[0, n_samples],
        variables_list=variables_names,
        batch_size=batch_size,
        dump_path=os.path.join(data_path, dataset_name + "reconstruction.h5py"),
    )

    l2_norm = L2Norm()
    error = l2_norm(
        data=reconstructed,
        reference_data=dataset,
        relative_norm=True,
        data_interval=[0, n_samples],
    )

    print("Projection error: {} %".format(100 * error))

    return projected, reconstructed, pipeline


# After validating the pipeline, use it for reducing dimension
# and creating latent space time-series
def prepare_time_series(
    pipeline=None, dataset=None, dataset_name=None, batch_size=None
):
    n_samples = dataset.shape[0]
    if norm_usage:
        rescaler = BatchNormalization(norm=UnitaryNormalization())
        nldas_dataset_norm = rescaler.transform(
            data=dataset,
            data_interval=[0, n_samples],
            batch_size=batch_size,
            dump_path=os.path.join(data_path, dataset_name + "_normalization.h5"),
        )
        dataset = nldas_dataset_norm
    else:
        dataset = dataset

    variables_names = list(dataset.dtype.names)
    projected = pipeline.project_data(
        data=dataset,
        data_interval=[0, n_samples],
        variables_list=variables_names,
        batch_size=batch_size,
    )

    return projected


parser = ArgumentParser(description="Reading input arguments")
parser.add_argument("--data_path", type=str, help="The path tot the datasets.")
parser.add_argument("--grid_path", type=str, help="The path tot the datasets.")
parser.add_argument(
    "--n_components",
    type=int,
    help="Number of components to be used in the POD decomposition.",
)
parser.add_argument(
    "--memory_usage", type=float, help="The maximum memory usage to be employed"
)
parser.add_argument("--norm_usage", type=str, help="Use normalization ?", default=False)
parser.add_argument("--sufix", type=str, help="Sufix for the filename", default="")

#################
# PRE=PROCESSING
#################

#  Reading input arguments
args = parser.parse_args()

data_path = args.data_path
grid_path = args.grid_path
memory_usage = args.memory_usage
n_components = args.n_components
sufix = args.sufix
norm_usage = args.norm_usage
# Limiting the memory usage in this process to 30 %
memory_limiter = MemorySizeEval(memory_tol_percent=memory_usage)


# Training anf testing fraction of the dataset
train_frac = 0.6
test_frac = 0.4

# Datasets names (default choice)
nldas_field_dataset_name = "nldas"
nldas_special_field_dataset_name = "nldas_special"
nldas_forcings_dataset_name = "nldas_forcings"
nldas_secondary_forcings_dataset_name = "nldas_secondary_forcings"

# The NLDAS files can be large too much to fit in memory, so they will be manipulated
# on disk using HDF5
nldas_dataset = h5py.File(
    os.path.join(data_path, nldas_field_dataset_name + sufix + ".h5"), "r"
).get(nldas_field_dataset_name + sufix)

nldas_special_dataset = h5py.File(
    os.path.join(data_path, nldas_special_field_dataset_name + sufix + ".h5"), "r"
).get(nldas_special_field_dataset_name + sufix)

nldas_forcings_dataset = h5py.File(
    os.path.join(data_path, nldas_forcings_dataset_name + sufix + ".h5"), "r"
).get(nldas_forcings_dataset_name + sufix)

nldas_secondary_forcings_dataset = h5py.File(
    os.path.join(data_path, nldas_secondary_forcings_dataset_name + sufix + ".h5"), "r"
).get(nldas_secondary_forcings_dataset_name + sufix)

n_samples = nldas_dataset.shape[0]

train_samples = int(train_frac * n_samples)
test_samples = n_samples - train_samples

# Dumping the expect output to a file for further comparisons
nldas_exact_test_data = BatchCopy().copy(
    data=nldas_dataset,
    data_interval=[train_samples, train_samples + test_samples],
    batch_size=memory_limiter,
    dump_path=os.path.join(data_path, "nldas_exact_test_data.h5"),
)

# Projecting into the modes basis in order to obtain the reduced space
nldas_series_, nldas_rec, pipeline_nldas = construct_and_test_pipeline(
    nldas_dataset,
    fraction=train_frac,
    n_components=n_components,
    batch_size=memory_limiter,
    dataset_name="nldas",
)

nldas_s_series_, nldas_sl_rec, pipeline_nldas_s = construct_and_test_pipeline(
    nldas_special_dataset,
    fraction=train_frac,
    n_components=n_components,
    batch_size=memory_limiter,
    dataset_name="nldas_special",
)

nldas_f_series_, nldas_f_rec, pipeline_nldas_f = construct_and_test_pipeline(
    nldas_forcings_dataset,
    fraction=train_frac,
    n_components=n_components,
    batch_size=memory_limiter,
    dataset_name="nldas_forcings",
)

nldas_sf_series_, nldas_sf_rec, pipeline_nldas_sf = construct_and_test_pipeline(
    nldas_secondary_forcings_dataset,
    fraction=train_frac,
    n_components=n_components,
    batch_size=memory_limiter,
    dataset_name="nldas_secondary" "_forcings",
)

# Projecting into the modes basis in order to obtain the reduced space
nldas_series = prepare_time_series(
    pipeline=pipeline_nldas,
    dataset=nldas_dataset,
    batch_size=memory_limiter,
    dataset_name="nldas",
)

nldas_s_series = prepare_time_series(
    pipeline=pipeline_nldas_s,
    dataset=nldas_special_dataset,
    batch_size=memory_limiter,
    dataset_name="nldas_special",
)

nldas_f_series = prepare_time_series(
    pipeline=pipeline_nldas_f,
    dataset=nldas_forcings_dataset,
    batch_size=memory_limiter,
    dataset_name="nldas_forcings",
)

nldas_sf_series = prepare_time_series(
    pipeline=pipeline_nldas_sf,
    dataset=nldas_secondary_forcings_dataset,
    batch_size=memory_limiter,
    dataset_name="nldas_secondary_forcings",
)

variables_names = nldas_dataset.dtype.names

# Separating training and testing datasets and applying normalization
# Preparing NLDAS field data
nldas_norm = UnitarySymmetricalNormalization()
nldas_train_data = nldas_series[:train_samples, :]
nldas_test_data = nldas_series[train_samples:, :]

if norm_usage:
    nldas_train_data = nldas_norm.transform(data=nldas_train_data, eval=True, axis=1)
    nldas_test_data = nldas_norm.transform(data=nldas_test_data, eval=True, axis=1)
else:
    pass

# Preparing NLDAS special field data
nldas_special_norm = UnitarySymmetricalNormalization()
nldas_special_train_data = nldas_s_series[:train_samples, :]
nldas_special_test_data = nldas_s_series[train_samples:, :]

if norm_usage:
    nldas_special_train_data = nldas_special_norm.transform(
        data=nldas_special_train_data, eval=True, axis=1
    )
    nldas_special_test_data = nldas_special_norm.transform(
        data=nldas_special_test_data, eval=True, axis=1
    )
else:
    pass

# Preparing NLDAS forcing field data
nldas_forcings_norm = UnitarySymmetricalNormalization()
nldas_forcings_train_data = nldas_f_series[:train_samples, :]
nldas_forcings_test_data = nldas_f_series[train_samples:, :]

if norm_usage:
    nldas_forcings_train_data = nldas_forcings_norm.transform(
        data=nldas_forcings_train_data, eval=True, axis=1
    )
    nldas_forcings_test_data = nldas_forcings_norm.transform(
        data=nldas_forcings_test_data, eval=True, axis=1
    )
else:
    pass

# Preparing NLDAS secondary forcing data
nldas_secondary_forcings_norm = UnitarySymmetricalNormalization()
nldas_secondary_forcings_train_data = nldas_sf_series[:train_samples, :]
nldas_secondary_forcings_test_data = nldas_sf_series[train_samples:, :]

if norm_usage:
    nldas_secondary_forcings_train_data = nldas_special_norm.transform(
        data=nldas_secondary_forcings_train_data, eval=True, axis=1
    )
    nldas_secondary_forcings_test_data = nldas_special_norm.transform(
        data=nldas_secondary_forcings_test_data, eval=True, axis=1
    )
else:
    pass


# Train data
# The model input is [nldas, nldas_special, nldas_forcings, nldas_secondary_forcings]
input_train_data = np.hstack(
    [
        nldas_train_data,
        nldas_special_train_data,
        nldas_forcings_train_data,
        nldas_secondary_forcings_train_data,
    ]
)
# The model output must be just nldas
target_train_data = nldas_train_data

# Test data
input_test_data = np.hstack(
    [
        nldas_test_data,
        nldas_special_test_data,
        nldas_forcings_test_data,
        nldas_secondary_forcings_test_data,
    ]
)

# The auxiliary data complements the input space in order to make predicitons
initial_auxiliary_data = np.hstack(
    [
        nldas_special_train_data,
        nldas_forcings_train_data,
        nldas_secondary_forcings_train_data,
    ]
)[-1:]
auxiliary_data = np.hstack(
    [
        nldas_special_test_data,
        nldas_forcings_test_data,
        nldas_secondary_forcings_test_data,
    ]
)

target_test_data = nldas_test_data

############
# MODELLING
############

# The modelling stage receives input_train_data, target_train_data, input_test_data,
# target_test_data and auxiliary_data

# Hyper-parameters
group_size = 10
stencil_size = 12
skip_size = group_size
reservoir_dim = 2000
sparsity_level = 1
radius = 0.99
beta = 1e-5
sigma = 0.1
transformation = "T1"

# control parameters
sub_horizon = 1

# Configuration of ModelPool
pool_config = {
    "group_size": group_size,
    "stencil_size": stencil_size,
    "skip_size": skip_size,
}

# Configuration of the sub-networks (in this case, ESN-RC)
rc_config = {
    "reservoir_dim": reservoir_dim,
    "sparsity_level": sparsity_level,
    "radius": radius,
    "sigma": sigma,
    "beta": beta,
    "transformation": transformation,
}


initial_state = input_train_data[-1:, :]
horizon = input_test_data.shape[0]

# ESN-RC receives the prior current state and forecasts the next one
input_data = input_train_data[:-1, ...]
target_data = target_train_data[1:, ...]
auxiliary_data = np.vstack([initial_auxiliary_data, auxiliary_data[:-1, ...]])

# Instantiating the pool of sub-networks in order to execute
# the parallel training
pool = ModelPool(
    config=pool_config, model_type="EchoStateNetwork", model_config=rc_config
)

pool.fit(input_data=input_data, target_data=target_data)

# Dynamic extrapolation using the trained model
extrapolation_data = pool.predict(
    initial_state=initial_state, horizon=horizon, auxiliary_data=auxiliary_data
)
# Inverse normalization transform
if norm_usage:
    nldas_extrapolation_series = nldas_norm.transform_inv(data=extrapolation_data)
else:
    nldas_extrapolation_series = extrapolation_data

initial_state = input_train_data[-1:, :]

n_steps = nldas_test_data.shape[0]
one_shot_extrapolation_list = list()


# One-step ahead extrapolation
for step in range(0, n_steps, sub_horizon):
    current_data = pool.predict(initial_state=initial_state, horizon=sub_horizon)

    initial_state = np.hstack(
        [nldas_test_data[step, :][None, ...], auxiliary_data[step][None, ...]]
    )

    one_shot_extrapolation_list.append(current_data)

one_shot_extrapolation = np.vstack(one_shot_extrapolation_list)

nldas_one_shot_extrapolation_series = one_shot_extrapolation

# The output of the modelling stage are nldas_extrapolation_series
# and nldas_one_shot_extrapolation_series

##################
# POST-PROCESSING
##################

# Visualizing the latent results
for cc in range(n_components):
    plt.plot(nldas_test_data[:, cc], label="exact")
    plt.plot(one_shot_extrapolation[:, cc], label="one-shot")
    plt.plot(nldas_extrapolation_series[:, cc], label="dynamic")

    plt.title("Latent space, series {}".format(cc))
    plt.grid(True)
    plt.xlabel("Time-step")
    plt.legend()
    plt.savefig(os.path.join(data_path, "_{}_series-{}_step.png".format(cc, step)))
    plt.close()

# Error evaluation
l2_norm = L2Norm()
error_dynamic = l2_norm(
    data=nldas_extrapolation_series, reference_data=nldas_test_data, relative_norm=True
)

error_one_shot = l2_norm(
    data=nldas_one_shot_extrapolation_series,
    reference_data=nldas_test_data,
    relative_norm=True,
)

print("Dynamic extrapolation relative error: {}\n".format(100 * error_dynamic))
print("One-shot ahead extrapolation relative error: {}\n".format(100 * error_one_shot))

# Reconstructing the time-series to the original space before the ROM
# application
nldas_extrapolation = pipeline_nldas.reconstruct_data(
    data=nldas_extrapolation_series,
    data_interval=[0, horizon],
    variables_list=variables_names,
    batch_size=memory_limiter,
    dump_path=os.path.join(data_path, "nldas_extrapolation.h5"),
)
# Reconstructing the time-series to the original space before the ROM
# application
nldas_one_shot_extrapolation = pipeline_nldas.reconstruct_data(
    data=nldas_one_shot_extrapolation_series,
    data_interval=[0, horizon],
    variables_list=variables_names,
    batch_size=memory_limiter,
    dump_path=os.path.join(data_path, "nldas_one_shot_" "extrapolation.h5"),
)
# Visualizing the results
lat_long_grid = np.load(grid_path)
Lat = lat_long_grid["Lat"]
Long = lat_long_grid["Long"]

for step in range(0, horizon, 10):
    for var in variables_names:
        estimated_var_state = nldas_extrapolation[var][step, 0, ...]
        exact_var_state = nldas_dataset[var][train_samples + step, 0, ...]
        one_shot_estimated_var_state = nldas_one_shot_extrapolation[var][step, 0, ...]

        exact_var_state = np.where(exact_var_state > 1e15, np.NaN, exact_var_state)

        exact_var_state_norm = exact_var_state[np.isnan(exact_var_state) == False]
        norm = Normalize(
            vmin=exact_var_state_norm.min(), vmax=exact_var_state_norm.max()
        )

        plt.title("{} dynamically estimated, time step {}".format(var, step))
        plt.pcolormesh(Long, Lat, estimated_var_state, norm=norm)
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()
        plt.axis("scaled")

        plt.savefig(
            os.path.join(data_path, "{}_step-{}_approximated.png".format(var, step))
        )
        plt.close()

        plt.title("{} one-step-ahead estimated, time step {}".format(var, step))
        plt.pcolormesh(Long, Lat, one_shot_estimated_var_state, norm=norm)
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()
        plt.axis("scaled")

        plt.savefig(
            os.path.join(
                data_path, "{}_step-{}_one_step_approximated.png".format(var, step)
            )
        )
        plt.close()

        plt.title("{} exact, time step {}".format(var, step))
        plt.pcolormesh(Long, Lat, exact_var_state, norm=norm)
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()
        plt.axis("scaled")

        plt.savefig(os.path.join(data_path, "{}_step-{}_exact.png".format(var, step)))
        plt.close()

print("Process concluded.")
