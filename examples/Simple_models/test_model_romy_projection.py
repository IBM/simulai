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

from simulai.io import Reshaper
from simulai.metrics import L2Norm
from simulai.rom import POD
from simulai.simulation import Pipeline

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
# Loading simulation data
data = np.load(data_path)
# Correcting the shape of the input data.
# It must be (n_batches, n_variables, **(other dimensions))
data = data.transpose(0, 3, 1, 2)

# Problem variables names
variables_list = list(data.dtype.names)

# Convert the simulation data to simple NumPy arrays (it is originally a structured array)
input_data_ = np.concatenate([data[var] for var in variables_list], 1)

n_batches = data.shape[0]
frac = 0.5

# Training data
input_data = data[: int(frac * n_batches), :, :, :]
test_data = data[int(frac * n_batches) :, :, :, :]

# Configurations
rom_config = {"n_components": 25}

l2_norm = L2Norm()

pipeline = Pipeline([("data_preparer", Reshaper()), ("rom", POD(config=rom_config))])

pipeline.exec(data=input_data, input_data=input_data)

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
