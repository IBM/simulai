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
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import convolve

os.environ["engine"] = "pytorch"

from simulai.optimization import Optimizer
from simulai.regression import ConvolutionalNetwork


# Applying a simple nonlinear transformation
def apply_transformation(data, k, cval=0):
    y_ = convolve(data, k, cval=cval)

    return np.where(y_ > y_.mean(), y_, y_**2)


# MinMax normalization
def normalize(data_train, data_test):
    maximum = data_train.max()
    minimum = data_train.min()

    data_train_ = (data_train - minimum) / (maximum - minimum)
    data_test_ = (data_test - minimum) / (maximum - minimum)

    return data_train_, data_test_


# Loading and pre-processing data
path = "/tmp/mnist.npz"
path_new = "/tmp/mnist_new.npz"

# Loading the MNIST dataset
mnist = np.load(path)

x_train = mnist["x_train"]
y_train = mnist["y_train"]

x_test = mnist["x_test"]
y_test = mnist["y_test"]

n_samples_train = x_train.shape[0]
n_samples_test = x_test.shape[0]
dims = x_train.shape[1:]
dims_ = (56, 56)
grid_x, grid_y = np.mgrid[0:1:28j, 0:1:28j]
grid_x_new, grid_y_new = np.mgrid[0:1:56j, 0:1:56j]

points = np.hstack([grid_x.flatten()[:, None], grid_y.flatten()[:, None]])
values_train = x_train.reshape(-1, np.product(dims[:]))
values_test = x_test.reshape(-1, np.product(dims[:]))

if not os.path.isfile(path_new):
    interpolated_train_list = list()
    interpolated_test_list = list()

    for ii in range(n_samples_train):
        sys.stdout.write(f"\r{ii}/{n_samples_train}")
        sys.stdout.flush()
        interpolated = (
            griddata(
                points, values_train[ii], (grid_x_new, grid_y_new), method="linear"
            )
            .reshape(-1, *dims_)
            .astype("uint8")
        )
        interpolated_train_list.append(interpolated)

    for ii in range(n_samples_test):
        sys.stdout.write(f"\r{ii}/{n_samples_test}")
        sys.stdout.flush()
        interpolated = (
            griddata(points, values_test[ii], (grid_x_new, grid_y_new), method="linear")
            .reshape(-1, *dims_)
            .astype("uint8")
        )

        interpolated_test_list.append(interpolated)

    x_train_hr = np.vstack(interpolated_train_list)
    x_test_hr = np.vstack(interpolated_test_list)

    np.savez(path_new, x_train_hr=x_train_hr, x_test_hr=x_test_hr)

else:
    dataset_new = np.load(path_new)

    x_train_hr = dataset_new["x_train_hr"]
    x_test_hr = dataset_new["x_test_hr"]

x_train_norm, x_test_norm = normalize(x_train, x_test)
x_train_norm_hr, x_test_norm_hr = normalize(x_train_hr, x_test_hr)

# Kernel to be applied to the original data convolve transformation
divergence_kernel = np.array([[0, -1 / 2, 0], [-1 / 2, 0, 1 / 2], [0, 1 / 2, 0]])[
    None, ...
]

# Preparing datasets
x_train_tra = apply_transformation(x_train_norm_hr, divergence_kernel)[:, None, ...]
x_test_tra = apply_transformation(x_test_norm_hr, divergence_kernel)[:, None, ...]

# Configuring model
n_inputs = 1
lr = 1e-3  # Initial learning rate for the ADAM algorithm
n_epochs = 10

optimizer_config = {"lr": lr}

layers = [
    {
        "in_channels": n_inputs,
        "out_channels": 4,
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": (1, 1),
        "after_conv": {"type": "upsample", "scale_factor": 2},
    },
    {
        "in_channels": 4,
        "out_channels": 2,
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": (1, 1),
    },
    {
        "in_channels": 2,
        "out_channels": 1,
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": (1, 1),
    },
]

# Instantiating network
convnet = ConvolutionalNetwork(layers=layers, activations="sigmoid")

# Instnatiating optimizer
params = {"lambda_1": 0.0, "lambda_2": 0.0}
optimizer = Optimizer("adam", params=optimizer_config)

### Training
optimizer.fit(
    op=convnet,
    input_data=x_train_norm,
    target_data=x_train_tra,
    n_epochs=n_epochs,
    loss="rmse",
    params=params,
    batch_size=1000,
    device="gpu",
)

### Evaluating
x_test_eval = convnet.eval(input_data=x_test_norm)

### Visualizing
test_index = 10

# Input
plt.imshow(x_test_norm[test_index, ...])
plt.colorbar()
plt.show()

# Target
plt.imshow(x_test_eval[test_index, 0, ...])
plt.colorbar()
plt.show()

# Output
plt.imshow(x_test_tra[test_index, 0, ...])
plt.colorbar()
plt.show()
