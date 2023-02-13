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
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from simulai.math.differentiation import CollocationDerivative
from simulai.math.integration import LSODA, ClassWrapper
from simulai.metrics import L2Norm
from simulai.regression import OpInf

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--data_path", type=str, help="Save path", default="/tmp")

args = parser.parse_args()
data_path = args.data_path
train_fraction = 0.80
validation_fraction = (1 - train_fraction) / 2

data = np.load(data_path)

input_data = data["projected"]
closure_input_data = data["qqm_projected"]

dt = 1 / input_data.shape[0]
t = np.arange(0.0, 1, dt)

diff = CollocationDerivative(config={})
output_data = diff.solve(data=input_data, x=t)

n_samples = input_data.shape[0]
train_samples = int(train_fraction * n_samples)
validation_samples = int(validation_fraction * n_samples)
test_samples = n_samples - train_samples - validation_samples

input_train_data = input_data[:train_samples]
closure_train_data = closure_input_data[:train_samples]
extended_input_train_data = np.hstack([input_train_data, closure_train_data])
input_validation_data = input_data[train_samples : train_samples + validation_samples]

output_train_data = output_data[:train_samples]
output_validation_data = output_train_data[
    train_samples : train_samples + validation_samples
]
t_validation = t[train_samples : train_samples + validation_samples]

input_test_data = input_data[train_samples + validation_samples :]
output_test_data = output_data[train_samples + validation_samples :]
t_test = t[train_samples + validation_samples :]

# OpInf + QQM
q_interval = np.arange(-3, 3, 0.5)
l_interval = np.arange(-3, 3, 0.5)
regs = list(product(l_interval, q_interval))

models = dict()
errors = dict()

for reg in regs:
    opinf_op = OpInf(bias_rescale=1e-15)

    lambda_linear = 10.0 ** reg[0]
    lambda_quadratic = 10.0 ** reg[-1]

    opinf_op.set(lambda_linear=lambda_linear, quadratic_linear=lambda_quadratic)
    opinf_op.fit(input_data=input_train_data, target_data=output_train_data)

    right_operator = ClassWrapper(opinf_op)

    solver = LSODA(right_operator)

    init_state = input_train_data[-1:]
    initial_state = init_state[0]

    estimated_field = solver.run(initial_state, t_validation)

    l2_norm = L2Norm()
    error = 100 * l2_norm(
        data=estimated_field, reference_data=input_validation_data, relative_norm=True
    )

    print(f"regularization: {reg}, error: {error} %")

    models[f"opinf_{reg}"] = opinf_op
    errors[f"opinf_{reg}"] = error

min_value = min(errors.values())
best_model = [models[key] for key in models.keys() if errors[key] == min_value][0]

right_operator = ClassWrapper(best_model)

solver = LSODA(right_operator)

init_state = input_validation_data[-1:]
initial_state = init_state[0]

estimated_field = solver.run(initial_state, t_test)

l2_norm = L2Norm()
error = 100 * l2_norm(
    data=estimated_field, reference_data=input_test_data, relative_norm=True
)

for i in range(0, estimated_field.shape[1]):
    plt.plot(estimated_field[:, i])
    plt.plot(input_test_data[:, i])
    plt.show()

print(f"error: {error} %")
