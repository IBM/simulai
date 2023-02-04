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

from simulai.file import SPFile

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--save_path", type=str, help="Save path", default="/tmp")
args = parser.parse_args()

save_path = args.save_path

initial_state_test = np.array([1, 0, 0])
n_outputs = 3
n_times = 500
Q = 2

# Testing to reload from disk
saver = SPFile(compact=False)
rober_net = saver.read(model_path=save_path)

branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))
trunk_input_test = np.linspace(0, 1, Q)[:, None]

eval_list = list()

for i in range(0, n_times):
    branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))

    approximated_data = rober_net.eval(
        trunk_data=trunk_input_test, branch_data=branch_input_test
    )
    initial_state_test = approximated_data[-1]

    eval_list.append(approximated_data)

evaluation = np.vstack(eval_list) * np.array([1, 1e4, 1])
time = np.linspace(0, n_times, evaluation.shape[0])

np.save("evaluation.npy", evaluation)
plt.plot(time, evaluation, label="Approximated")
plt.xlabel("t (s)")
plt.savefig("rober_approximation.png")

for i in range(n_outputs):
    plt.plot(time, evaluation[:, i])
    plt.xlabel("t (s)")
    plt.savefig(f"rober_approximation_{i}.png")
    plt.close()
