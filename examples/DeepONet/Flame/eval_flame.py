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

initial_state_test = np.array([1e-3])
n_outputs = 1
Delta_t = 0.05
n_times = int(2/(initial_state_test[0]*Delta_t))
Q = 1000

# Testing to reload from disk
saver = SPFile(compact=False)
flame_net = saver.read(model_path=save_path)

branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))
trunk_input_test = np.linspace(0, Delta_t, Q)[:, None]

eval_list = list()

for i in range(0, 2):
    branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))

    approximated_data = flame_net.eval(
        trunk_data=trunk_input_test, branch_data=branch_input_test
    )
    initial_state_test = approximated_data[-1]

    eval_list.append(approximated_data)

evaluation = np.vstack(eval_list)
time = np.linspace(0, n_times, evaluation.shape[0])

np.save("evaluation.npy", evaluation)
plt.plot(time, evaluation, label="Approximated")
plt.xlabel("t (s)")
plt.savefig("flame_approximation.png")
plt.close()

plt.figure(figsize=(15, 6))

for i in range(n_outputs):
    plt.plot(time, evaluation[:, i], label=f"s_{i+1}")
    plt.xlabel("t (s)")

plt.yticks(np.linspace(0, 1, 5))
plt.legend()
plt.grid(True)
plt.savefig(f"flame_approximation_custom.png")
