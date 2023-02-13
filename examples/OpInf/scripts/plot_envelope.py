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

# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--data_path", type=str, help="Data path")
parser.add_argument("--F", type=float, help="Forcing value")
parser.add_argument("--save_path", type=str, help="Save path", default="/tmp")

plt.rcParams.update(
    {
        "axes.linewidth": 2,
        "axes.edgecolor": "black",
        "legend.edgecolor": "black",
        "legend.shadow": False,
        "legend.fancybox": True,
        "legend.frameon": True,
    }
)

args = parser.parse_args()

data_path = args.data_path
save_path = args.save_path
F = args.F

tol = 0.5
dt = 0.01

if F == 8:
    x_lim = 21
    lambda_1 = 1 / 1.68
else:
    x_lim = 21
    lambda_1 = 1 / 2.27


def VPT(nrmse, t_test):
    time_ref = (t_test - t_test[0]) / lambda_1
    t_ref = time_ref[nrmse <= tol]
    VPT = t_ref[-1]

    return VPT


NMRSE = np.load(data_path)
t = np.arange(0.0, 2000.0, dt)
n_steps = t[t >= 1000].shape[0]
nt = int(0.5 * n_steps)
nt_test = n_steps - nt
t_test = t[t >= 1000][nt:]

n_simulations = NMRSE.shape[-1]
simulations_dict = dict()

for index in range(n_simulations):
    nmrse = NMRSE[:, index]

    vpt = VPT(nmrse, t_test)

    simulations_dict[index] = vpt

key_max = max(simulations_dict, key=lambda x: simulations_dict[x])
key_min = min(simulations_dict, key=lambda x: simulations_dict[x])

curve_max = NMRSE[:, key_max]
curve_min = NMRSE[:, key_min]

time_ref = (t_test - t_test[0]) / lambda_1

mean_curve = NMRSE.mean(axis=1)
std_value = NMRSE.std(axis=1)

plt.fill_between(
    time_ref,
    y1=mean_curve + std_value,
    y2=mean_curve - std_value,
    interpolate=True,
    color="turquoise",
    alpha=0.7,
)
plt.plot(time_ref, mean_curve, label="Mean")
plt.legend(loc=2, fontsize=15)
plt.xlim(0, x_lim)
plt.ylim(-0.15, 1.75)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(loc=2, fontsize=14)
plt.xticks(np.linspace(0, x_lim, int(x_lim / 3) + 1))
plt.axhline(y=0.5, ls="--", lw=1.25, c="black")
plt.ylabel(r"$NRMSE$", fontsize=16)
plt.xlabel(r"$t/T^{\Lambda_1}$", fontsize=16)
plt.grid(True)
plt.tight_layout()

if F == 8.0:
    plt.errorbar(
        time_ref[1000],
        mean_curve[1000],
        std_value[1000],
        linestyle="",
        fmt="o",
        color="black",
        lw=1.0,
    )
    plt.text(15.5, 1.28, r"$\sigma$", fontsize=14)
else:
    plt.errorbar(
        time_ref[750],
        mean_curve[750],
        std_value[750],
        linestyle="",
        fmt="o",
        color="black",
        lw=1.0,
    )
    plt.text(15, 1.40, r"$\sigma$", fontsize=14)

plt.savefig(os.path.join(save_path, f"nmrse_along_time_{F}_envelope.png"))

plt.show()
