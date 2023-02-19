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

#!/usr/bin/env python
import warnings

with warnings.catch_warnings():
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from scipy.integrate import odeint

    from simulai.math.differentiation import CollocationDerivative
    from simulai.math.integration import LSODA, ClassWrapper
    from simulai.regression import OpInf

# Special customization for matplotlib
matplotlib_custom = False  # True

if matplotlib_custom == True:
    plt.rcParams.update(
        {
            #'font.size': 16,
            "axes.linewidth": 2,
            #'axes.titlesize': 20,
            "axes.edgecolor": "black",
            #'axes.labelsize': 20,
            "axes.grid": True,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "figure.figsize": (15, 6),
            #'xtick.labelsize': 14,
            #'ytick.labelsize': 14,
            "font.family": "Arial",
            #'legend.fontsize': 13,
            "legend.framealpha": 1,
            "legend.edgecolor": "black",
            "legend.shadow": False,
            "legend.fancybox": True,
            "legend.frameon": True,
        }
    )
else:
    pass


def NRMSE(exact, approximated):
    return np.sqrt(
        np.mean(np.square(exact - approximated) / exact.std(axis=0) ** 2, axis=1)
    )


def NRSE(exact, approximated):
    return np.sqrt(np.square(exact - approximated) / exact.std(axis=0) ** 2)


# Reading command line arguments.
parser = ArgumentParser(description="Reading input parameters")

parser.add_argument("--save_path", type=str, help="Save path", default="/tmp")
parser.add_argument("--F", type=float, help="Forcing value", default=8)

args = parser.parse_args()

save_path = args.save_path
F = args.F

initial_states_file = os.path.join(save_path, "initial_random.npy")

tol = 0.5

# These are our constants
N = 40  # Number of variables

n_initial = 100

if F == 8:
    lambda_1 = 1 / 1.68
else:
    lambda_1 = 1 / 2.27


def Lorenz96(x, t):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


x0 = F * np.ones(N)  # Initial state (equilibrium)

if os.path.isfile(initial_states_file):
    initial_states = np.load(initial_states_file)
    print("No new initial_states file is being created.")
else:
    initial_states_list = list()
    for nn in range(n_initial):
        x0_nn = x0 + 0.01 * np.random.rand(
            N
        )  # Add small perturbation to the first variable
        initial_states_list.append(x0_nn[None, :])
    initial_states = np.vstack(initial_states_list)

    np.save(initial_states_file, initial_states)

vpt_list = dict()
nrmse_list = list()

for si in range(n_initial):
    print(f"Running case: {si}/{n_initial}")

    x0 = initial_states[si]

    dt = 0.01
    t = np.arange(0.0, 2000.0, dt)
    lorenz_data = odeint(Lorenz96, x0, t)[t >= 1000]

    diff = CollocationDerivative(config={})
    derivative_lorenz_data = diff.solve(data=lorenz_data, x=t[t >= 1000])

    n_steps = t[t >= 1000].shape[0]
    nt = int(0.5 * n_steps)
    nt_test = n_steps - nt
    t_test = t[t >= 1000][nt:]
    n_field = N

    train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
    train_field_derivative = derivative_lorenz_data[:nt]

    test_field = lorenz_data[nt:]  # manufactured nonlinear oscillator data
    test_field_derivatives = derivative_lorenz_data[nt:]

    label = f"n_{N}_F_{F}_init_{si}"

    lorenz_op = OpInf(bias_rescale=1, solver="pinv")
    lorenz_op.fit(input_data=train_field, target_data=train_field_derivative)

    init_state = train_field[-1:]
    estimated_field_derivatives = lorenz_op.eval(input_data=test_field)
    tags = [rf"x_{i}" for i in range(n_field)]

    # Construcing jacobian tensor (It could be used during the time-integrations, but seemingly it is not).
    lorenz_op.construct_K_op()

    # Time-integrating the trained model and visualizing the output.

    # Using the derivatives surrogate for time-integrating
    right_operator = ClassWrapper(lorenz_op)

    solver = LSODA(right_operator)

    initial_state = init_state[0]

    estimated_field = solver.run(initial_state, t_test)

    # Estimating the number of Lyapunov units for the extrapolation.
    nrmse = NRMSE(test_field, estimated_field)
    nrmse_ = nrmse[nrmse <= tol]

    time_ref = (t_test - t_test[0]) / lambda_1
    t_ref = time_ref[nrmse <= tol]
    VPT = t_ref[-1]

    print(f"VPT is {VPT}")

    vpt_list[si] = VPT

    plt.plot(t_ref, nrmse_)
    plt.ylabel(r"$NRMSE$")
    plt.xlabel(r"$t/T^{\Lambda_1}$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"nmrse_along_time_{label}.png"))
    plt.close()

    nrmse_list.append(nrmse[:, None])

    nrse = NRSE(test_field, estimated_field)

    vars = [0, 10, 20, 30, 39]

    for vv in vars:
        plt.figure(0, figsize=(8, 4.3))
        plt.plot(time_ref, estimated_field[:, vv], label="Predicted", color="blue")
        plt.plot(time_ref, test_field[:, vv], label="Ground Truth", color="red")
        plt.title(rf" Variable $X_{str({vv})}$", fontsize=20)
        plt.xlabel(r"$t/T^{\Lambda_1}$", fontsize=16)
        plt.xlim(0, 10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"evaluation_{vv}_{label}.png"))
        plt.close()

    t_plot = time_ref
    x_plot = np.arange(0, N)
    aspect = 15
    extent = (x_plot.min(), x_plot.max(), t_plot.min(), t_plot.max())

    color_norm = Normalize(
        vmin=min(test_field.min(), estimated_field.min()),
        vmax=max(test_field.max(), estimated_field.max()),
    )

    plt.figure(0, figsize=(2.8, 4.8))
    plt.imshow(
        np.flip(estimated_field, axis=0),
        extent=extent,
        aspect=aspect,
        cmap="seismic",
        interpolation="bilinear",
        norm=color_norm,
    )
    plt.colorbar()
    plt.title("Predicted")
    plt.xlabel("Observable")
    plt.ylabel(r"$t/T^{\Lambda_1}$")
    plt.xticks([0, N - 1])
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"approximated_2D_plot_{label}.png"))
    plt.close()

    plt.figure(1, figsize=(2.8, 4.8))
    plt.imshow(
        np.flip(test_field, axis=0),
        extent=extent,
        aspect=aspect,
        cmap="seismic",
        interpolation="bilinear",
        norm=color_norm,
    )
    plt.colorbar()
    plt.title("Ground Truth")
    plt.xlabel("Observable")
    plt.ylabel(r"$t/T^{\Lambda_1}$")
    plt.xticks([0, N - 1])
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"exact_2D_plot_{label}.png"))
    plt.close()

    plt.figure(2, figsize=(2.8, 4.8))
    plt.imshow(
        np.flip(nrse, axis=0),
        extent=extent,
        aspect=aspect,
        interpolation="bilinear",
        cmap="Oranges",
    )

    plt.colorbar()
    plt.title("NRSE")
    plt.xlabel("Observable")
    plt.ylabel(r"$t/T^{\Lambda_1}$")
    plt.xticks([0, N - 1])
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"nrse_2D_plot_{label}.png"))
    plt.close()

    # Concatenated plots
    color_norm = Normalize(vmin=test_field.min(), vmax=test_field.max())

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    ax1.imshow(
        np.flip(estimated_field, axis=0),
        extent=extent,
        aspect=aspect,
        cmap="seismic",
        interpolation="bilinear",
        norm=color_norm,
    )

    ax1.set_title("Predicted", fontsize=10.5)
    ax1.set_xlabel("Observable")
    ax1.set_ylabel(r"$t/T^{\Lambda_1}$")
    ax1.set_xticks([0, 39])
    ax1.set(ylim=(0, 10))

    values2 = ax2.imshow(
        np.flip(test_field, axis=0),
        extent=extent,
        aspect=aspect,
        cmap="seismic",
        interpolation="bilinear",
        norm=color_norm,
    )

    ax2.set_title("Ground Truth", fontsize=10.5)
    ax2.set_xlabel("Observable")
    ax2.set_xticks([0, 39])
    ax2.set_yticks([])
    ax2.set(ylim=(0, 10))

    cbar1 = fig.colorbar(values2, ax=ax2)
    fig.subplots_adjust(wspace=-0.6, bottom=0.1, left=0.2, right=0.7)

    for t in cbar1.ax.get_yticklabels():
        t.set_fontsize(9.5)

    plt.savefig(
        os.path.join(save_path, f"comparison_exact_predicted_2D_plot_{label}.png")
    )
    plt.close()

    fig1, ax3 = plt.subplots()
    values3 = ax3.imshow(
        np.flip(nrse, axis=0),
        extent=extent,
        aspect=aspect,
        interpolation="bilinear",
        cmap="Oranges",
    )
    ax3.set_title("NRSE", fontsize=10.5)
    ax3.set_xlabel("Observable")
    ax3.set_xticks([0, 39])
    ax3.set_yticks([])
    ax3.set(ylim=(0, 10))

    cbar2 = fig.colorbar(values3, ax=ax3)
    fig1.subplots_adjust(wspace=0.1, right=0.5, hspace=0)

    for t in cbar2.ax.get_yticklabels():
        t.set_fontsize(9.5)

    plt.savefig(os.path.join(save_path, f"no_number_nrse_2D_plot_{label}.png"))
    plt.close()

values = list(vpt_list.values())

mean_value = np.mean(values)
max_value = np.max(values)
std_value = np.std(values)
min_value = np.min(values)

nrmse_array = np.hstack(nrmse_list)

print(f"Mean VPT for F={F}: {mean_value}")
print(f"Minimum VPT for F={F}: {min_value}")
print(f"Maximum VPT for F={F}: {max_value}")
print(f"Standard deviation VPT for F={F}: {std_value}")

key_max = max(vpt_list, key=lambda x: vpt_list[x])

print(f"The index of the best result is: {key_max}")

lines = [
    f"Mean VPT for F={F}: {mean_value}\n",
    f"Minimum VPT for F={F}: {min_value}\n",
    f"Maximum VPT for F={F}: {max_value}\n",
    f"Standard deviation VPT for F={F}: {std_value}\n",
    f"The index of the best result is: {key_max}\n",
]

with open(os.path.join(save_path, f"output_F_{F}.log"), "w+") as fp:
    fp.writelines(lines)

np.save(os.path.join(save_path, f"nrmse_F_{F}.npy"), nrmse_array)
