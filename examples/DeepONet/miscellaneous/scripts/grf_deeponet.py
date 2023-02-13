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

import os.path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from simulai.math.integration import RK4
from simulai.math.spaces import GaussianRandomFields
from simulai.metrics import L2Norm
from simulai.models import DeepONet
from simulai.regression import DenseNetwork


# Nonlinear ODE system used for generating test data
class NonlinearODE:
    def __init__(self):
        pass

    def __call__(self, data):
        s = data[:, 0]
        u = data[:, 1]

        return -(s**2) + u


# Some forcing terms used
def sinx_forcing(x):
    return np.sin(np.pi * x)


def sin2x_forcing(x):
    return np.sin(2 * np.pi * x)


def x_forcing(x):
    return x


def solver(x_interval=None, N=None, x=None, dx=None, u=x_forcing):
    if x is None and (N is not None and x_interval is not None):
        x = np.linspace(0, 1, N)
        dx = (x_interval[1] - x_interval[0]) / N

    elif isinstance(x, np.ndarray):
        assert dx, "dx must be provided."
        N = u.shape[0]

    else:
        raise Exception("Case not covered.")

    initial_state = np.array([0])[None, :]

    if callable(u):
        forcings = u(x)[:, None]
    elif isinstance(u, np.ndarray):
        assert u.shape[0] == x.shape[0]
        forcings = u
    else:
        raise Exception(f"It is expected a callable or np.ndarray, but received {u}")

    nonlinear_ODE = NonlinearODE()
    integrator = RK4(right_operator=nonlinear_ODE)
    output_array = integrator(
        initial_state=initial_state, epochs=N, dt=dx, forcings=forcings
    )

    return np.vstack([initial_state, output_array[:-1]]), forcings, x[:, None]


class TestDeepONet:
    def __init__(self):
        self.enable_plots = False
        self.grf_data_filename = "grf_data.npz"
        self.test_grf_data_filename = "test_grf_data.npz"
        self.estimated_grf_data_filename = "estimated_grf_data.npy"

    def generate_GRF_data(self, x_interval, N_tot, n_features):
        points = np.linspace(*x_interval, N_tot)  # Positions for sampling u data

        generator = GaussianRandomFields(
            x_interval=(0, 1), kernel="RBF", length_scale=0.2, N=N_tot, interp="cubic"
        )

        features = generator.random_u(n_features=n_features)

        u_exec = generator.generate_u(features, points)

        dx = (x_interval[1] - x_interval[0]) / N_tot

        outputs_list = list()
        for ff in range(n_features):
            s, _, _ = solver(N=N_tot, dx=dx, x=points, u=u_exec[:, ff][:, None])
            outputs_list.append(s)
            print(f"Executed test with the forcing {ff}")

        outputs_data = np.hstack(outputs_list)

        return outputs_data, u_exec, points[:, None]

    def test_deeponet_scalar_grf(
        self,
        N_tot=int(1e3),
        n_features=500,
        m=100,
        N_samples=20,
        N_epochs=10000,
        branch_width=100,
        trunk_width=100,
        path=None,
    ):
        x_interval = (0, 1)

        sensors_indices = np.arange(0, N_tot, int(N_tot / m))

        # Generating or restoring data
        data_file = os.path.join(path, self.grf_data_filename)

        if os.path.isfile(data_file):
            content = np.load(data_file)
            data = content["data"]
            u = content["u"]
            x = content["x"]
        else:
            data, u, x = self.generate_GRF_data(x_interval, N_tot, n_features)
            np.savez(data_file, data=data, u=u, x=x)

        maximums = np.max(np.abs(data), axis=0)
        nan_indices = np.where(maximums > 1.0)[0].tolist()
        valid_indices = list(set(np.arange(n_features)) - set(nan_indices))
        data = data[:, valid_indices]
        u = u[:, valid_indices]

        n_features = data.shape[1]
        n_features_train = int(0.9 * n_features)
        n_features_test = n_features - n_features_train

        timesteps_indices = np.random.choice(N_tot, N_samples, replace=False)

        if not 0 in timesteps_indices:
            timesteps_indices = np.hstack([timesteps_indices, 0])
            N_samples += 1

        # Training data
        u_train = u[:, :n_features_train]
        data_train = data[:, :n_features_train]
        u_sensors = u_train[sensors_indices].T
        U_sensors = np.tile(u_sensors, (N_samples, 1))

        x_samples = x[timesteps_indices]
        V = np.ravel(data_train[timesteps_indices])[:, None]
        X = np.tile(x_samples, (1, n_features_train)).flatten()[:, None]

        # Testing data
        u_test = u[:, n_features_train:]
        data_test = data[:, n_features_train:]
        x_test = x

        test_data_file = os.path.join(path, self.test_grf_data_filename)
        if not os.path.isfile(test_data_file):
            np.savez(test_data_file, data=data_test, u=u_test, x=x_test)

        p = 50

        trunk_architecture = 3 * [branch_width]

        trunk_setup = {
            "architecture": trunk_architecture,
            "dropouts_rates_list": [0, 0, 0],
            "activation_function": "relu",
            "input_dim": 1,
            "output_dim": p,
        }

        branches_architecture = 2 * [trunk_width]

        branches_setup = {
            "architecture": branches_architecture,
            "dropouts_rates_list": [0, 0],
            "activation_function": "relu",
            "input_dim": m,
            "output_dim": p,
        }

        trunk_net = DenseNetwork(
            architecture=trunk_architecture,
            config=trunk_setup,
            concat_output_tensor=True,
            concat_input_tensor=True,
        )

        branch_net = DenseNetwork(
            architecture=branches_architecture,
            config=branches_setup,
            concat_output_tensor=True,
            concat_input_tensor=True,
        )

        optimizers_config = {"Adam": {"maxiter": N_epochs}}

        # DeepONet wrapper
        operator = DeepONet(
            trunk_network=trunk_net,
            branch_network=branch_net,
            optimizers_config=optimizers_config,
            model_id="nonlinear_ode",
        )

        operator.fit(X, U_sensors, V, shuffle=False)

        V_evaluated_list = list()
        for feature in range(n_features_test):
            u_sensors = u_test[sensors_indices, feature : feature + 1].T
            U_sensors_test = np.tile(u_sensors, (x_test.shape[0], 1))

            x_samples = x_test
            V_test = data_test[:, feature : feature + 1]
            X_test = x_samples

            V_evaluated = operator.eval(trunk_data=X_test, branch_data=U_sensors_test)
            V_evaluated_list.append(V_evaluated)

            l2_norm = L2Norm()
            error = l2_norm(data=V_evaluated, reference_data=V_test, relative_norm=True)
            print(f"Evaluation error {100*error} %")

        estimated_data_file = os.path.join(path, self.estimated_grf_data_filename)

        V_evaluated = np.hstack(V_evaluated_list)
        np.save(estimated_data_file, V_evaluated)

        print("Concluded.")

    def plot(self, path=None):
        test_data_file = os.path.join(path, self.test_grf_data_filename)
        estimated_data_file = os.path.join(path, self.estimated_grf_data_filename)

        content = np.load(test_data_file)
        data_exact = content["data"]
        x = content["x"]

        data_estimated = np.load(estimated_data_file)

        n_features = data_estimated.shape[1]

        for feature in range(n_features):
            estimated = data_estimated[:, feature : feature + 1]
            exact = data_exact[:, feature : feature + 1]

            l2_norm = L2Norm()
            error = l2_norm(data=estimated, reference_data=exact, relative_norm=True)
            print(f"Evaluation error {100 * error} %")

            print(f"Plotting forcing {feature + 1}")

            plt.plot(x, estimated, label="Approximated")
            plt.plot(x, exact, label="Exact")
            plt.grid(True)
            plt.xlabel(r"$t$")
            plt.ylabel(f"$u_{feature + 1}$")
            plt.title(f"Comparison for the forcing $u_{feature}$")
            plt.legend()

            plt.savefig(os.path.join(path, f"feature_{feature + 1}.png"))

            plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Reading input arguments")

    parser.add_argument("--save_path", type=str)
    parser.add_argument("--N_tot", type=int, default=200)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--n_features", type=int, default=500)
    parser.add_argument("--N_samples", type=int, default=20)
    parser.add_argument("--N_epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, default="exec")

    args = parser.parse_args()

    save_path = args.save_path
    N_tot = args.N_tot
    m = args.m
    n_features = args.n_features
    N_samples = args.N_samples
    N_epochs = args.N_epochs
    case = args.case

    runner = TestDeepONet()

    if case == "exec":
        runner.test_deeponet_scalar_grf(
            N_tot=N_tot,
            m=m,
            n_features=n_features,
            N_samples=N_samples,
            N_epochs=N_epochs,
            path=save_path,
        )
        runner.plot(path=save_path)
    elif case == "plot":
        runner.plot(path=save_path)
