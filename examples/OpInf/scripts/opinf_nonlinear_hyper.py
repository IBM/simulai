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

import copy
import os

import matplotlib.pyplot as plt
import numpy as np

os.environ["engine"] = "pytorch"

from examples.utils.lorenz_solver import lorenz_solver
from simulai.math.integration import LSODA, ClassWrapper
from simulai.metrics import L2Norm
from simulai.regression import OpInf
from simulai.templates import HyperTrainTemplate
from simulai.workflows import ParamHyperOpt


class LorenzJacobian:
    def __init__(self, sigma=None, rho=None, beta=None):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def __call__(self, data):
        x = data[0]
        y = data[1]
        z = data[2]

        return np.array(
            [[self.sigma, self.sigma, 0], [-z + self.rho, -1, -x], [y, x, -self.beta]]
        )


# This class is a base for the OpInf hyperparameter tuning.
class HyperOpInfTwoParameters(HyperTrainTemplate):
    def __init__(
        self,
        trial_config: dict = None,
        set_type="hard",
        path_to_model: str = None,
        other_params: dict = None,
    ):
        self.model = None

        self.path_to_model = path_to_model
        self.others_params = other_params

        self.tag = "model_"
        self.id = None

        required_keys = ["id", "tag"]

        for key in required_keys:
            assert (
                key in trial_config.keys()
            ), f"The required parameter {key} is not in others_params."

        for key, value in trial_config.items():
            setattr(self, key, value)

        if "baseline_model" in trial_config.keys():
            self.baseline_model = trial_config.pop("baseline_model")
        else:
            self.baseline_model = None

        super().__init__(trial_config=trial_config, set_type=set_type)

        self.model_id = self.tag + str(self.id)

    def _set_model(self):
        rc_config = {
            "lambda_linear": 10 ** self.trial_config["lambda_linear_exp"],
            "lambda_quadratic": 10 ** self.trial_config["lambda_quadratic_exp"],
        }

        if self.baseline_model is None:
            self.model = OpInf(bias_rescale=1e-15, solver="lstsq")
        else:
            print("Using baseline model.")
            self.model = copy.deepcopy(self.baseline_model)

        self.model.set(**rc_config)

    def fit(self, input_train_data=None, target_train_data=None):
        msg = self.model.fit(input_data=input_train_data, target_data=target_train_data)

        return msg


class ObjectiveWrapper:
    def __init__(self, test_data=None, t_test=None, initial_state=None):
        self.test_data = test_data
        self.t_test = t_test
        self.initial_state = initial_state

    def __call__(self, trainer_instance=None, objective_function=None):
        return objective_function(
            model=trainer_instance,
            initial_state=self.initial_state,
            t_test=self.t_test,
            test_field=self.test_data,
        )


def objective(
    model=None, initial_state=None, t_test=None, test_field=None, jacobian=None
):
    model.model.construct_K_op(op=jacobian)

    # Using the derivatives surrogate for time-integrating
    right_operator = ClassWrapper(model.model)

    solver = LSODA(right_operator)

    estimated_field = solver.run(initial_state, t_test)

    l2_norm = L2Norm()

    error = 100 * l2_norm(
        data=estimated_field, reference_data=test_field, relative_norm=True
    )

    return error


def test_opinf_nonlinear_int_lsoda():
    dt = 0.0001
    T_max = 100
    rho = 28
    beta = 8 / 3
    beta_str = "8/3"
    sigma = 10
    n_field = 3
    train_fraction = 0.6
    validation_fraction = 0.2
    n_trials = 200
    use_baseline = True

    jacobian = LorenzJacobian(sigma=sigma, rho=rho, beta=beta)

    initial_state = np.array([1, 0, 0])[None, :]

    lorenz_data, derivative_lorenz_data, time = lorenz_solver(
        rho=rho,
        dt=dt,
        T=T_max,
        sigma=sigma,
        initial_state=initial_state,
        beta=beta,
        beta_str=beta_str,
        data_path="on_memory",
        solver="RK45",
    )

    t = time
    n_steps = time.shape[0]
    nt = int(train_fraction * n_steps)
    nv = int(validation_fraction * n_steps)
    t_validation = t[nt : nt + nv]
    t_test = t[nt + nv :]

    train_field = lorenz_data[:nt]  # manufactured nonlinear oscillator data
    train_field_derivative = derivative_lorenz_data[:nt]

    validation_field = lorenz_data[
        nt : nt + nv
    ]  # manufactured nonlinear oscillator data

    test_field = lorenz_data[nt + nv :]  # manufactured nonlinear oscillator data

    initial_state_validation = train_field[-1:]
    initial_state_test = validation_field[-1:]

    # If a baseline is used, the datasets pre-processing is executed and recurrently used
    # along the process.
    if use_baseline:
        rc_config = {"lambda_linear": 10**0, "lambda_quadratic": 10**0}

        baseline_model = OpInf(bias_rescale=1e-15, solver="lstsq")
        baseline_model.set(**rc_config)

        baseline_model.fit(
            input_data=train_field, target_data=train_field_derivative, batch_size=1000
        )

    else:
        baseline_model = None

    params_intervals = {"lambda_linear_exp": [-5, 0], "lambda_quadratic_exp": [-5, 0]}

    params_suggestions = {"lambda_linear_exp": "float", "lambda_quadratic_exp": "float"}

    others_params = {
        "initial_state": initial_state_validation[0],
        "tag": "model_",
        "jacobian": jacobian,
        "baseline_model": baseline_model,
    }

    objective_wrapper = ObjectiveWrapper(
        test_data=validation_field,
        t_test=t_validation,
        initial_state=initial_state_validation[0],
    )

    hyper_search = ParamHyperOpt(
        params_intervals=params_intervals,
        params_suggestions=params_suggestions,
        name="lorenz_hyper_search",
        direction="minimize",
        trainer_template=HyperOpInfTwoParameters,
        objective_wrapper=objective_wrapper,
        objective_function=objective,
        others_params=others_params,
    )

    hyper_search.set_data(
        input_train_data=train_field,
        target_train_data=train_field_derivative,
        input_validation_data=validation_field,
        target_validation_data=validation_field,
        input_test_data=test_field,
        target_test_data=test_field,
    )

    hyper_search.optimize(n_trials=n_trials)

    best_model = hyper_search.retrain_best_trial()
    best_model.model.construct_K_op(op=jacobian)

    init_state = train_field[-1:]

    # Using the derivatives surrogate for time-integrating
    right_operator = ClassWrapper(best_model.model)

    solver = LSODA(right_operator)

    estimated_field = solver.run(initial_state_test[0], t_test)

    l2_norm = L2Norm()

    error = 100 * l2_norm(
        data=estimated_field, reference_data=test_field, relative_norm=True
    )

    print(f"Approximation error is {error} %")

    tags = ["x", "y", "z"]

    for var in range(n_field):
        plt.title(f"Variable {tags[var]}")
        plt.plot(t_test, test_field[:, var], label="Exact")
        plt.plot(t_test, estimated_field[:, var], label="Approximated")
        plt.xlabel("time (s)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"integrated_var_{var}.png")
        plt.show()
