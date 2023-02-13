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

import math
import os
import pickle
import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import optuna

from simulai.metrics import DeterminationCoeff, L2Norm
from simulai.models import ModelPool


# Class dedicated to the hyper-optimization
class ObjectiveFunction:
    def __init__(
        self,
        train_data=None,
        validation_data=None,
        config=None,
        model_name=None,
        data_path=None,
        pool_path=None,
    ):
        self.train_data = train_data
        self.validation_data = validation_data
        self.extra_stencil = 0
        self.fixed_group_size = None

        # Some required parameters
        self.group_size = None  # It will be overwritten with config content

        # Hyper-parameters intervals
        for key, value in config.items():
            setattr(self, key, value)

        self.l2_norm = L2Norm()

        # Guaranteeing groups with regular size
        if not self.fixed_group_size:
            self.group_size_intv = [
                n
                for n in range(2, self.group_size - self.extra_stencil)
                if not self.group_size % n
            ]
        else:
            self.group_size_intv = [self.fixed_group_size]

        self.model_name = model_name
        self.data_path = data_path
        self.pool_path = pool_path

        self.coeff = DeterminationCoeff()
        self.best_trial = self.InlineTrial()
        self.best_model = None

        # https://web.archive.org/web/20201108091210/http://effbot.org/pyfaq/what-kinds-of-global-value-mutation-are-thread-safe.htm
        self._trials_pool = dict()
        self._saved_trials = dict()

        if not self.pool_path:
            self.objective = self._raw_objective
        else:
            self.objective = self._reload_objective

    class InlineTrial:
        def __init__(self):
            self.number = 0

    # It displays the current status
    def callback(self, study, trial):
        best_trial = study.best_trial
        print("The best trial is {}".format(best_trial))

        pool = self._trials_pool.pop(
            trial.number
        )  # this callback must run once after the objective of the corresponding trial
        do_save = self._is_best_or_better_trial(
            trial.value, trial.number, best_trial, study.direction
        )
        if do_save:
            self.save(pool, value=trial.value, number=trial.number)
        else:
            # pool will be forgotten...
            pass

        # delete saved but not best
        saved = self.get_saved()
        for s in saved:
            trial_data = self._saved_trials.get(
                s, None
            )  # get trial_data[s] if a reference still exists, otherwise None
            if trial_data is not None:
                safe_delete = not self._is_best_or_better_trial(
                    trial_data["value"], s, best_trial, study.direction
                )
                if safe_delete:
                    self.delete_saved(s)

    @staticmethod
    def _is_best_or_better_trial(
        trial_value, trial_number, incumbent_best, studyDirection
    ):
        keep_trial = False
        if studyDirection is optuna.study.StudyDirection.MAXIMIZE:
            keep_trial = (
                trial_value >= incumbent_best.value
                or incumbent_best.number == trial_number
            )
        elif studyDirection is optuna.study.StudyDirection.MINIMIZE:
            keep_trial = (
                trial_value <= incumbent_best.value
                or incumbent_best.number == trial_number
            )
        else:
            assert studyDirection is optuna.study.StudyDirection.NOT_SET
            assert len(optuna.study.StudyDirection) == 3
            warnings.warn("study.direction is not set")

        return keep_trial

    def _raw_objective(self, trial_model_setup):
        print("Executing the trial: {}".format(trial_model_setup.number))
        # Configuration of ModelPool
        group_size = trial_model_setup.suggest_categorical(
            "group_size", getattr(self, "group_size_intv")
        )

        if len(self.group_size_intv) == 1:
            stencil_size = 0
        else:
            stencil_size = 2 * (group_size // 2) + self.extra_stencil

        reservoir_dim = trial_model_setup.suggest_int(
            "reservoir_dim", *getattr(self, "reservoir_dim")
        )
        beta = trial_model_setup.suggest_float("beta", *getattr(self, "beta"))
        transformation = trial_model_setup.suggest_categorical(
            "transformation", getattr(self, "transformation")
        )
        radius = trial_model_setup.suggest_float("radius", *getattr(self, "radius"))
        sigma = trial_model_setup.suggest_float("sigma", *getattr(self, "sigma"))

        pool_config = {
            "group_size": group_size,
            "stencil_size": stencil_size,
            "skip_size": group_size,
        }

        # Configuration of the sub-networks (in this case, ESN-RC)
        rc_config = {
            "reservoir_dim": reservoir_dim,
            "sparsity_level": getattr(self, "sparsity_level"),
            "radius": radius,
            "sigma": sigma,
            "beta": beta,
            "Win_init": "blockwise",
            "transformation": transformation,
            "alpha": 1e-1,
        }

        initial_state = self.train_data[-1:, :]

        horizon = self.validation_data.shape[0]

        # ESN-RC receives the prior state and forecasts the next one
        input_data = self.train_data[:, ...]  # In hyper-parameter search,
        # we use the validation dataset
        target_data = self.train_data[:, ...]

        # Instantiating the pool of sub-networks in order to execute
        # the parallel training
        pool = ModelPool(
            config=pool_config, model_type="EchoStateNetwork", model_config=rc_config
        )

        pool.fit(input_data=input_data, target_data=target_data)

        extrapolation_data = pool.predict(initial_state=initial_state, horizon=horizon)

        # This value is intended to be as close as possible to zero
        objective_value = math.sqrt(
            math.pow(
                1
                - DeterminationCoeff()(
                    data=extrapolation_data, reference_data=self.validation_data
                ),
                2,
            )
        )

        self._trials_pool[trial_model_setup.number] = pool
        print(f"trial number f{trial_model_setup.number} objecitve = {objective_value}")

        return objective_value

    def _reload_objective(self, trial_model_setup):
        print("Executing the trial: {}".format(trial_model_setup.number))

        beta = trial_model_setup.suggest_float("beta", *getattr(self, "beta"))
        radius = trial_model_setup.suggest_float("radius", *getattr(self, "radius"))
        sigma = trial_model_setup.suggest_float("sigma", *getattr(self, "sigma"))

        # Configuration of the sub-networks (in this case, ESN-RC)
        rc_config = {
            "radius": radius,
            "sigma": sigma,
            "beta": beta,
            "is_it_a_raw_model": False,
        }

        initial_state = self.train_data[-1:, :]

        horizon = self.validation_data.shape[0]

        # ESN-RC receives the prior state and forecasts the next one
        input_data = self.train_data[:, ...]  # In hyper-parameter search,
        # we use the validation dataset
        target_data = self.train_data[:, ...]

        # Instantiating the pool of sub-networks in order to execute
        # the parallel training
        with open(self.pool_path, "rb") as fp:
            print("Restoring a pre-trained model.")
            pool = pickle.load(fp)

        pool.set_parameters(rc_config, is_it_a_raw_model=False)

        pool.fit(input_data=input_data, target_data=target_data)

        extrapolation_data = pool.predict(initial_state=initial_state, horizon=horizon)

        # This value is intended to be as close as possible to zero
        objective_value = math.sqrt(
            math.pow(
                1
                - DeterminationCoeff()(
                    data=extrapolation_data, reference_data=self.validation_data
                ),
                2,
            )
        )

        self._trials_pool[trial_model_setup.number] = pool
        print(f"trial number f{trial_model_setup.number} objecitve = {objective_value}")

        return objective_value

    def load(self, number):
        model_path = os.path.join(
            self.data_path, self.model_name + "_" + f"{number}" + ".pkl"
        )

        with open(model_path, "rb") as fp:
            pool = pickle.load(fp)

        return pool

    def get_saved(self):
        # Careful it could be that some of these trials are already deleted when we try to consume them...
        # This list contains trails that exist or once existed
        return frozenset(self._saved_trials.keys())

        # This save method dumps ModelPool objects to pickle files

    def save(self, pool, value=-1, number=-1):
        model_path = os.path.join(
            self.data_path, self.model_name + "_" + f"{number}" + ".pkl"
        )

        pool.save(path=model_path)
        self._saved_trials[number] = {"value": value, "path": model_path}

    def delete_saved(self, number):
        # dict.pop is an atomic operation, i.e., only one worker will get the data, if it exist, all others will get None
        trial_data = self._saved_trials.pop(number, None)
        if trial_data is not None:
            try:
                os.remove(trial_data["path"])
            except:
                assert not os.path.exists(trial_data["path"])


if __name__ == "__main__":
    # Symbolic expressions for teh Lorenz system
    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--n_procs", type=int)
    parser.add_argument("--n_trials", type=int)
    parser.add_argument("--exec_type", type=str)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--restore_model", type=str, default=False)

    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path
    model_name = args.model_name
    n_procs = args.n_procs
    n_trials = args.n_trials
    exec_type = args.exec_type
    load_model = args.load_model
    restore_model = args.restore_model

    save_path = save_path + "/" + model_name + "/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    dataset = np.load(data_path)

    data = np.hstack([dataset["x"], dataset["y"], dataset["z"]])
    derivatives = np.hstack([dataset["dx_dt"], dataset["dy_dt"], dataset["dz_dt"]])

    train_percent = 0.8
    test_percent = 0.2
    # The validation percent is taken over the test data
    validation_percent = 0.5

    n_batches = data.shape[0]
    n_series = data.shape[-1]

    n_batches_train = int(train_percent * n_batches)
    n_batches_test = int(test_percent * n_batches)
    n_batches_validation = int(0.5 * n_batches_test)

    n_series = data.shape[1]

    # Connecting interfaces
    train_data = data[:n_batches_train, :]
    test_data = data[n_batches_train:, :]

    data_max = train_data.max(0)
    data_min = train_data.min(0)

    validation_data = test_data[:n_batches_validation, :]
    test_data = test_data[n_batches_validation:, :]

    initial_state = train_data[-1:, :]
    horizon = test_data.shape[0]

    """
        Modelling
    """

    fixed_group_size = 0

    if not load_model:
        # Some configurations used for testing
        if exec_type == "dev":
            group_size = n_series
            fixed_group_size = n_series
            reservoir_dim = [4000, 4100]
            sparsity_level = 10
            radius = [0.10, 0.11]
            sigma = [0.5, 0.51]
            beta = [1e-4, 1.1e-4]
            transformation = ["T1", "T2", "T3"]
        else:
            raise Exception("Specify an execution case. ")

        # Dictionary containing the hyper-parameters intervals to be passed
        # to Optuna
        config = {
            "group_size": group_size,
            "fixed_group_size": fixed_group_size,
            "reservoir_dim": reservoir_dim,
            "sparsity_level": sparsity_level,
            "radius": radius,
            "beta": beta,
            "sigma": sigma,
            "transformation": transformation,
        }

        # Class for wrapping the objective function used by Optuna
        modeler = ObjectiveFunction(
            train_data=train_data,
            validation_data=validation_data,
            config=config,
            model_name="lorenz",
            data_path=save_path,
            pool_path=restore_model,
        )

        # Instantiating and executing an Optuna study
        hyper_search = optuna.create_study(direction="minimize")
        hyper_search.optimize(
            modeler.objective,
            n_trials=n_trials,
            n_jobs=n_procs,
            callbacks=[modeler.callback],
        )

        best_model_number = modeler.best_trial.number

        print("The best model is {}".format(hyper_search.best_trial.number))

        pool = modeler.load(hyper_search.best_trial.number)

    else:
        print("Restoring model from disk.")

        with open(load_model, "rb") as fp:
            pool = pickle.load(fp)

    initial_state = validation_data[-1:, :]
    horizon = test_data.shape[0]

    """
        Dynamic extrapolation for the test data
    """

    extrapolation_data = pool.predict(initial_state=initial_state, horizon=horizon)

    """
        Open-Loop feedback (OLF) for the test data
    """

    n_steps = test_data.shape[0]
    one_shot_extrapolation_list = list()

    initial_state = validation_data[-1:, :]
    sub_horizon = 1

    for step in range(0, n_steps, sub_horizon):
        current_data = pool.predict(initial_state=initial_state, horizon=sub_horizon)
        initial_state = test_data[step][None, ...]

        one_shot_extrapolation_list.append(current_data)

    one_shot_extrapolation = np.vstack(one_shot_extrapolation_list)

    """
        Post-processing
    """

    n_variables = extrapolation_data.shape[-1]

    for var in range(n_variables):
        plt.plot(extrapolation_data[:, var], label="Approximated")
        plt.plot(test_data[:, var], label="Exact")

        plt.grid(True)
        plt.title("Dynamic Extrapolation")
        plt.legend()

        plt.savefig(os.path.join(save_path, "extrapolation_{}.png".format(var)))

        plt.close()

    for var in range(n_variables):
        plt.plot(one_shot_extrapolation[:, var], label="Approximated")
        plt.plot(test_data[:, var], label="Exact")

        plt.grid(True)
        plt.title("One-step ahead")
        plt.legend()

        plt.savefig(
            os.path.join(save_path, "one_shot_extrapolation_{}.png".format(var))
        )

        plt.close()
