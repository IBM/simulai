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

import numpy as np
import optuna

try:
    import torch
except:
    print("It is necessary to configure it.")


# Searching up for a set of scalar parameters
class ParamHyperOpt:
    def __init__(
        self,
        params_intervals: dict = None,
        params_suggestions: dict = None,
        name: str = None,
        direction: str = "minimize",
        trainer_template=None,
        objective_function: callable = None,
        objective_wrapper=None,
        weights_initialization: str = None,
        others_params: dict = None,
        refresh: bool = False,
    ) -> None:
        self.params_intervals = params_intervals
        self.params_suggestions = params_suggestions
        self.name = name
        self.direction = direction
        self.trainer_template = trainer_template
        self.refresh = refresh

        self.history = 1
        # If trainer template a class instance, weights_initialization must be
        # a file containing weights to be used.
        self.weights_initialization = weights_initialization

        # The others_params are parameters keep fixed during the hyperopt
        # search, but that must be provided to the model instance
        self.others_params = others_params

        # Objective function ia a customized callable that defines how the
        # hyperoptimization algorithm must seek the best solutions
        self.objective_function = objective_function

        if objective_wrapper is not None:
            self.objective_wrapper = objective_wrapper
        else:
            self.objective_wrapper = self._default_instance_wrapper

        if self.trainer_template.raw == False:
            self.set_type = "soft"
            self._optuna_generate_instance = self._optuna_generate_instance_soft
        else:
            self.set_type = "hard"
            self._optuna_generate_instance = self._optuna_generate_instance_hard

        self.study = optuna.create_study(study_name=name, direction=direction)

        # If no others_params dictionary is provided
        # it is used an empty one
        if others_params is None:
            self.others_params = dict()

        # The option refresh clean up all the no more necessary models
        # in the save path keeping just the current best one
        if self.refresh is not False:
            assert "path_to_save" in self.others_params.keys(), (
                "If the option refresh is activated"
                "it is necessary to provide a path "
                f"to the saved models in order to {self}"
                f"be able of cleaning up them."
            )
            self.path_to_save = self.others_params["path_to_save"]
            self.refresher = self._refresh

        else:
            self.refresher = self._refresh_nothing

        # Checking up if the model template has an inner fit method
        if hasattr(self.trainer_template, "fit") is False:
            raise Exception(
                f"The model template {self.trainer_template} has no method fit"
            )

        self.there_are_datasets = False

        # Datasets as attributes
        self.input_train_data = None
        self.target_train_data = None
        self.auxiliary_train_data = None
        self.input_validation_data = None
        self.target_validation_data = None
        self.auxiliary_validation_data = None
        self.input_test_data = None
        self.target_test_data = None
        self.auxiliary_test_data = None

        # List for storing all the already instantiated trials ids
        self.models_ids_list = dict()

    def _set_data(self, data: list = None, name: str = None) -> None:
        indata, tardata, auxdata = tuple(data)

        assert indata.shape[0] == tardata.shape[0], (
            f" Dimensions mismatch in {name}."
            f" Input has shape {indata.shape}"
            f" and target {tardata.shape}"
        )

        if auxdata is not None:
            assert indata.shape[0] <= auxdata.shape[0], (
                f" Dimensions mismatch in {name}."
                f" Input has shape {indata.shape}"
                f" and auxiliary {auxdata.shape}"
            )
        else:
            print("Auxiliary data is not being used.")

        setattr(self, "input_" + name + "_data", indata)
        setattr(self, "target_" + name + "_data", tardata)
        setattr(self, "auxiliary_" + name + "_data", auxdata)  # auxdata can be None

    # Setting up the datasets
    def set_data(
        self,
        input_train_data: np.ndarray = None,
        target_train_data: np.ndarray = None,
        auxiliary_train_data: np.ndarray = None,
        input_validation_data: np.ndarray = None,
        target_validation_data: np.ndarray = None,
        auxiliary_validation_data: np.ndarray = None,
        input_test_data: np.ndarray = None,
        target_test_data: np.ndarray = None,
        auxiliary_test_data: np.ndarray = None,
    ) -> None:
        self._set_data(
            [input_train_data, target_train_data, auxiliary_train_data], "train"
        )
        self._set_data(
            [input_validation_data, target_validation_data, auxiliary_validation_data],
            "validation",
        )
        self._set_data([input_test_data, target_test_data, auxiliary_test_data], "test")

        self.there_are_datasets = True

    def _optuna_generate_instance_hard(self, trial: optuna.Trial):
        config = {
            key: getattr(trial, "suggest_" + value)(
                key, *self.params_intervals.get(key)
            )
            for key, value in self.params_suggestions.items()
        }

        # Including the others not adjustable parameters
        config.update(self.others_params)

        # Storing the trial number in order to send it to the model class
        config.update({"id": trial.number})

        print(f"Creating an instance from {self.trainer_template}")

        trainer_instance = self.trainer_template(config)

        # The model id is stored in a global attribute list
        self.models_ids_list[trial.number] = trainer_instance.model_id

        return trainer_instance

    def _optuna_generate_instance_soft(self, trial: optuna.Trial):
        config = {
            key: getattr(trial, "suggest_" + value)(
                key, *self.params_intervals.get(key)
            )
            for key, value in self.params_suggestions.items()
        }

        # Including the others not adjustable parameters
        config.update(self.others_params)

        # Storing the trial number in order to send it to the model class
        config.update({"id": trial.number})

        print(
            f"Using and modifying the instance {self.trainer_template} already created."
        )

        trainer_instance = copy.copy(self.trainer_template)
        trainer_instance.set_trial(trial_config=config)
        trainer_instance.soft_set()

        if self.weights_initialization is not None:
            print("Restoring pre-initialized weights.")
            trainer_instance.model.load_state_dict(
                torch.load(self.weights_initialization)
            )

        return trainer_instance

    # It stacks the required inputs
    def input_data(self, name: str = None):
        input_ = getattr(self, "input_" + name + "_data")
        auxiliary_ = getattr(self, "auxiliary_" + name + "_data")

        if auxiliary_ is not None:
            return np.hstack([input_, auxiliary_])
        else:
            return input_

    def _refresh_nothing(self):
        pass

    # Serial refresh
    def _refresh(self):
        models_ids_list = {
            key: value
            for key, value in self.models_ids_list.items()
            if key != self.study.best_trial.number
        }

        for number, m_id in models_ids_list.items():
            filename = os.path.join(self.path_to_save, m_id)

            print(f"Removing {filename}.")
            os.remove(filename + ".pkl")
            self.models_ids_list.pop(number)

    def _default_instance_wrapper(
        self, trainer_instance=None, objective_function: callable = None
    ):
        return objective_function(
            model=trainer_instance,
            input_validation_data=self.input_data(name="validation"),
            target_validation_data=self.target_validation_data,
        )

    def _objective_optuna_wrapper(self, trial: optuna.Trial):
        self.refresher()

        trainer_instance = self._optuna_generate_instance(trial)

        trainer_instance.fit(
            input_train_data=self.input_data(name="train"),
            target_train_data=self.target_train_data,
        )

        return self.objective_wrapper(
            trainer_instance=trainer_instance,
            objective_function=self.objective_function,
        )

    def optimize(self, n_trials: int = None):
        assert self.there_are_datasets == True, "The datasets were not informed."

        assert callable(self.objective_function), (
            f"The objective function must be a callable,"
            f" but received {type(self.objective_function)}"
        )

        self.study.optimize(self._objective_optuna_wrapper, n_trials=n_trials)

    def retrain_best_trial(self):
        if not hasattr(self.trainer_template, "params"):
            trainer_template = self.trainer_template
        else:
            trainer_template = self.trainer_template.__class__

        best_trial_params = self.study.best_trial.params

        # Including the others not adjustable parameters
        best_trial_params.update(self.others_params)
        best_trial_params["id"] = "chosen"

        trainer_instance = trainer_template(best_trial_params)

        print(f"Retraining using the best trial parameters {best_trial_params}")
        print(f"And the baseline trainer instance {trainer_instance}")

        trainer_instance.fit(
            input_train_data=self.input_train_data,
            target_train_data=self.target_train_data,
        )

        return trainer_instance
