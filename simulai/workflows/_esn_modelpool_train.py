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

import contextlib
import glob
import json
import math
import os
import pickle
import sys
import warnings
from typing import Dict, List

import numpy as np
import optuna
from optuna.trial import FrozenTrial

from simulai.metrics import DeterminationCoeff
from simulai.models import ModelPool
from simulai.regression import EchoStateNetwork
from simulai.utilities import make_temp_directory

from ._cloud_object_storage import CloudObjectStorage


def _compute_damped_residual_block(resblock_damping, esn_input_size, target_ix):
    A = np.zeros((1, esn_input_size), dtype=float)
    A[0, target_ix] = 1 - resblock_damping
    return A


class ObjectiveESNIndependent:
    def __init__(
        self,
        train_data: np.ndarray = None,
        train_data_aux: np.ndarray = None,
        validation_data: np.ndarray = None,
        validation_data_aux: np.ndarray = None,
        target_ix: int = None,
        base_model_path: str = None,
        esn_override_configs: Dict = None,
        reservoir_config_space: Dict = None,
        pool_template: str = None,
    ):
        self.train_data = train_data
        self.train_data_aux = train_data_aux
        self.validation_data = validation_data
        self.validation_data_aux = validation_data_aux
        self.target_ix = target_ix
        self.base_model_path = base_model_path
        self.pool_template = None
        self.set_pool_template(pool_template)

        if esn_override_configs is None:
            esn_override_configs = dict()
        self.esn_override_configs = esn_override_configs

        self.reservoir_config_space = reservoir_config_space
        if self.reservoir_config_space is None:
            self.reservoir_config_space = self._default_reservoir_config_space

        self._pool_ix = ModelPool(
            config={
                "template": pool_template,
                "n_inputs": self._pool_n_inputs,
                "n_auxiliary": self.train_data_aux.shape[1],
                "n_outputs": self.train_data.shape[1],
            },
            model_type="EchoStateNetwork",
        )

        self.instability_penalty = 100

    @staticmethod
    def get_model_ids(base_model_path):
        ids = [
            os.path.basename(os.path.splitext(m)[0])
            for m in glob.glob(os.path.join(base_model_path, "*.pkl"))
        ]
        ids = sorted(ids, key=lambda p: int(p.split("_")[-1]))
        return ids

    def set_pool_template(self, pool_template):
        assert pool_template in [
            "independent_series",
            "no_communication_series",
        ], f"pool_template={self.pool_template} is not recognized"
        self.pool_template = pool_template

    @property
    def _default_reservoir_config_space(self):
        return {
            "radius": [0.1, 1.2],
            "sigma": [0.1, 1.2],
            "exp_beta": [-100, -1],
            "leak_rate": [0, 1],
            "activation": ["tanh"],
            "tau": [0, 1],
            "transformation": ["T1"],
            "resblock_damping": [0.0, 1.0],
            "estimate_bias_transition": [False, True],
            "estimate_linear_transition": [False, True],
            "input_augmented_reservoir": [False, True],
            "model_id": self.get_model_ids(self.base_model_path),
        }

    def optuna_objective(self, trial: optuna.Trial):
        float_vars = [
            "radius",
            "sigma",
            "exp_beta",
            "leak_rate",
            "tau",
            "resblock_damping",
        ]
        float_config = {
            fv: trial.suggest_float(fv, *self.reservoir_config_space[fv])
            for fv in float_vars
            if fv in self.reservoir_config_space
        }

        categorical_vars = [
            "activation",
            "transformation",
            "model_id",
            "estimate_bias_transition",
            "estimate_linear_transition",
            "input_augmented_reservoir",
        ]
        categorical_config = {
            cv: trial.suggest_categorical(cv, self.reservoir_config_space[cv])
            for cv in categorical_vars
            if cv in self.reservoir_config_space
        }

        configs = {**float_config, **categorical_config}
        return self.objective(configs)

    def get_esn_model_instance(
        self,
    ) -> EchoStateNetwork:
        return self._pool_ix.get_model_instance(index=self.target_ix)

    @property
    def _pool_n_inputs(self):
        return self.train_data.shape[1] + self.train_data_aux.shape[1]

    @property
    def _esn_n_inputs(self):
        if self.pool_template == "independent_series":
            return self.train_data.shape[1] + self.train_data_aux.shape[1]
        elif self.pool_template == "no_communication_series":
            return 1 + self.train_data_aux.shape[1]
        else:
            raise RuntimeError(f"pool_template={self.pool_template} is not recognized")

    def check_stability(self):
        esn = self.get_esn_model_instance()
        trainable_variables = esn.trainable_variables

        if self.pool_template == "independent_series":
            self_ix = self.target_ix
        elif self.pool_template == "no_communication_series":
            self_ix = 0
        else:
            raise RuntimeError(f"pool_template={self.pool_template} is not recognized")

        A_ix = 0
        if trainable_variables["A"] is not None:
            A_ix = trainable_variables["A"][0, self_ix]

        aug_ix = 0
        if esn.input_augmented_reservoir:
            aug_ix = trainable_variables["W_out"][0, -esn.number_of_inputs + self_ix]

        return abs(A_ix + aug_ix) < 1

    def _configure_pool(self, configs: Dict, trainable_variables: Dict = None):
        local_config = configs.copy()
        model_id = local_config.pop("model_id")
        if "exp_beta" in local_config:
            local_config["beta"] = 10 ** local_config.pop("exp_beta")
        if "resblock_damping" in local_config:
            local_config["A"] = _compute_damped_residual_block(
                local_config.pop("resblock_damping"), self._esn_n_inputs, self.target_ix
            )
        if trainable_variables is not None:
            local_config.update(trainable_variables)

        esn = EchoStateNetwork.restore(self.base_model_path, model_id)
        merged_configs = {**self.esn_override_configs, **local_config}
        esn.set_parameters(merged_configs)
        esn.set_reference()
        esn.reset()

        self._pool_ix.load_model_instance(esn, index=self.target_ix)

    def _fit(
        self,
    ):
        self._pool_ix.fit(
            input_data=self.train_data[0:-1, :],
            target_data=self.train_data[1:, :],
            auxiliary_data=self.train_data_aux[0:-1, :],
            index=self.target_ix,
        )

    def _extrapolate_target_model(
        self, init_state=None, init_aux=None, next_data=None, next_aux=None, reset=True
    ):
        if reset:
            self._pool_ix.set_reference()
            self._pool_ix.reset()

        extrapolation_data = []
        total_steps = next_aux.shape[0] + 1
        for n in range(total_steps):
            sys.stdout.write(f"\rPredicting step {n+1}/{total_steps}")
            sys.stdout.flush()
            with contextlib.redirect_stdout(None):
                s = self._pool_ix.predict(
                    initial_state=init_state,
                    auxiliary_data=init_aux,
                    index=self.target_ix,
                )

            extrapolation_data.append(s)

            if n < next_aux.shape[0]:
                init_state = next_data[n : n + 1, :].copy()
                init_state[:, self.target_ix : self.target_ix + 1] = s
                init_aux = next_aux[n : n + 1, :]

        sys.stdout.write("\r")
        sys.stdout.flush()

        extrapolation_data = np.vstack(extrapolation_data)

        return extrapolation_data

    def objective(self, configs: Dict):
        self._configure_pool(configs=configs)

        self._fit()

        extrapolation_data = self._extrapolate_target_model(
            init_state=self.train_data[-1:, :],
            init_aux=self.train_data_aux[-1:, :],
            next_data=self.validation_data[:-1, :],
            next_aux=self.validation_data_aux[:-1, :],
            reset=False,
        )

        # This value is intended to be as close as possible to zero
        objective_value = math.sqrt(
            math.pow(
                1
                - DeterminationCoeff()(
                    data=extrapolation_data,
                    reference_data=self.validation_data[
                        :, self.target_ix : self.target_ix + 1
                    ],
                ),
                2,
            )
        )

        is_stable = self.check_stability()
        if not is_stable:
            objective_value += self.instability_penalty

        return objective_value

    def cos_save_callback(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        cos_config,
        bucket_name,
        dir_path_in_bucket,
    ):
        esn = self.get_esn_model_instance()
        save_trainable_variables(
            cos_config=cos_config,
            bucket_name=bucket_name,
            dir_path_in_bucket=dir_path_in_bucket,
            study_name=study.study_name,
            trial_number=trial.number,
            trainable_variables=esn.trainable_variables,
        )


def _trial_filename(study: str, trial_number: int):
    assert study is not None
    assert trial_number is not None

    return f"study_{study}_trial_{str(trial_number)}.pkl"


def load_trainable_variables(
    cos_config, bucket_name, dir_path_in_bucket, study_name, trial_number
):
    success = False
    trainable_variables = {}
    if cos_config is not None:
        cos = CloudObjectStorage.from_config_json(cos_config)
        trial_filename = _trial_filename(study_name, trial_number)
        filename_in_bucket = os.path.join(dir_path_in_bucket, trial_filename)
        success = cos.check_file_exists(bucket_name, filename_in_bucket)
        if success:
            with make_temp_directory() as tmp_dir:
                saved_variables_filename = os.path.join(tmp_dir, trial_filename)
                cos.get_file(bucket_name, filename_in_bucket, saved_variables_filename)
                with open(saved_variables_filename, "rb") as saved_variables_file:
                    trainable_variables = pickle.load(saved_variables_file)

    return success, trainable_variables


def save_trainable_variables(
    cos_config,
    bucket_name,
    dir_path_in_bucket,
    study_name,
    trial_number,
    trainable_variables,
):
    if cos_config is not None:
        cos = CloudObjectStorage.from_config_json(cos_config)
        trial_filename = _trial_filename(study_name, trial_number)
        filename_in_bucket = os.path.join(dir_path_in_bucket, trial_filename)
        with make_temp_directory() as tmp_dir:
            saved_variables_filename = os.path.join(tmp_dir, trial_filename)
            try:
                with open(saved_variables_filename, "wb") as fp:
                    pickle.dump(trainable_variables, fp, protocol=4)
            except Exception as e:
                print(e, e.args)
            cos.put_file(bucket_name, filename_in_bucket, saved_variables_filename)


def optuna_assess_best_solution_ESNIndependent(
    train_data: np.ndarray = None,
    train_data_aux: np.ndarray = None,
    full_data: np.ndarray = None,
    full_data_aux: np.ndarray = None,
    target_ix: int = 0,
    base_model_path: str = None,
    study_name: str = None,
    storage: str = None,
    trial_number: int = None,
    cos_config: Dict = None,
    bucket_name: str = None,
    dir_path_in_bucket: str = None,
    recompute_trainable_variables: bool = False,
    pool_template: str = None,
):
    hyper_search = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    if trial_number is None:
        trial = hyper_search.best_trial
        trial_number = trial.number
    else:
        trial = hyper_search._storage.get_trial(
            hyper_search._storage.get_trial_id_from_study_id_trial_number(
                hyper_search._study_id, trial_number
            )
        )

    print(f"Extrapolating trial with number {trial.number}")

    if recompute_trainable_variables:
        print(f"Recomputing trainable variables")
        success_loading = False
        trainable_variables = None
    else:
        success_loading, trainable_variables = load_trainable_variables(
            cos_config, bucket_name, dir_path_in_bucket, study_name, trial_number
        )

        print(f"Loading trainable variables success_loading={success_loading}")

    o = ObjectiveESNIndependent(
        train_data=train_data,
        train_data_aux=train_data_aux,
        validation_data=None,
        validation_data_aux=None,
        target_ix=target_ix,
        base_model_path=base_model_path,
        reservoir_config_space=None,
        pool_template=pool_template,
    )

    o._configure_pool(configs=trial.params, trainable_variables=trainable_variables)

    if not success_loading:
        o._fit()
        esn = o.get_esn_model_instance()
        save_trainable_variables(
            cos_config,
            bucket_name,
            dir_path_in_bucket,
            study_name,
            trial_number,
            esn.trainable_variables,
        )

    extrapolation_data = o._extrapolate_target_model(
        init_state=full_data[0:1, :],
        init_aux=full_data_aux[0:1, :],
        next_data=full_data[1:, :],
        next_aux=full_data_aux[1:, :],
        reset=True,
    )

    return extrapolation_data


def optuna_objectiveESNIndependent(
    train_data: np.ndarray = None,
    train_data_aux: np.ndarray = None,
    validation_data: np.ndarray = None,
    validation_data_aux: np.ndarray = None,
    target_ix: int = 0,
    base_model_path: str = None,
    reservoir_config_space: Dict = None,
    n_trials: int = 1,
    study_name: str = None,
    storage: str = None,
    cos_config: Dict = None,
    bucket_name: str = None,
    dir_path_in_bucket: str = None,
    pool_template: str = None,
    do_grid_search: bool = False,
):
    o = ObjectiveESNIndependent(
        train_data=train_data,
        train_data_aux=train_data_aux,
        validation_data=validation_data,
        validation_data_aux=validation_data_aux,
        target_ix=target_ix,
        base_model_path=base_model_path,
        reservoir_config_space=reservoir_config_space,
        pool_template=pool_template,
    )

    sampler = None
    if do_grid_search:
        sampler = optuna.samplers.GridSampler(reservoir_config_space)
    hyper_search = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )

    def saver(s, t):
        o.cos_save_callback(s, t, cos_config, bucket_name, dir_path_in_bucket)

    hyper_search.optimize(o.optuna_objective, n_trials=n_trials, callbacks=[saver])


def optuna_assess_best_joint_solution_ESNIndependent(
    initial_data: np.ndarray = None,
    data_aux: np.ndarray = None,
    base_model_paths: List[str] = None,
    study_names: List[str] = None,
    storage: str = None,
    trial_numbers: List[int] = None,
    cos_config: Dict = None,
    bucket_name: str = None,
    study_paths_in_bucket: List[str] = None,
    compare_data: np.ndarray = None,
    pool_template: str = None,
    hidden_state_startup_burning_steps: int = 0,
):
    _n_inputs = initial_data.shape[1] + data_aux.shape[1]
    if pool_template == "independent_timeseries":
        _esn_n_inputs = _n_inputs
    elif pool_template == "no_communication_series":
        _esn_n_inputs = 1 + data_aux.shape[1]
    else:
        raise RuntimeError(f"Unrecognized pool_template={pool_template}")

    pool = ModelPool(
        config={
            "template": pool_template,
            "n_inputs": _n_inputs,
            "n_auxiliary": data_aux.shape[1],
            "n_outputs": initial_data.shape[1],
        },
        model_type="EchoStateNetwork",
    )

    for ix, study_name in enumerate(study_names):
        hyper_search = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

        if trial_numbers is not None:
            trial_number = trial_numbers[ix]
        else:
            trial_number = None

        if trial_number is None:
            trial = hyper_search.best_trial
            trial_number = trial.number
        else:
            trial = hyper_search._storage.get_trial(
                hyper_search._storage.get_trial_id_from_study_id_trial_number(
                    hyper_search._study_id, trial_number
                )
            )

        print(
            f"Extrapolating time_series={int(ix)}, trial with number {trial.number} in study={str(study_name)}"
        )

        dir_path_in_bucket = study_paths_in_bucket[ix]
        configs = trial.params
        success_loading, trainable_variables = load_trainable_variables(
            cos_config, bucket_name, dir_path_in_bucket, study_name, trial_number
        )

        if not success_loading:
            raise RuntimeError(
                f"Loading trainable variables success_loading={success_loading}"
            )

        local_config = configs.copy()
        model_id = local_config.pop("model_id")
        if "exp_beta" in local_config:
            local_config["beta"] = 10 ** local_config.pop("exp_beta")
        if "resblock_damping" in local_config:
            local_config["A"] = _compute_damped_residual_block(
                local_config.pop("resblock_damping"), _esn_n_inputs, ix
            )

        local_config.update(trainable_variables)

        esn = EchoStateNetwork.restore(base_model_paths[ix], model_id)
        esn.set_parameters({**local_config})
        esn.set_reference()
        esn.reset()

        pool.load_model_instance(esn, index=ix)

    one_step_ahead_list = []
    for i in range(hidden_state_startup_burning_steps):
        if compare_data is not None and i < compare_data.shape[0]:
            c = compare_data[i : i + 1, :]
        else:
            c = None
        one_step_ahead = pool.predict(
            initial_state=initial_data[i : i + 1, :],
            auxiliary_data=data_aux[i : i + 1, :],
            compare_data=c,
        )
        one_step_ahead_list.append(one_step_ahead)

    if (
        compare_data is not None
        and hidden_state_startup_burning_steps < compare_data.shape[0]
    ):
        c = compare_data[hidden_state_startup_burning_steps:, :]
    else:
        c = None
    extrapolation_data = pool.predict(
        initial_state=initial_data[hidden_state_startup_burning_steps:, :],
        auxiliary_data=data_aux[hidden_state_startup_burning_steps:, :],
        compare_data=c,
    )

    extrapolation_data = np.vstack((*one_step_ahead_list, extrapolation_data))

    return extrapolation_data


def define_reservoir_configs_for_affine_training(
    data_truncation_n: int = None,
    data_aux_truncation_n: int = None,
    cos_config: Dict = None,
    bucket_name: str = None,
    reservoir_config_path: str = None,
    mount_path: str = None,
    pool_template: str = None,
):
    with contextlib.ExitStack() as stack:
        if mount_path is None:
            mount_path = os.getenv("MOUNT_PATH", "")
        if mount_path == "":
            mount_path = stack.enter_context(make_temp_directory())

        base_model_path = os.path.join(mount_path, reservoir_config_path)
        os.makedirs(base_model_path, exist_ok=True)

        if pool_template == "independent_series":
            sub_model_number_of_inputs = (
                data_truncation_n + data_aux_truncation_n
            )  # size of the data
        elif pool_template == "no_communication_series":
            sub_model_number_of_inputs = 1 + data_aux_truncation_n  # size of the data
        else:
            raise RuntimeError(f"Unsupported pool_template={pool_template}")

        rc_config = {
            "reservoir_dim": 0,
            "sparsity_level": 1,
            "number_of_inputs": sub_model_number_of_inputs,
        }
        esn = EchoStateNetwork(**rc_config)
        esn.save(base_model_path, "EchoStateNetwork_0")
        model_filename = os.path.join(base_model_path, "EchoStateNetwork_0" + ".pkl")

        truncation = {
            "data_truncation_n": data_truncation_n,
            "data_aux_truncation_n": data_aux_truncation_n,
            "pool_template": pool_template,
        }
        truncation_filename = os.path.join(base_model_path, "truncation.json")
        with open(truncation_filename, "w") as f_truncation:
            json.dump(truncation, fp=f_truncation, indent=1)

        reservoir_config_space = {
            "estimate_bias_transition": [False, True],
            "estimate_linear_transition": [False, True],
            "model_id": ["EchoStateNetwork_0"],
        }
        reservoir_config_space_filename = os.path.join(
            base_model_path, "reservoir_config_space.json"
        )
        with open(reservoir_config_space_filename, "w") as f_reservoir_config_space:
            json.dump(reservoir_config_space, fp=f_reservoir_config_space, indent=1)

        if cos_config is not None:
            cos = CloudObjectStorage.from_config_json(cos_config)
            cos.put_file(
                bucket_name,
                f"{reservoir_config_path}/EchoStateNetwork_0.pkl",
                model_filename,
            )
            cos.put_file(
                bucket_name,
                f"{reservoir_config_path}/truncation.json",
                truncation_filename,
            )
            cos.put_file(
                bucket_name,
                f"{reservoir_config_path}/reservoir_config_space.json",
                reservoir_config_space_filename,
            )
