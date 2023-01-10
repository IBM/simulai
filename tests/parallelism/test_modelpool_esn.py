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

import json
import random

import numpy as np
from unittest import TestCase
import os
import pickle
import glob

#import optuna
import optuna

from simulai.workflows import ObjectiveESNIndependent, optuna_assess_best_solution_ESNIndependent, define_reservoir_configs_for_affine_training, optuna_objectiveESNIndependent

from simulai.models import ModelPool
from simulai.utilities import make_temp_directory
from simulai.regression import EchoStateNetwork
from simulai.special import reservoir_generator

class TestModelPoolESN(TestCase):

    def setUp(self) -> None:

        self.test_case = None
        self.atol = 1e-10

    def test_reservoir_creation(self):

        number_of_reservoirs = 20
        reservoir_dim = 1000
        sparsity_level = 0.2*reservoir_dim

        reservoirs = reservoir_generator(number_of_reservoirs=number_of_reservoirs,
                                         sparsity_level=sparsity_level,
                                         reservoir_dim=reservoir_dim)

    def test_define_reservoir_configs_for_affine_training(self):

        for pool_template in ['no_communication_series', 'independent_series', ]:

            with make_temp_directory() as tmp:
                define_reservoir_configs_for_affine_training(
                    data_truncation_n=5,
                    data_aux_truncation_n=3,
                    reservoir_config_path='reservoir_config_path',
                    mount_path=tmp,
                    pool_template=pool_template)

                self.assertTrue(len(os.listdir(os.path.join(tmp, 'reservoir_config_path'))) == 3, 'Must write 3 config files')



    def test_modelpool_ESN_damping(self):

        # Configuration of the sub-networks (in this case, ESN-RC)
        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1',
            'solver': 'linear_system',  # This option provides an interface to the SciPy.linalg solvers
            'tau': 0.1
        }
        pool = self._run_ESN(rc_config)
        for i, e in pool.model_instances_list.items():
            damping = e.kappa
            self.assertTrue(damping[0, int(rc_config['tau']*damping.size)] <= 0.99)
            self.assertTrue(damping[0, int(rc_config['tau']*damping.size)+1] >= 0.99)

        self.assertTrue(True, 'end')

    def test_ESN_save_load(self):

        nt = 10
        ns = 3
        data = np.random.rand(nt, ns)
        ix = slice(1, 2)

        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1',
            'solver': 'linear_system',  # This option provides an interface to the SciPy.linalg solvers
            'number_of_inputs': data.shape[1],
        }

        esn = EchoStateNetwork(**rc_config)

        # sets the solution
        esn.fit(input_data=data[:-1, :], target_data=data[1:, ix])

        with make_temp_directory() as tmp_dir:
            esn.save(tmp_dir, 'model')
            esn_2 = EchoStateNetwork.restore(tmp_dir, 'model')

        for k in vars(esn).keys():
            if k in {'_construct_r_til', 'solver_op', 'global_matrix_constructor', 'logger'}:
                continue
            if k in {'s_reservoir_matrix'}:
                ok = np.all(getattr(esn, k).todense() == getattr(esn_2, k).todense())
            else:
                ok = np.all(getattr(esn, k) == getattr(esn_2, k))

            self.assertTrue(ok, f'esn.{k} is not equal')

        self.assertTrue(True)

    def test_modelpool_ESN_direct_inversion(self):

        # Configuration of the subnetworks (in this case, ESN-RC)
        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1'
        }
        self._run_ESN(rc_config)

    def test_modelpool_ESN_linear_system(self):

        # Configuration of the sub-networks (in this case, ESN-RC)
        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1',
            'solver': 'linear_system'  # This option provides an interface to the SciPy.linalg solvers
        }

        self._run_ESN(rc_config)

    def _train_single_model_in_pool(self, model_path, model_id, index):

        field_train_data, forcings_train_data, field_test_data, forcings_test_data = self.get_manufactured_dataset()

        input_data = np.hstack([field_train_data, forcings_train_data])[:-1]
        target_data = field_train_data[1:]

        pool_config = {'template': 'independent_series',
                       'n_inputs': field_train_data.shape[1] + forcings_train_data.shape[1],
                       'n_outputs': field_train_data.shape[1],
                       'n_auxiliary': forcings_train_data.shape[1]
                       }

        pool_single = ModelPool(config=pool_config, model_type='EchoStateNetwork')
        pool_single.load_model(model_path, model_id, index)

        pool_single.fit(input_data=input_data, target_data=target_data, index=index)

        m1 = pool_single._get_regression_class().restore(model_path, model_id)
        m2 = pool_single.model_instances_list[pool_single.model_ids_list[0]]

        for label in ['W_out', 'current_state', 'reference_state']:
            self.assertTrue(np.array_equal(getattr(m1, label), getattr(m2, label)))

    def test_modelpool_ESN_independently_multi_fit(self):

        # Configuration of the sub-networks (in this case, ESN-RC)
        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1'
        }

        """ The option 'template' as 'independent_series' corresponds to the dictionary:
            {
                'group_size': 1,
                'stencil_size': 0,
                'skip_size': 1
            }
        """
        pool_config = {'template': 'independent_series'}

        self.test_case = "independent_series_multi_fit"

        pool, extrapolation_data, one_shot_extrapolation = self._run_ESN(rc_config, pool_config=pool_config, output=True)
        with make_temp_directory() as tmp_dir:
            for k in pool.model_ids_list:
                pool.save_model(tmp_dir, k)

            model_pool_path = os.path.join(tmp_dir, 'model_pool_config_test_save.json')
            pool.save_pool_config(path=model_pool_path)

            self._train_single_model_in_pool(tmp_dir, pool.model_ids_list[2], 2)

            new_pool = ModelPool(config=pool_config, model_type='EchoStateNetwork',
                                 model_config=rc_config)

            for k in pool.model_ids_list:
                new_pool.load_model(tmp_dir, k)

        # The configuration dictionary is passed in order to restore the model, but it could be restored via a json file
        pool_2, extrapolation_data_2, one_shot_extrapolation_2 = self._run_ESN(self, pool_config=pool_config, fit=False,
                                                                               model=new_pool, output=True)

        self.assertTrue(np.array_equal(extrapolation_data, extrapolation_data_2))
        self.assertTrue(np.array_equal(one_shot_extrapolation, one_shot_extrapolation_2))

        self.assertTrue(True)

    def test_modelpool_ESN_independently_single_fit(self):

        # Configuration of the subnetworks (in this case, ESN-RC)
        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1'
        }

        """ The option 'template' as 'independent_series' corresponds to the dictionary:
            {
                'group_size': 1,
                'stencil_size': 0,
                'skip_size': 1
            }
        """
        pool_config = {'template': 'independent_series'}

        self.test_case = "independent_series_single_fit"

        self._run_ESN(rc_config, pool_config=pool_config)

    ''' Maybe this test is no more useful
    def test_modelpool_ESN_no_parallelism(self):

        # Configuration of the sub-networks (in this case, ESN-RC)
        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1'
        }

        """ The option 'template' as 'independent_series' corresponds to the dictionary:
            {
                'group_size': -1,
                'stencil_size': 0,
                'skip_size': 1
            }
        """

        pool_config = {'template': 'no_parallelism'}

        self._run_ESN(rc_config, pool_config=pool_config)
    '''

    def test_modelpool_ESN_save_restore_entire_model(self):

        # Configuration of the sub-networks (in this case, ESN-RC)
        rc_config = {
            'reservoir_dim': 250,
            'sparsity_level': 6,
            'radius': 0.5,
            'sigma': 0.5,
            'beta': 1e-4,
            'Win_init': 'blockwise',
            'transformation': 'T1'
        }

        with make_temp_directory() as tmp_dir:
            model_object_path = os.path.join(tmp_dir, 'model_pool_test_save.pkl')

            pool, dynamic, one_shot = self._run_ESN(rc_config, output=True)
            pool.save(path=model_object_path)
            for model_id in pool.model_ids_list:
                pool.save_model(tmp_dir, model_id)

            with open(model_object_path, "rb") as fp:
                _pool = pickle.load(fp)
                _, dynamic_, one_shot_  = self._run_ESN(rc_config, fit=False, model=_pool,
                                                                            output=True)

                assert (np.array_equal(dynamic_,dynamic) and
                        np.array_equal(one_shot_, one_shot)), "The saved and the restored model are not equal"


    def get_manufactured_dataset(self):
        # One-dimensional case
        # Constructing dataset
        train_factor = 0.50

        N = 40
        Nt = 100

        N_train = int(train_factor * Nt)

        # Constructing dataset
        x = np.linspace(0, 1, N)
        t = np.linspace(0, 1, Nt)
        i = np.linspace(1, 10, N)
        j = np.linspace(1, 10, Nt)

        T, X = np.meshgrid(t, x, indexing='ij')
        J, I = np.meshgrid(j, i, indexing='ij')

        Z = np.sin(J * np.pi * T) * np.cos(I * np.pi * X)

        n_forcings = 10
        Z_field = Z[:, :-n_forcings]
        Z_forcings = Z[:, -n_forcings:]

        field_train_data = Z_field[:N_train, :]
        field_test_data = Z_field[N_train:, :]

        forcings_train_data = Z_forcings[:N_train, :]
        forcings_test_data = Z_forcings[N_train:, :]

        return field_train_data, forcings_train_data, field_test_data, forcings_test_data

    def _run_ESN(self, rc_config, pool_config=None, fit=True, model=None, output=False):

        field_train_data, forcings_train_data, field_test_data, forcings_test_data = self.get_manufactured_dataset()

        input_data = field_train_data[:-1]
        auxiliary_data = forcings_train_data[:-1]
        target_data = field_train_data[1:]

        n_fields = field_train_data.shape[1]
        n_forcings = forcings_train_data.shape[1]

        N = field_train_data.shape[1] + forcings_train_data.shape[1]

        # Default config dictionary
        if pool_config is None:

            pool_config = {
                'group_size': 10,
                'stencil_size': 10,
                'skip_size': 10
            }

        # Complementing pool_config with information about n_inputs and n_outputs
        pool_config['n_inputs'] = N
        pool_config['n_outputs'] = n_fields
        pool_config['n_auxiliary'] = n_forcings
        # The dictionary pool_config could be directly declared, but it is done in stages given the
        # characteristics of the function _run_ESN, which is used in multiple different cases

        if fit and model is None:

            pool = ModelPool(config=pool_config, model_type='EchoStateNetwork',
                             model_config=rc_config)

            if self.test_case is not "independent_series_multi_fit":
                pool.fit(input_data=input_data, target_data=target_data,
                         auxiliary_data=auxiliary_data)
            else:
                for group_index in range(n_fields):
                    pool.fit(input_data=input_data, target_data=target_data,
                             auxiliary_data=auxiliary_data, index=group_index)

        elif not fit and model is not None:
            pool = model

        else:
            raise Exception(f"pool and fit are respectively {model} and {fit}. "
                            f"It is not possible to execute.")

        initial_state = field_train_data[-1:]

        horizon = field_test_data.shape[0]

        extrapolation_data = pool.predict(initial_state=initial_state,
                                          horizon=horizon,
                                          auxiliary_data=forcings_test_data)
        n_steps = horizon
        one_shot_extrapolation_list = list()

        sub_horizon = 1

        pool.reset()

        for step in range(0, n_steps, sub_horizon):

            current_data = pool.predict(initial_state=initial_state,
                                        horizon=sub_horizon,
                                        auxiliary_data=forcings_test_data)

            initial_state = field_test_data[step:step+1]

            one_shot_extrapolation_list.append(current_data)

        one_shot_extrapolation = np.vstack(one_shot_extrapolation_list)

        assert np.array_equal(extrapolation_data[0], one_shot_extrapolation[0]),\
            "The first iteration must be equal for both the short term and the dynamic extrapolation"

        pool.reset()

        if output:
            return pool, extrapolation_data, one_shot_extrapolation
        else:
            return pool

    def test_training_ident_Wout_perfect_esn(self):

        ns = 3  # number of forcing terms
        nt = 11
        esn_input_data = np.random.rand(nt, ns)  # manufactured observed data
        reservoir_dim = 5

        for input_augmented_reservoir in [True, False]:

            rc_config = {
                'reservoir_dim': reservoir_dim,
                'sparsity_level': 4,
                'radius': 0.5,
                'sigma': 0.5,
                'beta': 0,
                'number_of_inputs': ns,
                'input_augmented_reservoir': input_augmented_reservoir,
            }

            with make_temp_directory() as default_model_dir:

                esn = EchoStateNetwork(**rc_config)
                esn.A = np.random.rand(esn.state_dim, esn.number_of_inputs)*0.1
                esn.b = np.random.rand(esn.state_dim)*0.1
                esn.save(default_model_dir, "my_esn")
                esn.W_out = np.eye(esn.state_dim)

                esn.set_reference(esn.default_state)
                esn.reset()
                outs = []
                for step in range(esn_input_data.shape[0]):
                    outs.append(esn.step(esn_input_data[step, :]))
                outs = np.vstack(outs)

                esn_2 = EchoStateNetwork.restore(default_model_dir, "my_esn")
                esn_2.fit(input_data=esn_input_data, target_data=outs)

                esn_2.set_reference(esn.default_state)
                esn_2.reset()
                outs_2 = []
                for step in range(esn_input_data.shape[0]):
                    outs_2.append(esn_2.step(esn_input_data[step, :]))
                outs_2 = np.vstack(outs_2)

                max_error = np.max(np.abs(outs_2-outs))
                print(f'max_error={max_error}')
                self.assertTrue(max_error < 1e-10)


    def test_training_predicting_perfect_esn(self):

        ns = 3  # number of forcing terms
        nt = 23
        initial_data = np.random.rand(ns)  # manufactured observed data

        for input_augmented_reservoir in [False, True]:
            if input_augmented_reservoir:
                estimate_linear_transitions = [False, ]
                estimate_bias_transitions = [False, ]
            else:
                estimate_linear_transitions = [False, True,]
                estimate_bias_transitions = [False, True,]

            for estimate_linear_transition in estimate_linear_transitions:
                for estimate_bias_transition in estimate_bias_transitions:

                    rc_config = {
                        'reservoir_dim': 5,
                        'sparsity_level': 4,
                        'radius': 0.5,
                        'sigma': 0.5,
                        'beta': 0,
                        'number_of_inputs': ns,
                        'input_augmented_reservoir': input_augmented_reservoir,
                        'solver': 'linear_system',
                        'estimate_bias_transition': estimate_bias_transition,
                        'estimate_linear_transition': estimate_linear_transition,
                    }

                    with make_temp_directory() as default_model_dir:

                        esn = EchoStateNetwork(**rc_config)
                        esn.A = np.random.rand(ns, esn.number_of_inputs)*0.1
                        esn.b = np.random.rand(ns)*0.1
                        esn.save(default_model_dir, "my_esn")
                        if estimate_linear_transition:
                            esn.W_in *= 0
                            esn.s_reservoir_matrix *= 0
                            esn.A = np.random.rand(ns, esn.number_of_inputs) * 0.1
                        if estimate_bias_transition:
                            esn.W_in *= 0
                            esn.s_reservoir_matrix *= 0
                            esn.b = np.random.rand(ns)*0.1

                        W_out = np.random.rand(ns, esn.state_dim)*0.4
                        esn.W_out = W_out

                        esn.set_reference(esn.default_state)
                        esn.reset()
                        outs = esn.predict(initial_data, nt)

                        esn_2 = EchoStateNetwork.restore(default_model_dir, "my_esn")
                        esn_2.fit(input_data=np.vstack((initial_data, outs[0:-1, :])),
                                  target_data=outs)

                        esn_2.set_reference(esn_2.default_state)
                        esn_2.reset()
                        outs_2 = esn_2.predict(initial_data, nt)

                        max_error_A = np.max(np.abs(esn.A-esn_2.A))
                        print(f'max_error_A={max_error_A}')
                        self.assertTrue(max_error_A < 1e-8)

                        max_error_b = np.max(np.abs(esn.b-esn_2.b))
                        print(f'max_error_b={max_error_b}')
                        self.assertTrue(max_error_b < 1e-8)

                        max_error = np.max(np.abs(outs_2-outs))
                        print(f'max_error={max_error}')
                        self.assertTrue(max_error < 1e-8)

    def test_grid_search_for_linear_training(self):

        data_truncation_n = 1  # number of field values to predict
        data_aux_truncation_n = 2  # number of forcing terms
        nt = 10  # size of time steps

        train_data = np.random.rand(nt, data_truncation_n)  # manufactured observed data
        train_data_aux = np.random.rand(nt, data_aux_truncation_n)  # manufactured observed data
        validation_data = np.random.rand(nt, data_truncation_n)  # manufactured observed data
        validation_data_aux = np.random.rand(nt, data_aux_truncation_n)  # manufactured observed data
        target_ix = 0

        with make_temp_directory() as tmp:
            pool_template = 'no_communication_series'

            define_reservoir_configs_for_affine_training(
                data_truncation_n=data_truncation_n,
                data_aux_truncation_n=data_aux_truncation_n,
                reservoir_config_path='reservoir_config_path',
                mount_path=tmp,
                pool_template=pool_template)

            base_model_path = os.path.join(tmp, 'reservoir_config_path')
            reservoir_config_space_filename = os.path.join(base_model_path, 'reservoir_config_space.json')
            with open(reservoir_config_space_filename, 'r') as f_res_config:
                reservoir_config_space = json.load(f_res_config)

            n_trials = 3 #int(np.prod([len(v) for k, v in reservoir_config_space.items()]))
            study_name = 'my_affine_test_study'
            storage = optuna.storages.InMemoryStorage()

            optuna_objectiveESNIndependent(
                train_data=train_data,
                train_data_aux=train_data_aux,
                validation_data=validation_data,
                validation_data_aux=validation_data_aux,
                target_ix=target_ix,
                base_model_path=base_model_path,
                reservoir_config_space=reservoir_config_space,
                n_trials=n_trials,
                study_name=study_name,
                storage=storage,
                pool_template=pool_template,
                do_grid_search=True,
            )
            study = optuna.load_study(storage=storage, study_name=study_name)
            trials = study.get_trials()
            self.assertEqual(len(trials), n_trials)
            for trial in trials:
                params = trial.params
                for k, v in params.items():
                    self.assertTrue(v in reservoir_config_space[k])

    #def test_perfect_manufactured_optimization_model_pool_esn(self):

    #    n_field = 3  # number of field values to predict
    #    n_forcing = 8  # number of forcing terms
    #    nt = 130  # size of time steps

    #    initial_field_data = np.random.rand(1, n_field)  # manufactured observed data
    #    forcings_data = np.random.rand(nt, n_forcing)  # manufactured observed data

    #    n_esn = 17  # 20 random models search space for the solution

    #    model_type = 'EchoStateNetwork'
    #    for zero_reservoir_dim in [False, True]:
    #        for pool_template in ['no_communication_series', 'independent_series', ]:
    #            if pool_template == 'independent_series':
    #                sub_model_number_of_inputs = n_field + n_forcing  # size of the data
    #            else:
    #                sub_model_number_of_inputs = 1 + n_forcing  # size of the data
    #
    #            with make_temp_directory() as default_model_dir:

                    ####
                    ####  start: Create and save random EchoStateNetworks
                    ####

    #                for i in range(n_esn):
    #                    reservoir_dim = random.randint(7, 15)
    #                    if zero_reservoir_dim:
    #                        reservoir_dim = 0

    #                    rc_config = {
    #                        'reservoir_dim': reservoir_dim,
    #                        'sparsity_level': 6,
    #                        'radius': 0.5,
    #                        'sigma': 0.5,
    #                        'beta': 0,
    #                        'leak_rate': 1,
    #                        'activation': 'tanh',
    #                        'tau': 0,
    #                        'transformation': 'T1',
    #                        'number_of_inputs': sub_model_number_of_inputs,
    #                        'estimate_bias_transition': False,
    #                        'estimate_linear_transition': False,
    #                        'input_augmented_reservoir': False,
    #                    }

    #                    esn = EchoStateNetwork(**rc_config)
    #                    esn.A = np.random.rand(1, esn.number_of_inputs)*0.1
    #                    esn.b = np.random.rand(1)*0.1
    #                    esn.save(default_model_dir, f'{model_type}_{str(i)}')  # This ESN has no W_out matrix

                    ####
                    ####  end: Create and save random EchoStateNetworks
                    ####

                    ######
                    ######  start: Create and simulate a manufactured solution pool
                    ######

    #                solution_pool = ModelPool(config={'template': pool_template,
    #                                                  'n_inputs': n_field + n_forcing,
    #                                                  'n_auxiliary': n_forcing,
    #                                                  'n_outputs': n_field},
    #                                          model_type=model_type)

    #                model_solution = {}
    #                model_solution_hyperparameters_from = {}

    #                for i in range(n_field):

    #                    random_model_id = f'{model_type}_{str(random.randint(0,n_esn-1))}'
    #                    solution_pool.load_model(path=default_model_dir,
    #                                             model_id=random_model_id,
    #                                             index=i)

    #                    mi_id = solution_pool._make_id(i)
    #                    mi = solution_pool.model_instances_list[mi_id]
    #                    W_out = np.random.rand(1, mi.state_dim)
    #                    mi.W_out = W_out
    #                    model_solution[mi_id] = W_out
    #                    model_solution_hyperparameters_from[mi_id] = random_model_id

    #               solution_pool.reset()
    #                predicted_states = solution_pool.predict(initial_state=initial_field_data,
    #                                                         auxiliary_data=forcings_data)
    #                target_states = predicted_states
    #                field_data = np.vstack((initial_field_data, target_states[0:-1, :]))


                    ######
                    ######   end: Create and simulate a manufactured solution pool
                    ######


                    ######
                    ######   start: find the optimal model_solution_hyperparameters_from and model_solution provided
                    ######   start: default_model_dir, initial_field_data, field_data, and forcings_data
                    ######

    #                fit_model_solution = {}
    #                errors = list()

    #                v_ix = int(field_data.shape[0] * 0.9)

    #                 for ix in range(n_field):
    #                    mi_id = solution_pool._make_id(ix)
    #                    hs = HyperSearchObjective(ix, default_model_dir, target_states, field_data, forcings_data, model_type, pool_template)
    #                    error, esn = hs.objective(model_solution_hyperparameters_from[f'{model_type}_{str(ix)}'])
    #                    fit_model_solution[mi_id] = esn.W_out
    #                    errors.append(error)

    #                    o = ObjectiveESNIndependent(train_data=field_data[:v_ix, :],
    #                                                train_data_aux=forcings_data[:v_ix, :],
    #                                                validation_data=field_data[v_ix:, :],
    #                                                validation_data_aux=forcings_data[v_ix:, :],
    #                                                target_ix=ix,
    #                                                base_model_path=default_model_dir,
    #                                                pool_template=pool_template)

    #                    conf_ix = {**rc_config, 'model_id': model_solution_hyperparameters_from[f'{model_type}_{str(ix)}']}
    #                    conf_ix.pop('reservoir_dim', None)
    #                    conf_ix.pop('sparsity_level', None)
    #                    conf_ix.pop('number_of_inputs', None)
    #                    e2 = o.objective(conf_ix)
    #                    errors.append(e2)

    #                for key, value in fit_model_solution.items():

    #                    max_atol = np.max(np.concatenate((np.abs(model_solution[key] - value),[[0]],), axis=1))
    #                    assert np.allclose(model_solution[key], value, atol=self.atol),f"The W_out matrix for {key} are" \
    #                                                                                   f" not close enough for the model {key}" \
    #                                                                                   f"with difference  {max_atol} > {self.atol}"

    #                    print(f"W_out verification passed for {key} with difference <= {self.atol} for each entry")

    #                print(f"The errors list is {errors}")
    #                for item in errors:
    #                    self.assertAlmostEqual(item, 0)


    #                for ix in range(n_field):
    #                    storage = optuna.storages.InMemoryStorage()
    #                    study_name = f'test_{ix}'
    #                    optuna_search = optuna.create_study(direction='minimize', storage=storage, study_name=study_name)
    #                    conf_ix = {**rc_config, 'model_id': model_solution_hyperparameters_from[f'{model_type}_{str(ix)}']}
    #                    conf_ix.pop('reservoir_dim', None)
    #                    conf_ix.pop('sparsity_level', None)
    #                    conf_ix.pop('number_of_inputs', None)
    #                    conf_ix['exp_beta'] = -100


    #                    o = ObjectiveESNIndependent(train_data=field_data[:v_ix, :],
    #                                            train_data_aux=forcings_data[:v_ix, :],
    #                                            validation_data=field_data[v_ix:, :],
    #                                            validation_data_aux=forcings_data[v_ix:, :],
    #                                            target_ix=ix,
    #                                            base_model_path=default_model_dir,
    #                                            pool_template=pool_template)


    #                    # having resblock_damping in reservoir_config_space will force the algorithm to impose a linear transition matrix
    #                    reservoir_config_space = o._default_reservoir_config_space
    #                    reservoir_config_space.pop('resblock_damping')
    #                    o.reservoir_config_space = reservoir_config_space


    #                    optuna_search.enqueue_trial(conf_ix)
    #                    optuna_search.optimize(o.optuna_objective, n_trials=2)

    #                    extrapolation_data = optuna_assess_best_solution_ESNIndependent(
    #                        train_data=field_data[:v_ix, :],
    #                        train_data_aux=forcings_data[:v_ix, :],
    #                        full_data=field_data[:-1, :],
    #                        full_data_aux=forcings_data[:-1, :],
    #                        target_ix=ix,
    #                        base_model_path=default_model_dir,
    #                        study_name=study_name,
    #                        storage=storage,
    #                        pool_template=pool_template
    #                    )
    #                    error_extrapolation = np.max(np.abs(field_data[1:, ix:ix+1]-extrapolation_data))
    #                    self.assertAlmostEqual(error_extrapolation, 0.0)

    #            self.assertTrue(True)

#    def _optimize_independent_series(self, ix, default_model_dir, initial_field_data, field_data, forcings_data, model_type):

#        hs = HyperSearchObjective(ix, default_model_dir, initial_field_data, field_data, forcings_data, model_type)
#        hyper_search_alg = optuna.create_study()
#        hyper_search_alg.optimize(hs, n_trials=50)
#        self.assertTrue(True)


class HyperSearchObjective:

    def __init__(self, ix, default_model_dir, target_states, field_data, forcings_data, model_type, pool_template):
        self.ix = ix
        self.default_model_dir = default_model_dir
        self.models = [os.path.basename(os.path.splitext(m)[0]) for m in glob.glob(os.path.join(default_model_dir, "*.pkl"))]
        self.target_states = target_states
        self.field_data = field_data
        self.forcings_data = forcings_data
        self.model_type = model_type
        self.pool_template = pool_template

#    def __call__(self, trial: optuna.Trial):
#        model = trial.suggest_categorical('model', self.models)
#        return self.objective(model)

    def objective(self, model):
        p_ix = ModelPool(config={'template': self.pool_template,
                                              'n_inputs': self.field_data.shape[1] + self.forcings_data.shape[1],
                                              'n_auxiliary': self.forcings_data.shape[1],
                                              'n_outputs': self.field_data.shape[1]},
                                  model_type=self.model_type)
        p_ix.load_model(self.default_model_dir, model, self.ix)

        p_ix.fit(input_data=self.field_data,
                 target_data=self.target_states,
                 auxiliary_data=self.forcings_data,
                 index=self.ix)

        p_ix.set_reference()
        p_ix.reset()

        init_state = self.field_data[0:1, :]
        extrapolation_data = []
        for n in range(self.field_data.shape[0]):
            s = p_ix.predict(initial_state=init_state,
                             horizon=1,
                             auxiliary_data=self.forcings_data[n:n+1, :],
                             index=self.ix)

            extrapolation_data.append(s)

            if n < self.field_data.shape[0] - 1:
                init_state = self.field_data[n+1:n+2, :].copy()
                init_state[:, self.ix: self.ix+1] = s

        target = np.vstack(extrapolation_data)
        error = np.linalg.norm(np.reshape(self.target_states[:, self.ix:self.ix+1] - target, (-1, )))

        esn = p_ix.model_instances_list[p_ix._make_id(self.ix)]
        return error, esn
