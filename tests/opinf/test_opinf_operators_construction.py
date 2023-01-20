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
import sys
from unittest import TestCase
import numpy as np

from simulai.regression import OpInf
from simulai.metrics import L2Norm
from simulai.file import load_pkl

# Testing the correctness of the OpInf operators construction
class TestOperatorsConstruction(TestCase):

    def setUp(self) -> None:
        pass

    def test_operators_construction_pinv(self):

        n_samples = 10_000
        BATCH_SIZES = [None, 1000]
        n_vars = 5
        FORCE_LAZY=[False, True]

        for force_lazy, batch_size in zip(FORCE_LAZY, BATCH_SIZES):

            data_input = np.random.rand(n_samples, n_vars)
            data_output = np.random.rand(n_samples, n_vars)

            # Instantiating OpInf
            model = OpInf(bias_rescale=1, solver='pinv')

            # Training
            model.fit(input_data=data_input, target_data=data_output, batch_size=batch_size,
                      force_lazy_access=force_lazy)

            assert isinstance(model.A_hat, np.ndarray)
            assert isinstance(model.H_hat, np.ndarray)
            assert isinstance(model.c_hat, np.ndarray)

    def test_operators_jacobian(self):

        n_samples = 1000
        n_vars = 5

        data_input = np.random.rand(n_samples, n_vars)
        data_output = np.random.rand(n_samples, n_vars)

        # Instantiating OpInf
        model = OpInf(bias_rescale=1, solver='pinv')

        # Training
        model.fit(input_data=data_input, target_data=data_output)

        model.construct_K_op()

        assert isinstance(model.K_op, np.ndarray)

    def test_operators_construction_lstsq(self):

        n_samples = 1000
        n_vars = 5

        data_input = np.random.rand(n_samples, n_vars)
        data_output = np.random.rand(n_samples, n_vars)

        data_input[:, 0] = 1

        batch_sizes = [50, 100, 1000]
        lambda_linear = 1
        lambda_quadratic = 1

        D_o_list = list()
        R_matrix_list = list()

        for batch_size in batch_sizes:

            # Instantiating OpInf
            model = OpInf(bias_rescale=1, solver='lstsq')

            # Training
            model.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
            model.fit(input_data=data_input, target_data=data_output, batch_size=batch_size)

            assert isinstance(model.A_hat, np.ndarray)
            assert isinstance(model.H_hat, np.ndarray)
            assert isinstance(model.c_hat, np.ndarray)

            D_o = model.D_o
            R_matrix = model.R_matrix

            D_o_list.append(D_o)
            R_matrix_list.append(R_matrix)

        ref_D_o = D_o_list.pop(0)
        ref_R_matrix = R_matrix_list.pop(0)

        l2_norm = L2Norm()

        for ii, (d_o, r_matrix) in enumerate(zip(D_o_list, R_matrix_list)):
            assert np.all(np.isclose(ref_D_o, d_o).astype(int)), "The case with batch_size" \
                                                                 f" {batch_sizes[ii]} is divergent for the matrix D_o"

            assert np.all(np.isclose(ref_R_matrix, r_matrix).astype(int)), "The case with batch_size" \
                                                                           f" {batch_sizes[ii]} is divergent for" \
                                                                           " the matrix R_matrix."

            error_d_o = l2_norm(data=d_o, reference_data=ref_D_o, relative_norm=True)
            error_r_matrix = l2_norm(data=r_matrix, reference_data=ref_R_matrix, relative_norm=True)

            maximum_deviation_d_o = np.abs(d_o - ref_D_o).max()
            maximum_deviation_r_matrix = np.abs(r_matrix - ref_R_matrix).max()

            print(f"Maximum D_o error: {100 * error_d_o} %.")
            print(f"Maximum R_matrix error: {100 * error_r_matrix} %.")

            print(f"Maximum D_o deviation: {maximum_deviation_d_o}.")
            print(f"Maximum R_matrix deviation: {maximum_deviation_r_matrix}.")

    def test_operators_construction_forcing_linear(self):

        n_samples = 1000
        n_vars = 5
        n_vars_forcing = 5

        data_input = np.random.rand(n_samples, n_vars)
        data_output = np.random.rand(n_samples, n_vars)
        data_forcing = np.random.rand(n_samples, n_vars_forcing)

        data_input[:, 0] = 1

        batch_sizes = [10, 100, 1000]
        lambda_linear = 1
        lambda_quadratic = 1

        D_o_list = list()
        R_matrix_list = list()

        forcing_case = 'linear'

        for batch_size in batch_sizes:

            # Instantiating OpInf
            model = OpInf(bias_rescale=1, solver='lstsq', forcing=forcing_case)

            # Training
            model.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
            model.fit(input_data=data_input, target_data=data_output, forcing_data=data_forcing,
                      batch_size=batch_size)

            # Basic checks
            assert isinstance(model.B_hat, np.ndarray)
            assert model.B_hat.shape == (n_vars, n_vars_forcing)

            if forcing_case == 'nonlinear':

                assert model.H_hat.shape == (n_vars, int((n_vars_forcing + n_vars)*(n_vars_forcing + n_vars + 1)/2))

            D_o = model.D_o
            R_matrix = model.R_matrix

            D_o_list.append(D_o)
            R_matrix_list.append(R_matrix)

        ref_D_o = D_o_list.pop(0)
        ref_R_matrix = R_matrix_list.pop(0)

        l2_norm = L2Norm()

        for ii, (d_o, r_matrix) in enumerate(zip(D_o_list, R_matrix_list)):

            assert np.all(np.isclose(ref_D_o, d_o).astype(int)), "The case with batch_size" \
                                                                 f" {batch_sizes[ii]} is divergent for the matrix D_o"

            assert np.all(np.isclose(ref_R_matrix, r_matrix).astype(int)), "The case with batch_size" \
                                                                           f" {batch_sizes[ii]} is divergent for" \
                                                                           " the matrix R_matrix."

            error_d_o = l2_norm(data=d_o, reference_data=ref_D_o, relative_norm=True)
            error_r_matrix = l2_norm(data=r_matrix, reference_data=ref_R_matrix, relative_norm=True)

            maximum_deviation_d_o = np.abs(d_o - ref_D_o).max()
            maximum_deviation_r_matrix = np.abs(r_matrix - ref_R_matrix).max()

            print(f"Maximum D_o error: {100 * error_d_o} %.")
            print(f"Maximum R_matrix error: {100 * error_r_matrix} %.")

            print(f"Maximum D_o deviation: {maximum_deviation_d_o}.")
            print(f"Maximum R_matrix deviation: {maximum_deviation_r_matrix}.")

    def test_operators_construction_forcing_nonlinear(self):

        n_samples = 100
        n_vars = 5
        n_vars_forcing = 5

        data_input = np.random.rand(n_samples, n_vars)
        data_output = np.random.rand(n_samples, n_vars)
        data_forcing = np.random.rand(n_samples, n_vars_forcing)

        data_input[:, 0] = 1

        batch_sizes = [1, 10, 100]
        lambda_linear = 1
        lambda_quadratic = 1

        D_o_list = list()
        R_matrix_list = list()

        forcing_case = 'nonlinear'

        for batch_size in batch_sizes:

            # Instantiating OpInf
            model = OpInf(bias_rescale=1, solver='lstsq', forcing=forcing_case)

            # Training
            model.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
            model.fit(input_data=data_input, target_data=data_output, forcing_data=data_forcing,
                      batch_size=batch_size)

            # Basic checks
            assert isinstance(model.B_hat, np.ndarray)
            assert model.B_hat.shape == (n_vars, n_vars_forcing)
            assert model.H_hat.shape == (n_vars, int((n_vars_forcing + n_vars) * (n_vars_forcing + n_vars + 1) / 2))

            D_o = model.D_o
            R_matrix = model.R_matrix

            D_o_list.append(D_o)
            R_matrix_list.append(R_matrix)

        ref_D_o = D_o_list.pop(0)
        ref_R_matrix = R_matrix_list.pop(0)

        l2_norm = L2Norm()

        for ii, (d_o, r_matrix) in enumerate(zip(D_o_list, R_matrix_list)):
            assert np.all(np.isclose(ref_D_o, d_o).astype(int)), "The case with batch_size" \
                                                                 f" {batch_sizes[ii]} is divergent for the matrix D_o"

            assert np.all(np.isclose(ref_R_matrix, r_matrix).astype(int)), "The case with batch_size" \
                                                                           f" {batch_sizes[ii]} is divergent for" \
                                                                           " the matrix R_matrix."

            error_d_o = l2_norm(data=d_o, reference_data=ref_D_o, relative_norm=True)
            error_r_matrix = l2_norm(data=r_matrix, reference_data=ref_R_matrix, relative_norm=True)

            maximum_deviation_d_o = np.abs(d_o - ref_D_o).max()
            maximum_deviation_r_matrix = np.abs(r_matrix - ref_R_matrix).max()

            print(f"Maximum D_o error: {100 * error_d_o} %.")
            print(f"Maximum R_matrix error: {100 * error_r_matrix} %.")

            print(f"Maximum D_o deviation: {maximum_deviation_d_o}.")
            print(f"Maximum R_matrix deviation: {maximum_deviation_r_matrix}.")

    def test_operators_construction_multipliers(self):

        n_samples = 1000
        n_vars = 50
        multipliers = np.array([10**(i/20) for i in range(n_vars)])

        data_input = multipliers*np.random.rand(n_samples, n_vars)
        data_output = multipliers*np.random.rand(n_samples, n_vars)

        data_input[:,0] = 1

        batch_sizes = [50, 100, 1000]
        lambda_linear = 1
        lambda_quadratic = 1

        D_o_list = list()
        R_matrix_list = list()

        for batch_size in batch_sizes:

            # Instantiating OpInf
            model = OpInf(bias_rescale=1, solver='lstsq')

            # Training
            model.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
            model.fit(input_data=data_input, target_data=data_output, batch_size=batch_size)

            D_o = model.D_o
            R_matrix = model.R_matrix

            D_o_list.append(D_o)
            R_matrix_list.append(R_matrix)

        ref_D_o = D_o_list.pop(0)
        ref_R_matrix = R_matrix_list.pop(0)

        l2_norm = L2Norm()

        for ii, (d_o, r_matrix) in enumerate(zip(D_o_list, R_matrix_list)):

            assert np.all(np.isclose(ref_D_o, d_o).astype(int)), "The case with batch_size" \
                                                                 f" {batch_sizes[ii]} is divergent for the matrix D_o"


            assert np.all(np.isclose(ref_R_matrix, r_matrix).astype(int)), "The case with batch_size" \
                                                                            f" {batch_sizes[ii]} is divergent for" \
                                                                            " the matrix R_matrix."


            error_d_o = l2_norm(data=d_o, reference_data=ref_D_o, relative_norm=True)
            error_r_matrix = l2_norm(data=r_matrix, reference_data=ref_R_matrix, relative_norm=True)

            maximum_deviation_d_o = np.abs(d_o - ref_D_o).max()
            maximum_deviation_r_matrix = np.abs(r_matrix - ref_R_matrix).max()

            print(f"Maximum D_o error: {100*error_d_o} %.")
            print(f"Maximum R_matrix error: {100 * error_r_matrix} %.")

            print(f"Maximum D_o deviation: {maximum_deviation_d_o}.")
            print(f"Maximum R_matrix deviation: {maximum_deviation_r_matrix}.")

    def test_operators_setting(self):

        opinf = OpInf()

        coefficients = np.array([[-2.08042604e-16,  4.87816759e-16, -3.07462226e-16],
                                 [-1.00000011e+01,  2.79999981e+01, -9.10594479e-07],
                                 [ 1.00000008e+01, -9.99999365e-01,  6.26876120e-07],
                                 [ 1.20917885e-08, -5.26344927e-08, -2.66666616e+00],
                                 [ 1.19725785e-08, -1.89298728e-08,  2.63093188e-07],
                                 [-1.08876107e-08,  4.25828284e-08,  9.99999718e-01],
                                 [ 3.49844320e-08, -9.99999948e-01,  2.50026946e-08],
                                 [ 1.72896719e-09, -2.20334382e-08,  7.11072626e-08],
                                 [-2.71926419e-08, -9.17973564e-09, -1.94086956e-08],
                                 [-1.16400933e-09,  1.83556992e-09, -2.64827246e-08]])

        opinf.set_operators(global_matrix=coefficients)

        n_inputs = coefficients.shape[1]

        input_data = np.random.rand(1_000, n_inputs)

        output_data = opinf.eval(input_data=input_data)

        assert isinstance(output_data, np.ndarray), f"The output of opinf.eval must be a numpy.ndarray," \
                                                    f" but got {type(output_data)}"

    def test_operators_save_and_load(self):

        n_samples = 1000
        n_vars = 5
        multipliers = np.array([10 ** (i / 20) for i in range(n_vars)])

        data_input = multipliers * np.random.rand(n_samples, n_vars)
        data_output = multipliers * np.random.rand(n_samples, n_vars)

        data_input[:, 0] = 1

        batch_size = 100
        lambda_linear = 1
        lambda_quadratic = 1

        model = OpInf(bias_rescale=1, solver='lstsq')

        # Training
        model.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
        model.fit(input_data=data_input, target_data=data_output, batch_size=batch_size)


        input_data = np.random.rand(1_000, n_vars)

        model.save(model_name=f'opinf_{id(model)}', save_path='/tmp')

        opinf_reloaded = load_pkl(f'/tmp/opinf_{id(model)}.pkl')
        output_data = opinf_reloaded.eval(input_data=input_data)

        assert isinstance(opinf_reloaded, OpInf)
        assert isinstance(output_data, np.ndarray)

        model.lean_save(model_name=f'opinf_{id(model)}', save_path='/tmp')

        opinf_reloaded_lean = load_pkl(f'/tmp/opinf_{id(model)}.pkl')
        output_data = opinf_reloaded_lean.eval(input_data=input_data)

        assert isinstance(opinf_reloaded, OpInf)
        assert isinstance(output_data, np.ndarray)

