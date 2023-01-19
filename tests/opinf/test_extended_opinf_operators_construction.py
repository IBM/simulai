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

from simulai.regression import ExtendedOpInf
from simulai.metrics import L2Norm
from simulai.file import load_pkl

# Testing the correctness of the OpInf operators construction
class TestOperatorsConstruction(TestCase):

    def setUp(self) -> None:
        pass

    def test_operators_construction(self):

        n_samples = 1000
        n_vars = 5

        data_input = np.random.rand(n_samples, n_vars)
        data_output = np.random.rand(n_samples, n_vars)

        data_input[:, 0] = 1

        batch_sizes = [50, 100, 1000]
        lambda_linear = 1
        lambda_quadratic = 1

        operator_config = {'bias_rescale': 1, 'solver': 'lstsq'}

        observables = [['x'],
                       ['x', 'sin(4*x)', 'cos(4*x)'],
                       ['x', 'x**2', 'tanh(x)'],
                       ['x', 'Kronecker(x)'],
                       ['x', 'tanh(sin(Kronecker(x)))']]

        for ob in observables:

            D_o_list = list()
            R_matrix_list = list()

            for batch_size in batch_sizes:

                # Instantiating OpInf
                model =  ExtendedOpInf(observables=ob, operator_config=operator_config)

                # Training
                model.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
                model.fit(input_data=data_input, target_data=data_output, batch_size=batch_size, continuing=False)

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




