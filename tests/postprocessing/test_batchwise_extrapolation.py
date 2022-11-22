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

import numpy as np
from unittest import TestCase

from simulai.math.progression import gp
from simulai.io import MovingWindow, BatchwiseExtrapolation

''' Testing to perform batchwise extrapolation.
    This extrapolation is performed with any operator
    F: R^(history_size x n_features) -> R^(horizon_size X n_features')
'''

''' BatchIdentity just outputs the output generated
    by MovingWindow
'''
class BatchIdentity:
    def __init__(self, input_data=None, output_data=None):

        input_list = np.split(input_data, input_data.shape[0], axis=0)
        output_list = np.split(output_data, output_data.shape[0], axis=0)

        self.data_dict = {idata.tostring(): odata
                          for idata, odata in zip(input_list, output_list)}

    def __call__(self, input_data):

       return self.data_dict.get(input_data.tostring())

class TestBatchwiseExtrapolation(TestCase):

    def setUp(self) -> None:
        pass

    ''' The BatchIdentity is used as test operator.
    '''
    def test_without_auxiliary(self) -> None:

        # Constructing data
        t_max = 1
        t_min = 0

        N = 1000
        n_features = 8

        t = np.linspace(t_min, t_max, N)

        omega = np.array(gp(init=np.pi, factor=2, n=n_features))

        T, Omega = np.meshgrid(t, omega, indexing='ij')

        # Generic function U = cos(omega*t)
        U = np.cos(Omega*T)

        history_size = 10
        horizon_size = 1
        skip_size = 1

        moving_window = MovingWindow(history_size=history_size,
                                     horizon_size=horizon_size,
                                     skip_size=skip_size)

        U_input, U_target = moving_window(input_data=U, output_data=U)

        init_state = U_input[0][None, ...]

        op = BatchIdentity(input_data=U_input, output_data=U_target)

        extrapolator = BatchwiseExtrapolation(op=op)

        u_target = U_target.reshape(-1, np.prod(U_target.shape[1:]))

        extrapolated_dataset = extrapolator(init_state=init_state, history_size=history_size,
                                            horizon_size=horizon_size, testing_data_size=N-history_size)

        message = "The extrapolation does not correspond to the expected result.  It is necessary to check" \
                  " simulai.io.BatchwiseExtrapolation."

        assert np.linalg.norm(extrapolated_dataset - u_target) == 0, message

    # TODO: Batchwise test using auxiliary variables

