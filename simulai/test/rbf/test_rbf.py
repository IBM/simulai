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

from unittest import TestCase
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['engine'] = 'pytorch'

from simulai.regression import RBFLayer
from simulai.metrics import L2Norm

# Testing the correctness of the OpInf operators construction
class TestRBFProjection(TestCase):

    def setUp(self) -> None:
        pass

    def test_rbf_projection(self):

        time = np.linspace(0, 1, 1_000)

        A = np.random.rand(20, 100)
        B = np.random.rand(100)
        B = 2*(B - B.min(0))/(B.max(0) - B.min(0)) - 1

        omega = 10*np.pi
        data = np.sin(B*omega*time[:, None]) @ A.T

        n_latent = 200
        n_outputs = 20
        delta_t = 1

        config = {
            'xmin': 0,
            'xmax': delta_t,
            'Nk': n_latent,
            'var_dim': n_outputs,
            'Sigma': 0.0001,
            'name': 'trunk_net'
        }

        net = RBFLayer(**config)
        basis = net.eval(input_data=time[:, None])

        w = np.linalg.pinv(basis) @ data

        l2_norm = L2Norm()

        error = l2_norm(data=basis @ w, reference_data=data, relative_norm=True)
        print(f'{100*error} %')



