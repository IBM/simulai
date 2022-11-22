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
from scipy import interpolate
import sklearn.gaussian_process as gp

class GaussianRandomFields:

    def __init__(self, x_interval=(0, 1), kernel='RBF',
                       length_scale=1, N=None, interp='cubic'):

        self.x_interval = x_interval
        self.length_scale = length_scale
        self.N = N
        self.interp = interp
        self.tol = 1e-13

        self.x = np.linspace(*self.x_interval, self.N)[:, None]

        kernels_function = getattr(gp.kernels, kernel, None)
        assert kernels_function is not None,\
            f"The kernel {kernel} is not in sklearn.gaussian_process.kernels"

        kernels = kernels_function(length_scale=self.length_scale)
        self.kernels = kernels(self.x)

        self.space = np.linalg.cholesky(self.kernels + self.tol*np.eye(self.N))


    def random_u(self, n_features=None):

        u_ = np.random.randn(self.N, n_features)
        return np.dot(self.space, u_).T

    def generate_u(self, features, sensors):

        values = map(lambda y: interpolate.interp1d(
                               np.ravel(self.x), y, kind=self.interp,
                             copy=False, assume_sorted=True
                             )(sensors)[:, None], features)


        return np.hstack(list(values))




