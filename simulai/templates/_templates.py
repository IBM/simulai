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

import scipy.sparse as sparse

from simulai.abstract import Regression

class ReservoirComputing(Regression):

    def __init__(self, reservoir_dim=None, sparsity_level=None):

        super().__init__()

        self.sparsity_tolerance = 0.0025  # Default choice

        self.reservoir_dim = reservoir_dim

        self.sparsity_level = sparsity_level

    @property
    def _reservoir_dim_corrected_sparsity_level(self):
        # Guaranteeing a minimum sparsity tolerance
        dim = max(self.sparsity_level, self.sparsity_tolerance) if self.reservoir_dim == 0 else self.reservoir_dim
        effective_sparsity = self.sparsity_level / dim
        if effective_sparsity < self.sparsity_tolerance:
            return self.sparsity_tolerance
        else:
            return self.sparsity_level / dim

    # It creates a sparse and randomly distributed reservoir matrix
    def create_reservoir(self, reservoir_dim=None):
        return sparse.rand(self.reservoir_dim, self.reservoir_dim,
                           density=self._reservoir_dim_corrected_sparsity_level)

