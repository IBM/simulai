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

from typing import List, Optional, Union

import numpy as np

# Operator inference using quadratic non-linearity
class RandoNet:
    def __init__(
        self,
        N:int=None,
        d:int=None
    ) -> None:
        
        """Operator Inference (OpInf)
        """
        self.R = self._johnson_lierstraussn(N, d)

    def _johnson_lierstrauss(self, N, d):

        return np.random.randn(N, d)/np.sqrt(N)

    def poject_JL(self, x):

        return self.R @ x
        
        
