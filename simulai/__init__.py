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

# (C) Copyright IBM Corporation 2017, 2018, 2019
# U.S. Government Users Restricted Rights:  Use, duplication or disclosure restricted
# by GSA ADP Schedule Contract with IBM Corp.
#
# Author: Joao Lucas S. Almeida <joao.lucas.sousa.almeida@ibm.com>

import os
import numpy as np
import torch 

engine_var = os.environ.get("engine")

if engine_var is not None:
    engine = engine_var
else:
    engine = "pytorch"

__version__ = "1.3"

# Determining global dtype for data structures
TENSOR_DTYPE = torch.get_default_dtype()

if TENSOR_DTYPE == torch.float32:
    ARRAY_DTYPE = np.float32
elif TENSOR_DTYPE == torch.float64:
    ARRAY_DTYPE = np.float64
else:
    ARRAY_DTYPE = np.float64

