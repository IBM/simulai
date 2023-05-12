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
import torch 

def configure_dtype():

    test_dtype_var = os.environ.get("TEST_DTYPE")

    if test_dtype_var is not None:
        test_dtype = getattr(torch, test_dtype_var)
    else:
        test_dtype = torch.float32

    torch.set_default_dtype(test_dtype)

    print(f"Using dtype {test_dtype} in tests.")

    return torch



