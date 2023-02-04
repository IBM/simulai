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

from simulai import engine

assert engine is not None, "The variable engine was not defined."

from ._builtin import SpaRSA

if engine == "pytorch":
    from ._builtin_pytorch import BBI
    from ._losses import *
    from ._optimization import Optimizer, ScipyInterface
elif engine == "numpy":
    pass
else:
    raise Exception(f"Engine {engine} is not available.")
