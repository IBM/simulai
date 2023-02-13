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

from ._affine import AffineMapping
from ._elm import ELM

# No back-propagation mechanisms
from ._esn import DeepEchoStateNetwork, EchoStateNetwork, WideEchoStateNetwork
from ._extended_opinf import ExtendedOpInf
from ._opinf import OpInf
from ._pinv import CompressedPinv

# Back-propagation mechanisms
assert engine is not None, "The variable engine was not defined."

if engine == "pytorch":
    from ._pytorch._conv import ConvolutionalNetwork, ResConvolutionalNetwork
    from ._pytorch._dense import (
        SLFNN,
        ConvexDenseNetwork,
        DenseNetwork,
        Linear,
        ResDenseNetwork,
        ShallowNetwork,
    )
    from ._pytorch._koopman import AutoEncoderKoopman, KoopmanNetwork
    from ._pytorch._numpy import LinearNumpy
    from ._pytorch._opinf import OpInfNetwork
    from ._pytorch._rbf import ModalRBFNetwork, RBFLayer, RBFNetwork

elif engine == "numpy":
    pass
else:
    raise Exception(f"Engine {engine} is not available.")
