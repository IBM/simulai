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

# No back-propagation mechanisms
from ._esn import EchoStateNetwork, DeepEchoStateNetwork, WideEchoStateNetwork
from ._elm import ELM
from ._affine import AffineMapping
from ._opinf import OpInf
from ._koopman import KoopmanOperator
from ._pinv import CompressedPinv

# Back-propagation mechanisms
assert engine is not None, "The variable engine was not defined."

if engine == "pytorch":

    from .pytorch._dense import DenseNetwork, ResDenseNetwork, ConvexDenseNetwork, Linear, SLFNN, ShallowNetwork
    from .pytorch._opinf import OpInfNetwork
    from .pytorch._koopman import KoopmanNetwork, AutoEncoderKoopman
    from .pytorch._rbf import RBFLayer, RBFNetwork, ModalRBFNetwork
    from .pytorch._numpy import LinearNumpy
    from .pytorch._conv import ConvolutionalNetwork, ResConvolutionalNetwork

elif engine == "numpy":

    pass
else:

    raise Exception(f"Engine {engine} is not available.")
