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

from ._templates import (
    NetworkInstanceGen,
    ReservoirComputing,
    autoencoder_auto,
    cnn_autoencoder_auto,
    mlp_autoencoder_auto,
)

if engine == "pytorch":
    from ._pytorch_network import (
        ConvNetworkTemplate,
        HyperTrainTemplate,
        NetworkTemplate,
        as_array,
        as_tensor,
        channels_dim,
        guarantee_device,
    )
elif engine == "numpy":
    pass
else:
    pass
