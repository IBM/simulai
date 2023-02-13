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

from typing import Optional


# Prototype
class DeepOTimeIntegrator:
    def __init__(
        self,
        trunk_net_config: dict = None,
        branch_net_config: dict = None,
        encoder_net_config: Optional[dict] = None,
        decoder_net_config: Optional[dict] = None,
        trunk_architecture: str = "densenetwork",
        branch_architecture: str = "densenetwork",
        encoder_architecture: Optional[str] = None,
        decoder_architecture: Optional[str] = None,
        deeponet_architecture: str = "deeponet",
    ) -> None:
        self.trunk_architecture = trunk_architecture
        self.branch_architecture = branch_architecture

        self.trunk_net_config = trunk_net_config
        self.branch_net_config = branch_net_config

        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture

        self.encoder_net_config = encoder_net_config
        self.decoder_net_config = decoder_net_config

        self.deeponet_architecture = deeponet_architecture

    def fit(self):
        pass

    def predict(self):
        pass
