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

from simulai.metrics import FeatureWiseErrorNorm


def compute_datasets_to_reference_norm(
    data,
    reference_data,
    ords=None,
    norm_type=None,
    data_interval=None,
    reference_data_interval=None,
    names=None,
    batch_size=1,
):
    if ords is None:
        ords = [1, 2, np.inf]
    if norm_type is None:
        norm_type = ["absolute", "relative"]
    if names is None:
        names = list(data.dtype.names)

    projection_error = {
        n: {nt: {str(o): [] for o in ords} for nt in norm_type} for n in names
    }

    for nt in norm_type:
        for o in ords:
            error = FeatureWiseErrorNorm()(
                data=data,
                reference_data=reference_data,
                relative_norm=nt == "relative",
                data_interval=data_interval,
                reference_data_interval=reference_data_interval,
                key=names,
                batch_size=batch_size,
                ord=o,
            )
            for n in names:
                projection_error[n][nt][str(o)] = np.array(error[n]).tolist()

    return projection_error
