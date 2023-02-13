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

from collections import OrderedDict
from typing import Union  # TODO: It will be modified in Python 3.10

import numpy as np
from numpy import ndarray

from simulai.abstract import Regression
from simulai.templates import NetworkTemplate

#  and replaced by |


# This is a wrapper for extrapolation step by step using a pre-existent model
class StepwiseExtrapolation:
    def __init__(
        self,
        model: Union[NetworkTemplate, Regression, list, dict] = None,
        keys: list = None,
    ) -> None:
        # Assertion error messages
        self.model_isnt_callable = (
            "In case of model be a list all their elements must be callable."
        )
        self.model_hasnt_step = "The instance model must have the method step."
        self.keys_arent_necessary = (
            "It the input is a dictionary, it is not nessesary to provide keys."
        )
        self.lengths_must_be_equal = "The lengths must be equal for keys and models."
        self.array_must_be_two_dim = (
            "The array must have two dimensions but received {}"
        )
        self.keys = keys

        if isinstance(model, NetworkTemplate) or isinstance(model, Regression):
            assert hasattr(model, "step"), self.model_hasnt_step
            model_ = [model]
            assert len(keys) == len(model_), self.lengths_must_be_equal

            self.models_instances = OrderedDict(
                {key: item for key, item in zip(keys, model_)}
            )

        elif type(model) == list:
            self._check_if_callable_and_has_step(model=model)
            assert len(keys) == len(model), self.lengths_must_be_equal

            self.models_instances = OrderedDict(
                {key: item for key, item in zip(keys, model)}
            )

        elif type(model) == dict:
            self._check_if_callable_and_has_step(model=model.values())

            if keys != None:
                print(self.keys_arent_necessary)
                print("Using the keys already provided with the model dict.")

            # Now, self.model can be model, a list of at least one model
            self.model_instances = model

        else:
            raise Exception(
                f"The attribute model must be callable or list(callable), but received {type(model)}"
            )

    def _check_if_callable_and_has_step(
        self, model: Union[list, dict.values] = None
    ) -> None:
        assert all(
            [
                (
                    (isinstance(item, NetworkTemplate) or isinstance(item, Regression))
                    and hasattr(model, "step")
                )
                for item in model
            ]
        ), self.model_isnt_callable

    def _check_if_is_two_dimensional(self, array: ndarray = None) -> None:
        assert len(array.shape) == 2, self.array_must_be_two_dim.format(array.shape)

    def _serial_predict_with_auxiliary(
        self,
        initial_state: ndarray = None,
        auxiliary_data: ndarray = None,
        horizon: int = None,
    ) -> ndarray:
        current_state = np.hstack([initial_state, auxiliary_data[0:1]])
        extrapolation_list = list()

        # Time extrapolation loop
        for step in range(horizon):
            step_outputs_list = list()

            # Serial dispatcher
            for model_id, model in self.models_instances.items():
                out = model.step(data=current_state[0])
                step_outputs_list.append(out)

            current_output = np.hstack(step_outputs_list)[None, :]
            current_state = np.hstack([current_output, auxiliary_data[step : step + 1]])

            extrapolation_list.append(current_output)

        return np.vstack(extrapolation_list)

    def predict(
        self,
        initial_state: ndarray = None,
        auxiliary_data: ndarray = None,
        horizon: int = None,
        parallel: str = None,
    ):
        self._check_if_is_two_dimensional(array=initial_state)
        self._check_if_is_two_dimensional(auxiliary_data)

        if auxiliary_data is not None and parallel is None:
            self.extrapolator = self._serial_predict_with_auxiliary

        return self.extrapolator(
            initial_state=initial_state, auxiliary_data=auxiliary_data, horizon=horizon
        )
