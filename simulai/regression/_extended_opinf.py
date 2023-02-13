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

from typing import List, Union

import numpy as np

from simulai.math.expressions import FromSymbol2FLambda

from ._opinf import OpInf


# Wrapper class for constructing Koopman operators
class ExtendedOpInf(OpInf):
    def __init__(
        self,
        observables: List[str] = None,
        fobservables: List[str] = None,
        variables: List[str] = None,
        intervals: List[Union[int, list]] = None,
        fintervals: List[Union[int, list]] = None,
        operator_config: dict = None,
    ) -> None:
        super().__init__(**operator_config)

        if intervals is None:
            intervals = len(observables) * [-1]

        assert len(intervals) == len(
            observables
        ), "The number of intervals must be the same number of observables"

        # Intervals for indicating the modes in which each observable must be applied
        self.intervals = [self._construct_intervals(ii) for ii in intervals]

        if fintervals is not None:
            self.fintervals = [self._construct_intervals(ii) for ii in fintervals]
        else:
            self.fintervals = None

        if variables is not None:
            self.variables = variables
        else:
            self.variables = ["x"]

        self.observables_expressions = observables
        self.fobservables_expressions = fobservables

        self.built_observables = False

        self._build_observables_spaces(
            observables=self.observables_expressions,
            fobservables=self.fobservables_expressions,
        )

        self.black_list = ["observables", "fobservables", "funcgen", "all_observables"]

    def _build_observables_spaces(
        self, observables: List[str] = None, fobservables: List[str] = None
    ) -> None:
        # Generating observables lambda functions from string expressions
        # Some of this objects are not pickleable, so they must re-generated when necessary

        self.funcgen = FromSymbol2FLambda(variables=self.variables)

        self.observables = [
            self.funcgen.convert(ob) for ob in observables
        ]  # Observables for the field variables

        self.fobservables = fobservables  # Observables for the forcing variables

        if self.fobservables is not None:
            self.fobservables = [self.funcgen.convert(ob) for ob in self.fobservables]
            self.all_observables = self.observables + self.fobservables
        else:
            self.all_observables = self.observables

        # Checking up if all the observables are callable functions
        assert all(
            [callable(ob) for ob in self.all_observables]
        ), f"All the observable must be callable, but received {self.all_observables}"

        self.built_observables = True

    def _construct_intervals(self, interval: Union[int, list] = None) -> slice:
        if interval == -1:
            return slice(0, None)
        elif type(interval) and len(interval) == 2:
            return slice(*interval)
        elif type(interval) and len(interval) == 1:
            return slice(interval[0], None)
        else:
            raise Exception(
                f"It is expected 'interval' to be int or list, but received {type(interval)}."
            )

    # Generating the observables data using a properly defined transformation
    def _generate_observables(
        self, data: np.ndarray = None, observables: list = None
    ) -> np.ndarray:
        # The object data is guaranteed as a 2D matrix
        return np.hstack(
            [ob(data[:, self.intervals[oi]]) for oi, ob in enumerate(observables)]
        )

    def fit(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
        batch_size: int = None,
        **kwargs,
    ) -> None:
        if self.built_observables is False:
            self._build_observables_spaces(
                observables=self.observables_expressions,
                fobservables=self.fobservables_expressions,
            )

        # Conditional for dealing with forcing information
        if self.fobservables is not None and forcing_data is None:
            raise Exception(
                "There are forcing observables defined, but there is no forcing data provided."
            )
        elif self.fobservables is None and forcing_data is not None:
            raise Exception(
                "Forcing data was provided, but the forcing observables were not defined."
            )
        elif self.fobservables is not None and forcing_data is not None:
            forcing_observables = self._generate_observables(
                data=forcing_data, observables=self.fobservables
            )
        else:
            forcing_observables = None

        input_observables = self._generate_observables(
            data=input_data, observables=self.observables
        )

        super().fit(
            input_data=input_observables,
            target_data=target_data,
            forcing_data=forcing_observables,
            batch_size=batch_size,
            **kwargs,
        )

        self.funcgen.clean_engines()

    def _builtin_jacobian(self, x):
        if len(x.shape) == 1:
            x = x[None, :]

        x_observables = self._generate_observables(data=x, observables=self.observables)

        return self.A_hat + (self.K_op @ x_observables[0].T)

    def eval(self, input_data: np.ndarray = None, **kwargs) -> np.ndarray:
        input_observables = self._generate_observables(
            data=input_data, observables=self.observables
        )

        return super().eval(input_data=input_observables, **kwargs)

    def save(self, save_path: str = None, model_name: str = None) -> None:
        for item in self.black_list:
            setattr(self, item, None)

        self.built_observables = False

        super().save(save_path=save_path, model_name=model_name)

    def lean_save(self, save_path: str = None, model_name: str = None) -> None:
        for item in self.black_list:
            setattr(self, item, None)

        self.built_observables = False

        super().lean_save(save_path=save_path, model_name=model_name)
