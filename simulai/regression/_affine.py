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

import os
import pickle
from inspect import signature

import numpy as np

"""
    BEGIN Affine mapping
"""


class AffineMapping:

    """
    Echo State Network, a subclass of the Reservoir Computing methods
    """

    def __init__(
        self,
        number_of_inputs=None,
        number_of_outputs=None,
        A=None,
        b=None,
        estimate_linear_transition=False,
        estimate_bias_transition=False,
        input_augmented_reservoir=False,
        model_id=None,
    ):
        """
        :param reservoir_dim: dimension of the reservoir matrix
        :type reservoir_dim: int
        :param sparsity_level: level of sparsity at the matrices initialization
        :type sparsity_level: float
        :param radius:
        :param number_of_inputs: number of variables used as input
        :type number_of_inputs: int
        :param sigma: multiplicative constant for the interval of the input matrix initialization
        :type sigma: float
        :param beta: Ridge regularization parameter
        :type beta: float
        :param kappa:
        :param tau:
        :param activation:
        :param Win_interval:
        :param transformation:
        :param model_id:
        """

        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.input_augmented_reservoir = input_augmented_reservoir

        self.current_state = None
        self.reference_state = None
        self.A = A
        self.b = b
        self.estimate_linear_transition = estimate_linear_transition
        self.estimate_bias_transition = estimate_bias_transition

        self.model_id = model_id

        # Echo state networks does not use history during fitting
        self.depends_on_history = False

        self.size_default = np.array([0]).astype("float64").itemsize

    @property
    def trainable_variables(
        self,
    ):
        return {"A": self.A, "b": self.b}

    def fit(self, input_data=None, target_data=None):
        data_in = input_data.T

        if not self.estimate_bias_transition and not self.estimate_linear_transition:
            # dont estimate linear of bias terms
            pass
        elif self.estimate_bias_transition and not self.estimate_linear_transition:
            print("Estimating the bias")
            if self.A is not None:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data - (self.A @ data_in).T,
                    rcond=None,
                )[0]
            else:
                self.b = np.linalg.lstsq(
                    np.ones((target_data.shape[0], 1), dtype=target_data.dtype),
                    target_data,
                    rcond=None,
                )[0]
            self.b = np.reshape(self.b, (-1,))
        elif not self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition")
            if self.b is not None:
                self.A = np.linalg.lstsq(
                    input_data, target_data - np.reshape(self.b, (1, -1)), rcond=None
                )[0].T
            else:
                self.A = np.linalg.lstsq(input_data, target_data, rcond=None)[0].T
        elif self.estimate_bias_transition and self.estimate_linear_transition:
            print("Estimating the linear transition and bias")
            self.A, self.b = np.hsplit(
                np.linalg.lstsq(
                    np.hstack((input_data, np.ones((target_data.shape[0], 1)))),
                    target_data,
                    rcond=None,
                )[0].T,
                [input_data.shape[1]],
            )
            self.b = np.reshape(self.b, (-1,))
        else:
            raise RuntimeError("Unreachable line of code")

    def step(self, data=None):
        affine_term = 0
        if self.A is not None:
            affine_term += self.A @ data
        if self.b is not None:
            affine_term += self.b

        out = affine_term

        return out

    @property
    def state_dim(self):
        return self.number_of_inputs

    def predict(self, initial_data=None, horizon=None):
        output_data = np.zeros((self.number_of_outputs, horizon))

        data = initial_data

        for tt in range(horizon):
            print("Extrapolating for the timestep {}".format(tt))

            affine_term = 0
            if self.A is not None:
                affine_term += self.A @ data
            if self.b is not None:
                affine_term += self.b

            out = affine_term
            output_data[:, tt] = out
            data = out

        return output_data.T

    def save(self, save_path=None, model_name=None):
        configs = {
            k: getattr(self, k) for k in signature(self.__init__).parameters.keys()
        }

        to_save = {
            "current_state": self.current_state,
            "reference_state": self.reference_state,
            "configs": configs,
        }
        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(to_save, fp, protocol=4)
        except Exception as e:
            print(e, e.args)

    @classmethod
    def restore(cls, model_path, model_name):
        path = os.path.join(model_path, model_name + ".pkl")
        with open(path, "rb") as fp:
            d = pickle.load(fp)

        self = cls(**d["configs"])
        d.pop("configs")
        for k, v in d.items():
            setattr(self, k, v)

        return self


"""
    BEGIN Affine mapping
"""
