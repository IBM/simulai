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

# Set of abstract classes to be used as identifiers

class Transformation(object):

    def __init__(self):
        pass

    def transform(self, data=None):
        return data

    def transform_inv(self, data=None):
        return data

class Regression(object):

    def __init__(self):
        pass

class Model(object):

    def __init__(self):
        pass

class Dataset(object):

    def __init__(self):
        pass

    def __call__(self):
        pass

class Normalization(object):

    def __init__(self):
        pass

# Parent class for the DataPreparer classes
class DataPreparer(object):

    """
    DataPreparer classes are used to convert raw data to a proper format
    in order to be processed via others algorithms, as ROMs and ML algorithms
    """
    def __init__(self) -> None:

        self.purpose = "data_preparer"

    # Following, we have the default methods of the
    # DataPreparer classes

    def prepare_input_data(self, data):
        pass

    def prepare_output_data(self, data):
        pass

    def prepare_input_structured_data(self, data):
        pass

    def prepare_output_structured_data(self, data):
        pass

class ROM(Transformation):

    def __init__(self) -> None:

        self.purpose = 'rom'

        self.kind = 'global'

        super().__init__()

    def project(self, data:np.ndarray=None) -> np.ndarray:
        return data

    def reconstruct(self, data:np.ndarray=None) -> np.ndarray:
        return data

    def transform(self, data:np.ndarray=None, **kwargs) -> np.ndarray:
        return self.project(data=data, **kwargs)

    def transform_inv(self, data:np.ndarray=None) -> np.ndarray:
        return self.reconstruct(data=data)

    def fit(self, data:np.ndarray=None):
        """

        :param data:
        :type data: np.ndarray
        :return:

        """
        pass

class BaseFramework(object):

    def __init__(self):

        pass

    def fit(self):
        pass

    def eval(self):
        pass

    def eval_batch(self):
        pass

    def test(self):
        pass

    def save(self):
        pass

# This is like an identifier
class Integral(object):

    """
    Parent class for the integration classes
    """
    purpose = 'integration'

    def __init__(self):
        pass

# This is like an identifier
class ByPassIntegrator(Integral):

    """
    It does nothing
    """
    name = "no_post_process_op"

    def __init__(self):
        super().__init__()

    def __call__(self, data=None):
        return data
