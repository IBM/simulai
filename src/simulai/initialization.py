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
import numpy as np
import tensorflow as tf
from typing import Union

# This class is used within a TensorFlow framework
class XavierInitialization:

    def __init__(self) -> None:

        pass

    @staticmethod
    def init(size:int=None, index:int=None, model_id:str=None) -> tf.Variable:
        """
        Based on https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_identification%20(Navier-Stokes)/NavierStokes.py

        :param model_id: str
        :param size:
        :type size: List
        :param index:
        :type index: int or str
        :return:
        :rtype: tf.Variable
        """
        assert isinstance(size, list), "Error! size is not a list"
        assert len(size) == 2, "Error! Invalid parameter size: len={}".format(len(size))

        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(6 / (in_dim + out_dim))

        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                                      stddev=xavier_stddev),
                           dtype=tf.float32,
                           name='weights_{}_{}'.format(index, model_id))

# This class is used within a TensorFlow framework
class KaimingInitialization:

    def __init__(self) -> None:
        pass

    @staticmethod
    def init(size:int=None, index:int=None, model_id:str=None) -> tf.Variable:
        """
        :param model_id: str
        :param size:
        :type size: List
        :param index:
        :type index: int
        :return:
        :rtype: tf.Variable
        """
        assert isinstance(size, list), "Error! size is not a list"
        assert len(size) == 2, "Error! Invalid parameter size: len={}".format(len(size))

        in_dim = size[0]
        out_dim = size[1]
        kaiming_stddev = np.sqrt(2 /in_dim)

        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                                      stddev=kaiming_stddev),
                           dtype=tf.float32,
                           name='weights_{}_{}'.format(index, model_id))

# This class is used within a TensorFlow framework
class UniformInitialization:

    def __init__(self):
        pass

    @staticmethod
    def init(size:int=None, index:int=None, model_id:str=None) -> tf.Variable:
        """
        :param model_id: str
        :param size:
        :type size: List
        :param index:
        :type index: int
        :return:
        :rtype: tf.Variable
        """
        assert isinstance(size, list), "Error! size is not a list"
        assert len(size) == 2, "Error! Invalid parameter size: len={}".format(len(size))

        in_dim = size[0]
        out_dim = size[1]

        return tf.Variable(tf.random.uniform([in_dim, out_dim], minval=-2.0, maxval=2.0),
                           dtype=tf.float32,
                           name='weights_{}_{}'.format(index, model_id))


class FilePersistence:

    def __init__(self, path:str=None, name:str=None, default_initializer:callable=None) -> None:

        self.arrays_list = list()

        if default_initializer:
            self.initialization_op = default_initializer.init
        else:
            self.initialization_op = self._init

        if path and name:

            self.path = path
            self.name = name
            self.file_path = path + name + '.npy'

            if os.path.isfile(self.file_path):
                self.arrays_list = np.load(self.file_path, allow_pickle=True).tolist()
            else:
                pass
        else:
            pass

    def _init(self, size:int=None, index:int=None, model_id:str=None) -> tf.Variable:

        print('Restoring previously initialized array.')

        assert isinstance(size, list), "Error! size is not a list"
        assert len(size) == 2, "Error! Invalid parameter size: len={}".format(len(size))

        current_array = self.arrays_list[index]

        assert size == list(current_array.shape)

        return tf.Variable(current_array,
                           dtype=tf.float32,
                           name='weights_{}_{}'.format(index, model_id))

    def _dump(self, objects_list:Union[list, np.ndarray, None]=None) -> None:

        np.save(self.file_path, objects_list)

    def init(self, size:int=None, index:int=None, model_id:str=None):
        """
        :param model_id: str
        :param size:
        :type size: List
        :param index:
        :type index: int
        :return:
        :rtype: tf.Variable
        """

        current_object = self.initialization_op(size=size, index=index, model_id=model_id)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            current_array = sess.run(current_object)

        self.arrays_list.append(current_array)

        self._dump(objects_list=self.arrays_list)

        return current_object
