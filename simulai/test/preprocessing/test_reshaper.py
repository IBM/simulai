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

# (C) Copyright IBM Corporation 2017, 2018, 2019
# U.S. Government Users Restricted Rights:  Use, duplication or disclosure restricted
# by GSA ADP Schedule Contract with IBM Corp.
#
# Author: Leonardo P. Tizzei <ltizzei@br.ibm.com>
from unittest import TestCase
from simulai.io import Reshaper
from simulai.io import ScalerReshaper
import numpy as np
import os
import pytest

from simulai.utilities import make_temp_directory


class TestReshaper(TestCase):

    def setUp(self) -> None:

        # use make_temp_directory() above instead of os.path.join(os.getcwd(), '../Network/data') which might not exist
        # self.save_path = os.path.join(os.getcwd(), '../Network/data')
        pass

    def test_prepare_input_data(self):
        reshaper = Reshaper()
        data = np.random.rand(50, 1, 100, 100)
        collapsible = data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_data(data=data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)

    def test_prepare_input_data_invalid_data(self):
        reshaper = Reshaper()
        data = np.random.rand(1)
        with pytest.raises(AssertionError):
            reshaper.prepare_input_data(data=data)

    def test_prepare_output_data(self):
        reshaper = Reshaper()
        data = np.random.rand(50, 1, 100, 100)
        collapsible = data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_data(data=data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)
        new_data = reshaper.prepare_output_data(reshaped_data)
        self.assertIsInstance(new_data, np.ndarray)
        self.assertEqual(data.shape, new_data.shape)

    def test_prepare_input_structured_data(self):
        reshaper = Reshaper()
        # Constructing data
        N = 100
        Nt = 50
        lambd = 10

        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        t = np.linspace(0, 1, Nt)

        X, T, Y = np.meshgrid(x, t, y)

        # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
        U = np.exp(-lambd * T) * (X ** 2 * np.cos(Y) + X * Y)
        # Time derivative of U
        U_t = -lambd * U
        data = np.core.records.fromarrays([U, U_t], names='U, U_t', formats='f8, f8')[:, None]

        n_batches = data.shape[0]
        # Training data
        train_data = data[:int(n_batches / 2), :, :, :]

        input_data = np.core.records.fromarrays([train_data['U']], names='U', formats='f8')[:, None]
        collapsible = input_data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_structured_data(data=input_data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)

    def test_prepare_output_structured_data(self):

        reshaper = Reshaper()
        # Constructing data
        N = 100
        Nt = 50
        lambd = 10

        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        t = np.linspace(0, 1, Nt)

        X, T, Y = np.meshgrid(x, t, y)

        # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
        U = np.exp(-lambd * T) * (X ** 2 * np.cos(Y) + X * Y)
        # Time derivative of U
        U_t = -lambd * U
        data = np.core.records.fromarrays([U, U_t], names='U, U_t', formats='f8, f8')[:, None]

        n_batches = data.shape[0]
        # Training data
        train_data = data[:int(n_batches / 2), :, :, :]

        input_data = np.core.records.fromarrays([train_data['U']], names='U', formats='f8')
        collapsible = input_data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_structured_data(data=input_data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)
        new_data = reshaper.prepare_output_structured_data(reshaped_data)
        self.assertIsInstance(new_data, np.ndarray)
        self.assertEqual(input_data.shape, new_data.shape)

    def test_prepare_output_structured_data_channels_last(self):

        reshaper = Reshaper(channels_last=True)

        # Constructing data
        N = 100
        Nt = 50
        lambd = 10

        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        t = np.linspace(0, 1, Nt)

        X, T, Y = np.meshgrid(x, t, y)

        # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
        U = np.exp(-lambd * T) * (X ** 2 * np.cos(Y) + X * Y)
        # Time derivative of U
        U_t = -lambd * U
        data = np.core.records.fromarrays([U[..., None], U_t[..., None]], names='U, U_t', formats='f8, f8')

        n_batches = data.shape[0]
        # Training data
        train_data = data[:int(n_batches / 2), ...]

        input_data = train_data
        collapsible = input_data.shape[1:]
        result = np.prod(collapsible)*len(data.dtype.names)
        reshaped_data = reshaper.prepare_input_structured_data(data=input_data)

        self.assertIsInstance(reshaped_data, np.ndarray)

        last_dim = reshaped_data.shape[-1]

        self.assertEqual(result, last_dim)
        new_data = reshaper.prepare_output_structured_data(reshaped_data)

        self.assertIsInstance(new_data, np.ndarray)

        self.assertEqual(input_data.shape, new_data.shape)

        assert np.array_equal(input_data, new_data), "The arrays input_data and new_data must be equal."


class TestScalerReshaper(TestCase):


    def test_prepare_input_data(self):
        reshaper = ScalerReshaper(bias=5, scale=2)
        data = np.ones((50, 1, 100, 100))
        collapsible = data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_data(data=data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)
        self.assertTrue(np.all(reshaped_data == (1-5)/2))

    def test_prepare_input_data_invalid_data(self):

        reshaper = ScalerReshaper()
        data = np.random.rand(1)
        with pytest.raises(AssertionError):
            reshaper.prepare_input_data(data=data)

    def test_prepare_output_data(self):

        reshaper = ScalerReshaper(bias=5, scale=2)
        data = np.ones((50, 1, 100, 100))
        collapsible = data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_data(data=data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)
        new_data = reshaper.prepare_output_data(reshaped_data)
        self.assertIsInstance(new_data, np.ndarray)
        self.assertEqual(data.shape, new_data.shape)
        self.assertTrue(np.all(new_data == 1))

    def test_prepare_input_structured_data(self):

        bias = {'U': 1, 'U_t': 2}
        scale = {'U': 3, 'U_t': 4}
        reshaper = ScalerReshaper(bias=bias, scale=scale)
        # Constructing data
        N = 100
        Nt = 50
        lambd = 10

        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        t = np.linspace(0, 1, Nt)

        X, T, Y = np.meshgrid(x, t, y)

        # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
        U = np.exp(-lambd * T) * (X ** 2 * np.cos(Y) + X * Y)
        # Time derivative of U
        U_t = -lambd * U
        data = np.core.records.fromarrays([U, U_t], names='U, U_t', formats='f8, f8')[:, None]

        n_batches = data.shape[0]
        # Training data
        train_data = data[:int(n_batches / 2), :, :, :]

        input_data = np.core.records.fromarrays([train_data['U']], names='U', formats='f8')[:, None]
        collapsible = input_data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_structured_data(data=input_data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)
        for name, d in zip(input_data.dtype.names, np.split(reshaped_data, len(input_data.dtype.names), axis=1)):
            check = d == (np.reshape(input_data[name], (input_data[name].shape[0], -1)) - bias[name]) / scale[name]
            self.assertTrue(np.all(check))

    def test_prepare_output_structured_data(self):
        bias = {'U': 1, 'U_t': 2}
        scale = {'U': 3, 'U_t': 4}
        reshaper = ScalerReshaper(bias=bias, scale=scale)

        # Constructing data
        N = 100
        Nt = 50
        lambd = 10

        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        t = np.linspace(0, 1, Nt)

        X, T, Y = np.meshgrid(x, t, y)

        # Generic function U = exp(-lambd*t)*(x**2*cos(y) + x*y)
        U = np.exp(-lambd * T) * (X ** 2 * np.cos(Y) + X * Y)
        # Time derivative of U
        U_t = -lambd * U
        data = np.core.records.fromarrays([U, U_t], names='U, U_t', formats='f8, f8')[:, None]

        n_batches = data.shape[0]
        # Training data
        train_data = data[:int(n_batches / 2), :, :, :]

        input_data = np.core.records.fromarrays([train_data['U']], names='U', formats='f8')[:, None]
        collapsible = input_data.shape[2:]
        result = np.prod(collapsible)
        reshaped_data = reshaper.prepare_input_structured_data(data=input_data)
        self.assertIsInstance(reshaped_data, np.ndarray)
        last_dim = reshaped_data.shape[-1]
        self.assertEqual(result, last_dim)
        new_data = reshaper.prepare_output_structured_data(reshaped_data)
        self.assertIsInstance(new_data, np.ndarray)
        self.assertEqual(input_data.shape, new_data.shape)
        for name in input_data.dtype.names:
            self.assertAlmostEqual(np.max(abs(input_data[name] - new_data[name])), 0)

        new_data2 = reshaper.prepare_output_data(reshaped_data)
        self.assertIsInstance(new_data2, np.ndarray)
        self.assertEqual(input_data.shape, new_data2.shape)
        for name in input_data.dtype.names:
            self.assertAlmostEqual(np.max(abs(input_data[name] - new_data2[name])), 0)
