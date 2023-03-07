from unittest import TestCase

import numpy as np

from simulai.normalization import UnitaryNormalization, UnitarySymmetricalNormalization


class TestNormalization(TestCase):
    def test_unitary(self):
        factor = 20
        data = factor * np.random.rand(1_000, 100)
        data_ext = factor * np.random.rand(1_00, 100)

        norm = UnitaryNormalization()

        # Unitary normalization
        # Testing the normalization of a plain array
        data_norm_dict = norm.rescale(map_dict={"data": data})
        data_norm = data_norm_dict["data"]

        assert data_norm.max() == 1.0
        assert data_norm.min() == 0.0

        # De-normalization is good enough ?
        data_denorm_dict = norm.apply_descaling(map_dict={"data": data_norm})

        assert np.isclose(data, data_denorm_dict["data"], atol=1e-16).min() == True

        # When not seen data is normalized, there are interval compression
        data_ext_norm_dict = norm.apply_rescaling(map_dict={"data": data_ext})
        data_ext_norm = data_ext_norm_dict["data"]

        assert (data_ext_norm.max() - data_ext_norm.min()) / (
            data_ext.max() - data_ext.min()
        ) < 1

        # Updating global parameters
        norm.update_global_parameters(data=data_ext)

        data_norm_dict = norm.rescale(map_dict={"data": np.vstack([data, data_ext])})
        data_norm = data_norm_dict["data"]

        assert data_norm.max() == 1.0
        assert data_norm.min() == 0.0

    def test_simmetric_unitary(self):
        factor = 20
        data = factor * np.random.rand(1_000, 100)
        data_ext = factor * np.random.rand(1_00, 100)

        norm = UnitarySymmetricalNormalization()

        # Unitary normalization
        # Testing the normalization of a plain array
        data_norm_dict = norm.rescale(map_dict={"data": data})
        data_norm = data_norm_dict["data"]

        assert data_norm.max() == 1.0
        assert data_norm.min() == -1.0
