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

import copy
import importlib
import os
import pickle
from typing import List, Tuple, Union

import dask.array as da
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD

from simulai.abstract import ROM
from simulai.optimization import SpaRSA


class ByPassROM(ROM):
    name = "no_rom"

    def __init__(self) -> None:
        super().__init__()

    def __call__(self):
        pass


class IByPass(ROM):

    """It executes the Incremental Proper Orthogonal Decomposition using the SciKit-learn interface
    The IncrementalPCA class from SciKit-learn expects a two-dimensional array
    as input, so it is necessary to reshape the input data before processing it.
    This class is intended to be used for Big Data purposes.

    """

    name = "ibypass"

    def __init__(self, config=None, data_mean=None):
        super().__init__()

        self.kind = "batchwise"

    def fit(self, data: np.ndarray = None) -> None:
        """Output shape: (space_dimension, n_modes)

        Args:
            data (np.ndarray, optional):  (Default value = None)
        """
        pass

    def project(self, data: np.ndarray = None) -> np.ndarray:
        """Output shape: (n_timesteps, n_modes)

        Args:
            data (np.ndarray, optional):  (Default value = None)
        """

        return data

    def reconstruct(self, projected_data: np.ndarray = None) -> np.ndarray:
        """Output shape: (space_dimension, n_timesteps)

        Args:
            projected_data (np.ndarray, optional):  (Default value = None)
        """

        return projected_data

    def save(self, save_path: str = None, model_name: str = None) -> None:
        """It saves data in a NPZ file

        Args:
            save_path (str, optional): path to save the model (Default value = None)
            model_name (str, optional): name for the saved model (Default value = None)

        Returns:
            : nothing
        """

        np.savez(
            os.path.join(save_path, model_name + ".npz"), self.modes, self.data_mean
        )

    def restore(self, save_path: str = None, model_name: str = None) -> None:
        """It saves data in a NPZ file

        Args:
            save_path (str, optional): path to save the model (Default value = None)
            model_name (str, optional): name for the saved model (Default value = None)

        Returns:
            : nothing
        """

        self.modes, self.data_mean = np.load(
            os.path.join(save_path, model_name + ".npz")
        )


class ParallelSVD(ROM):
    name = "parallel_svd"

    def __init__(self, n_components: int = None, chunks: Tuple[int] = None) -> None:
        """Executing SVD using dask

        Args:
            n_components (int, optional):  (Default value = None)
            chunks (Tuple[int], optional):  (Default value = None)
        """
        super().__init__()

        self.n_components = n_components
        self.chunks = chunks
        self.default_chunks_numbers = (10, 10)
        self.U = None
        self.s = None
        self.V = None

    def _chunk_size_condition(self, size: int, chunk_size: int) -> int:
        if size // chunk_size == 0:
            return size
        else:
            return size // chunk_size

    def fit(
        self, data: Union[np.ndarray, da.core.Array] = None
    ) -> Union[np.ndarray, da.core.Array]:
        if self.chunks == None:
            chunks = [
                self._chunk_size_condition(size, self.default_chunks_numbers[j])
                for j, size in enumerate(data.shape)
            ]
        else:
            chunks = self.chunks

        if isinstance(data, np.ndarray):
            parallel_data = da.from_array(data, chunks=chunks)
        else:
            parallel_data = data

        U, s, V = da.linalg.svd_compressed(parallel_data, k=self.n_components)

        self.U = U
        self.s = s
        self.V = V


class POD(ROM):

    """It executes the classical Proper Orthogonal Decomposition using the SciKit-learn interface.
    The PCA class from SciKit-learn expects a two-dimensional array
    as input, so it is necessary to reshape the input data in order to
    ensure that

    """

    name = "pod"

    def __init__(self, config: dict = None, svd_filter: callable = None) -> None:
        """Propor Orthogonal Decomposition

        Args:
            config (dict, optional): configuration dictionary for the POD parameters (Default value = None)
            svd_filter (callable, optional): a filter callable applied to SVD decomposition (Default value = None)

        Returns:
            None: nothing
        """

        super().__init__()

        if "n_components" not in config:
            config["n_components"] = None

        if "mean_component" in config:
            self.mean_component = config.pop("mean_component")
        else:
            self.mean_component = True

        if "eig_norm" in config:
            self.eig_norm = config.pop("eig_norm")
        else:
            self.eig_norm = False

        self.pca = PCA(**config)

        self.modes = None

        self.data_mean = None

        self.svd_filter = svd_filter

    def fit(self, data: np.ndarray = None) -> None:
        """

        Args:
            data (np.ndarray, optional):  (Default value = None)

        Returns:
            : nothing
        """

        if self.mean_component:
            self.data_mean = data.mean(0)
            data_til = data - self.data_mean

            mean_contrib = np.linalg.norm(self.data_mean, 2) / np.linalg.norm(data, 2)

            print(
                "Relative contribution of the mean component: {}".format(mean_contrib)
            )
        else:
            data_til = data

        decomp = self.pca.fit(data_til)

        self.modes = decomp.components_
        self.singular_values = decomp.singular_values_

        # Executing SVD filtering over the singular values if necessary
        if self.svd_filter is not None:
            self.singular_values_truncated = self.svd_filter.exec(
                singular_values=self.singular_values, data_shape=data.shape
            )
            n_values = len(self.singular_values_truncated)
            self.singular_values = self.singular_values_truncated
            self.modes = self.modes[:n_values, :]

        else:
            pass

        relative_modal_energy = decomp.explained_variance_ratio_.sum()

        print("Relative Modal Energy {}".format(relative_modal_energy))

    def project(self, data: np.ndarray = None) -> np.ndarray:
        """

        Args:
            data (np.ndarray, optional): array of shape (n_samples, n_features) (Default value = None)

        Returns:
            np.ndarray: array of shape (n_samples, n_modes) containing
            np.ndarray: array of shape (n_samples, n_modes) containing
            the projection over the POD basis
        """

        if self.mean_component:
            data_til = data - self.data_mean
        else:
            data_til = data

        if not type(self.modes) == np.ndarray:
            self.fit(data_til)

        if self.eig_norm:
            return np.sqrt(self.singular_values)[None, :] * (data_til.dot(self.modes.T))
        else:
            return data_til.dot(self.modes.T)

    def reconstruct(self, projected_data: np.ndarray = None) -> np.ndarray:
        """

        Args:
            projected_data (np.ndarray, optional): array of shape (n_samples,
        n_modes) (Default value = None)

        Returns:
            np.ndarray: array of shape (n_samples, n_features)
        """

        n_modes_used = projected_data.shape[-1]
        n_modes = self.singular_values.shape[0]

        if n_modes_used < n_modes:
            print(f"Truncating the number of modes from {n_modes} to {n_modes_used}")

        if getattr(self, "eig_norm", False) != False:
            singular_values = self.singular_values[slice(0, n_modes_used)]
            projected_data = (1 / np.sqrt(singular_values)[None, :]) * projected_data
        else:
            pass

        """
               It is possible to reconstruct using less modes than
               created during the ROM construction, so we will
               adjust the size of self.modes according to projected_data
        """

        if self.mean_component:
            return (
                projected_data.dot(self.modes[slice(0, n_modes_used)]) + self.data_mean
            )
        else:
            return projected_data.dot(self.modes[slice(0, n_modes_used)])

    def save(self, save_path: str = None, model_name: str = None) -> None:
        """It saves data in a NPZ file

        Args:
            save_path (str, optional): path to save the model (Default value = None)
            model_name (str, optional): name for the saved model (Default value = None)

        Returns:
            : nothing
        """
        np.savez(
            os.path.join(save_path, model_name + ".npz"), self.modes, self.data_mean
        )

    def restore(self, save_path: str = None, model_name: str = None) -> None:
        """It saves data in a NPZ file

        Args:
            save_path (str, optional): path to save the model (Default value = None)
            model_name (str, optional): name for the saved model (Default value = None)

        Returns:
            : nothing
        """

        self.modes, self.data_mean = np.load(
            os.path.join(save_path, model_name + ".npz")
        )


class IPOD(ROM):

    """Incremental Propor Orthogonal Decomposition
    It executes the Incremental Proper Orthogonal Decomposition using the SciKit-learn interface
    The IncrementalPCA class from SciKit-learn expects a two-dimensional array
    as input, so it is necessary to reshape the input data before processing it.
    This class is intended to be used for Big Data purposes.

    """

    name = "ipod"

    def __init__(
        self,
        config: dict = None,
        data_mean: np.ndarray = None,
        svd_filter: callable = None,
    ) -> None:
        """

        Args:
            config (dict, optional): configuration dictionary for the POD parameters (Default value = None)
            data_mean (np.ndarray, optional): pre-evaluated mean of the dataset (Default value = None)
            svd_filter (callable, optional): a filter callable applied to SVD decomposition (Default value = None)

        Returns:
            None: nothing
        """

        super().__init__()

        self.kind = "batchwise"

        if "n_components" not in config:
            config["n_components"] = None

        if "mean_component" in config:
            self.mean_component = config.pop("mean_component")
        else:
            self.mean_component = True

        if "eig_norm" in config:
            self.eig_norm = config.pop("eig_norm")
        else:
            self.eig_norm = False

        self.pca = IncrementalPCA(**config)

        self.modes = None

        self.data_mean = data_mean

        self.data_size = None

        self.svd_filter = svd_filter

    def fit(self, data: np.ndarray = None) -> None:
        """Output shape: (space_dimension, n_modes)

        Args:
            data (np.ndarray, optional):  (Default value = None)
        """
        if self.data_mean is None:
            if not isinstance(self.data_mean, np.ndarray) and not self.data_size:
                self.data_mean = data.mean(0)
                self.data_size = data.shape[0]
            else:
                self.data_mean = (
                    self.data_size * self.data_mean + data.shape[0] * data.mean(0)
                ) / (self.data_size + data.shape[0])
                self.data_size += data.shape[0]
        else:
            assert (
                len(self.data_mean.shape) == 1
            ), f"The data_mean array must have dimension 1, but received shape {self.data_mean.shape}"

        if self.mean_component:
            data_til = data - self.data_mean
        else:
            data_til = data

        decomp = self.pca.partial_fit(data_til)
        self.modes = decomp.components_
        self.singular_values = decomp.singular_values_

        # Executing SVD filtering over the singular values if necessary
        if self.svd_filter is not None:
            self.singular_values_truncated = self.svd_filter.exec(
                singular_values=self.singular_values, data_shape=data.shape
            )
            n_values = len(self.singular_values_truncated)
            self.singular_values = self.singular_values_truncated
            self.modes = self.modes[:n_values, :]

        else:
            pass

        relative_modal_energy = decomp.explained_variance_ratio_.sum()

        print("Relative Modal Energy {}".format(relative_modal_energy))

        self.relative_modal_energy = relative_modal_energy

    def project(self, data: np.ndarray = None) -> np.ndarray:
        """

        Args:
            data (np.ndarray, optional): array of shape (n_samples, n_features) (Default value = None)

        Returns:
            np.ndarray: array of shape (n_samples, n_modes) containing
            np.ndarray: array of shape (n_samples, n_modes) containing
            the projection over the POD basis
        """

        if self.mean_component:
            data_til = data - self.data_mean
        else:
            data_til = data

        if not type(self.modes) == np.ndarray:
            self.fit(data)

        if self.eig_norm:
            return np.sqrt(self.singular_values)[None, :] * (data_til.dot(self.modes.T))
        else:
            return data_til.dot(self.modes.T)

    def reconstruct(self, projected_data: np.ndarray = None) -> np.ndarray:
        """

        Args:
            projected_data (np.ndarray, optional): array of shape (n_samples,
        n_modes) (Default value = None)

        Returns:
            np.ndarray: array of shape (n_samples, n_features)
        """

        n_modes_used = projected_data.shape[-1]

        if getattr(self, "eig_norm", False) != False:
            singular_values = self.singular_values[slice(0, n_modes_used)]
            projected_data = (1 / np.sqrt(singular_values)[None, :]) * projected_data
        else:
            pass

        """
            It is possible to reconstruct using less modes than
            created during the ROM construction, so we will
            adjust the size of self.modes according to projected_data
        """

        if self.mean_component:
            # We are using the approach of evaluating the mean value incrementally
            # If this is the best way for doing it, just the experiments will demonstrate
            return (
                projected_data.dot(self.modes[slice(0, n_modes_used)]) + self.data_mean
            )
        else:
            return projected_data.dot(self.modes[slice(0, n_modes_used)])

    def save(self, save_path: str = None, model_name: str = None) -> None:
        """It saves data in a NPZ file

        Args:
            save_path (str, optional): path to save the model (Default value = None)
            model_name (str, optional): name for the saved model (Default value = None)

        Returns:
            : nothing
        """

        np.savez(
            os.path.join(save_path, model_name + ".npz"), self.modes, self.data_mean
        )

    def restore(self, save_path: str = None, model_name: str = None) -> None:
        """It saves data in a NPZ file

        Args:
            save_path (str, optional): path to save the model (Default value = None)
            model_name (str, optional): name for the saved model (Default value = None)

        Returns:
            : nothing
        """

        self.modes, self.data_mean = np.load(
            os.path.join(save_path, model_name + ".npz")
        )


class HOSVD(ROM):

    """High-Order Singular Value Decomposition
    It executes the High-Order SVD using a multidimensional array as input.
    This class is intended to be used for Big Data purposes.

    """

    name = "hosvd"

    def __init__(
        self,
        n_components: List[int] = None,
        components_names: List[str] = None,
        engine: str = "sklearn",
        limit: str = "1 GiB",
    ) -> None:
        """

        Args:
            n_components (List[int], optional): list with the number of components for each direction (Default value = None)
            components_names (List[str], optional):  (Default value = None)
            engine (str, optional):  (Default value = "sklearn")
            limit (str, optional):  (Default value = "1 GiB")

        Returns:
            None: nothing
        """

        super().__init__()

        self.n_components = n_components

        # Naming the components of the HOSVD decomposition
        if components_names is None:
            self.components_names = [
                f"component_{i}" for i in range(len(self.n_components))
            ]
        else:
            assert len(components_names) == len(n_components), (
                "The number of components must be equal" " to the number of names."
            )
            self.components_names = components_names

        self.engine = engine
        self.limit = limit

        self.svd_classes = self._configure_SVD()

        self.sizelist = None
        self.shape = None
        self.n_dims = None
        self._comp_tag = "_decomp"
        self.U_list = list()
        self.S = None

        self.k_svd = self._k_svd

        if self.engine == "sklearn":
            self.lin = np
        elif self.engine == "dask":
            self.lin = da
        else:
            raise Exception(f"The engine {self.engine} is not supported.")

    def _configure_SVD(self) -> Union[List[TruncatedSVD], List[ParallelSVD]]:
        if self.engine == "sklearn":
            return [TruncatedSVD(n_components=n) for n in self.n_components]
        elif self.engine == "dask":
            return [ParallelSVD(n_components=n) for n in self.n_components]
        else:
            raise Exception(
                f"The engine {self.engine} is not supported, it must be in ['sklearn', 'dask']."
            )

    def _set_components(self) -> None:
        for j, name in enumerate(self.components_names):
            setattr(self, name.upper() + self._comp_tag, self.U_list[j])

    def _k_svd(
        self, data: np.ndarray = None, k: int = None
    ) -> Union[np.ndarray, da.core.Array]:
        """SVD applied to the k-mode flattening

        Args:
            data (np.ndarray, optional):  (Default value = None)
            k (int, optional):  (Default value = None)

        Returns:
            np.ndarray: Left eigenvectors matrix U
        """

        self.svd_classes[k].fit(data)

        if self.engine == "sklearn":
            s = self.svd_classes[k].singular_values_ * np.eye(self.n_components[k])
            VT = self.svd_classes[k].components_
            SVT = s @ VT
            U = (np.linalg.pinv(SVT.T) @ data.T).T

        else:
            U = getattr(self.svd_classes[k], "U")

        return U

    def _k_flattening(
        self, data: Union[np.ndarray, da.core.Array] = None, k: int = None
    ) -> Union[np.ndarray, da.core.Array]:
        """k-mode flattening

        Args:
            data (Union[np.ndarray, da.core.Array], optional):  (Default value = None)
            k (int, optional):  (Default value = None)

        Returns:
            np.ndarray: reshaped array of shape (n_1, n_2*n_3*...*n_n)
        """

        sizelist = copy.deepcopy(self.sizelist)
        sizelist_collapsible = copy.deepcopy(sizelist)

        sizelist[0] = k
        sizelist[k] = 0

        sizelist_collapsible.pop(k)
        collapsible_dims = np.prod([self.shape[s] for s in sizelist_collapsible])

        if isinstance(data, da.core.Array):
            return data.transpose(sizelist).reshape(
                (-1, collapsible_dims), limit=self.limit
            )
        else:
            return data.transpose(sizelist).reshape(-1, collapsible_dims)

    def fit(self, data: Union[np.ndarray, da.core.Array] = None) -> None:
        """Executing High-Order SVD

        Args:
            data (Union[np.ndarray, da.core.Array], optional): input array of shape (n_1, n_2, ..., n_n) (Default value = None)

        Returns:
            : nothing
        """
        import pprint

        pprinter = pprint.PrettyPrinter(indent=2)

        self.n_dims = len(data.shape)
        self.shape = data.shape
        S = data

        self.sizelist = np.arange(self.n_dims).tolist()

        print("Using the SVD classes:\n")
        pprinter.pprint(self.svd_classes)
        print("\n")

        for k in range(self.n_dims):
            print(f"Executing SVD for the dimension {k}")

            data_k_flatten = self._k_flattening(data=data, k=k)
            U = self.k_svd(data=data_k_flatten, k=k)

            self.U_list.append(U)

            S = self.lin.tensordot(S, U, axes=([0], [0]))

        self.S = np.array(S)

        self._set_components()

    def project(
        self, data: Union[np.ndarray, da.core.Array] = None
    ) -> Union[np.ndarray, da.core.Array]:
        """Projecting using the SVD basis

        Args:
            data (Union[np.ndarray, da.core.Array], optional): input array of shape (n_1, n_2, ..., n_n) (Default value = None)

        Returns:
            np.ndarray: reduced array of shape (n_1', n_2', ..., n_n')
        """

        assert len(data.shape) == self.n_dims
        S = data

        for k in range(self.n_dims):
            S = np.tensordot(S, self.U_list[k], axes=([0], [0]))

        return S

    def reconstruct(
        self,
        data: Union[np.ndarray, da.core.Array] = None,
        replace_components: dict = None,
    ) -> Union[np.ndarray, da.core.Array]:
        """Reconstruction using the pre-existent basis

        Args:
            data (Union[np.ndarray, da.core.Array], optional): reduced array of shape (n_1', n_2', ...,
        n_n') (Default value = None)
            replace_components (dict, optional):  (Default value = None)

        Returns:
            np.ndarray: reconstructed array of shape (n_1, n_2, n_3,...,
            np.ndarray: reconstructed array of shape (n_1, n_2, n_3,...,
            n_n)
        """

        if replace_components is not None:
            U_list = copy.deepcopy(self.U_list)

            for key, value in replace_components.items():
                try:
                    index = self.components_names.index(key)
                except:
                    raise Exception(f"The key {key} is not in the list of components.")

                U_list[index] = value
        else:
            U_list = self.U_list

        A = data
        modes = np.arange(self.n_dims).tolist()

        for k in modes:
            A = np.tensordot(U_list[k], A, axes=([1], [k]))

        return A.transpose()

        # Saving to disk the complete model

    def save(self, save_path: str = None, model_name: str = None) -> None:
        """Complete saving

        Args:
            save_path (str, optional): path to the saving directory (Default value = None)
            model_name (str, optional): name for the model (Default value = None)

        Returns:
            : nothing
        """
        blacklist = ["lin"]
        for el in blacklist:
            setattr(self, el, None)

        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(self, fp, protocol=4)
        except Exception as e:
            print(e, e.args)


class DMD(ROM):
    def __init__(self, config=None):
        """

        Args:
            config:  (Default value = None)
        
        """
        super().__init__()
        for key, value in config.items():
            setattr(self, key, value)

        self.Lambda = None
        self.Phi = None
        self.A_tilde = None
        self.initial_state = None

    """The method fit from DMD receives an array with shape (nt, np.product(*dims))
       and constructs a model for estimating the state nt+1
    """

    def fit(self, data=None):
        data = data.T

        X_aug = np.vstack((data[:, 0:-2], data[:, 1:-1]))
        X_aug_tilde = np.vstack((data[:, 1:-1], data[:, 2:]))

        U, Sig, VT = np.linalg.svd(X_aug, full_matrices=False)
        A_tilde = U.T @ X_aug_tilde @ VT.T @ np.linalg.inv(np.diag(Sig))

        self.A_tilde = A_tilde

        Lambda, W = np.linalg.eig(A_tilde)

        self.Lambda = Lambda

        Phi = X_aug_tilde @ VT.T @ np.linalg.inv(np.diag(Sig)) @ W

        self.Phi = Phi

        initial_state = (
            np.linalg.inv(Phi.conj().T @ Phi) @ Phi.conj().T @ X_aug_tilde[:, -1]
        )

        self.initial_state = initial_state

        print("Fitting process concluded.")

    def predict(self, step=None):
        return self.Phi @ np.diag(self.Lambda ** (step - 1)) @ self.initial_state


# Gappy POD
class GPOD(ROM):
    def __init__(self, pca_type="pod", pca_config=None, config=None):
        """GPOD

        Args:
            pca_type (str, optional): the kind of PCA to be used (Default value = "pod")
            pca_config:  (Default value = None)
            config:  (Default value = None)
        """
        super().__init__()

        this_module = importlib.import_module("simulai.rom")

        # A PCA instance is used for constructing the basis
        self.pca_type = pca_type
        self.config = config

        self.sensors_distribution = None
        self.n_sensors = None
        self.sensors_placer = None

        for key, value in config.items():
            setattr(self, key, value)

        assert self.sensors_distribution, "sensors_distribution must be provided"

        if not self.sensors_placer or self.sensors_placer != "extrema":
            print(
                "As no placement criteria eas provided for the sensor, the extrema method will be used."
            )
            self.sensors_placer = "extrema"
        else:
            raise Exception(
                f"The placement method {self.sensors_placer} is not supported."
            )

        if self.sensors_placer == "extrema":
            assert all(
                [not item % 2 for item in self.sensors_distribution]
            ), "If extrema placement is being used, all the number of sensors must be pair"

        self.placer = getattr(self, "_" + self.sensors_placer)

        self.n_sensors = sum(self.sensors_distribution)

        self.pca_class = getattr(this_module, self.pca_type.upper())

        self.pca = self.pca_class(config=pca_config)

        self.modes = None
        self.M = None
        self.M_inv = None
        self.mask_array = None

    # It gets the positions related to the n maximum and n minimum values to be used
    # to locate sensors
    def _extrema(self):
        locations = list()
        n_modes = self.modes.shape[0]

        for mode_i in range(n_modes):
            n_sensors = self.sensors_distribution[mode_i]
            n_minimum = n_maximum = int(n_sensors / 2)

            locations += self.modes[mode_i].argsort()[:n_minimum].tolist()
            locations += self.modes[mode_i].argsort()[-n_maximum:].tolist()

        return locations

    # The m dot product (a, b)_m = (m*a, m*b), in which m is a mask array
    def m_dot(self, a, b, mask_array=None):
        return (mask_array * a).dot((mask_array * b).T)

    def fit(self, data=None):
        self.pca.fit(data=data)
        self.modes = self.pca.modes

        n_features = self.modes.shape[1]

        sensors_locations = self.placer()

        mask_array = np.zeros((1, n_features))
        mask_array[:, sensors_locations] = 1

        self.mask_array = mask_array
        self.M = self.m_dot(self.modes, self.modes, mask_array=mask_array)
        self.M_inv = np.linalg.inv(self.M)

        print(f"The condition number for the matrix M is {np.linalg.cond(self.M)}")

    def project(self, data=None):
        data_til = self.mask_array * data
        f = self.m_dot(data_til, self.modes, mask_array=self.mask_array)

        return f @ self.M_inv.T

    def reconstruct(self, projected_data=None):
        return self.pca.reconstruct(projected_data=projected_data)


# Quasi-Quadratic Manifold
class QQM:
    def __init__(
        self,
        n_inputs: int = None,
        alpha_0: float = None,
        sparsity_tol: float = 1e-15,
        lambd: float = None,
        epsilon: float = 1e-10,
        use_mean: bool = False,
    ) -> None:
        """It extends and enriches the POD approach by determining a quadratic basis for its residual

        Args:
            n_inputs (int, optional): number of inputs used in the POD approximation (Default value = None)
            alpha_0 (float, optional): regularization parameter used in SparSA algorithm (Default value = None)
            sparsity_tol (float, optional): sparsity tolerance used in SpaRSA (Default value = 1e-15)
            lambd (float, optional): regularization parameter used in SparSA algorithm (Default value = None)
            epsilon (float, optional): threshold for zeroing columns in SpaRSA (Default value = 1e-10)
            use_mean (bool, optional): use mean for the SpaRSA loss function of not ? (Default value = False)

        Returns:
            : nothing
        """

        self.alpha_0 = alpha_0
        self.lambd = lambd
        self.epsilon = epsilon
        self.n_inputs = n_inputs
        self.i_u, self.j_u = np.triu_indices(self.n_inputs)
        self.V_bar = None
        self.valid_indices = None

        self.optimizer = SpaRSA(
            lambd=self.lambd,
            alpha_0=alpha_0,
            use_mean=use_mean,
            sparsity_tol=sparsity_tol,
            epsilon=epsilon,
            transform=self.W_transform,
        )

    def _kronecker_product(
        self, a: np.ndarray = None, b: np.ndarray = None
    ) -> np.ndarray:
        """It executes a Kronecker dot between two arrays

        Args:
            a (np.ndarray, optional): left array (Default value = None)
            b (np.ndarray, optional): right (transposed) array (Default value = None)

        Returns:
            np.ndarray: the Kronecker output array
        """

        assert (
            a.shape == b.shape
        ), f"a and b must have the same shape, but received {a.shape} and {b.shape}"

        kron_output = np.einsum("bi, bj->bij", a, b)

        assert (
            np.isnan(kron_output).max() == False
        ), "There are NaN in the Kronecker output"

        # Checking if the Kronecker output tensor is symmetric or not
        if np.array_equal(kron_output, kron_output.transpose(0, 2, 1)):
            return kron_output[:, self.i_u, self.j_u]
        else:
            shapes = kron_output.shape[1:]
            return kron_output.reshape(-1, np.prod(shapes))

    # Each batch in W has n_inputs*(n_inputs + 1)/2 columns
    def W_transform(self, data: np.ndarray = None) -> np.ndarray:
        """W_transform simply applied Kronecker product for data itself

        Args:
            data (np.ndarray, optional): the data to be W-transformed (Default value = None)

        Returns:
            np.ndarray: the Kronecker product between data and data
            np.ndarray: the Kronecker product between data and data
            itself
        """

        return self._kronecker_product(a=data, b=data)

    def fit(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        pinv: bool = False,
    ) -> None:
        """It executes the fitting process using the chosen optimization algorithm, SpaRSA
         or Moore-Penrose pseudoinverse

        Args:
            input_data (np.ndarray, optional): in general, the original latent
        series (Default value = None)
            target_data (np.ndarray, optional): in general, the residual of the
        linear approximation (Default value = None)
            pinv (bool, optional): use pseudoinverse or not (Default value = False)

        Returns:
            : nothing
        """

        if not pinv:
            self.V_bar = self.optimizer.fit(
                input_data=input_data, target_data=target_data
            )
        else:
            V_bar = np.linalg.pinv(self.W_transform(data=input_data)) @ target_data
            self.V_bar = np.where(np.abs(V_bar) < self.optimizer.sparsity_tol, 0, V_bar)

        self.valid_indices = np.argwhere(
            np.sum(np.abs(self.V_bar), axis=1) > 0
        ).flatten()

        print(
            f"\n Number of original modes: {self.i_u.size}. Number of modes selected: {self.valid_indices.size}"
        )

    def project(self, data: np.ndarray = None) -> np.ndarray:
        """Executes the W-transformation and collects just the valid modes determined
         by the optimization algorithm

        Args:
            data (np.ndarray, optional): the data to be projected (Default value = None)

        Returns:
            np.ndarray: the projection over the selected basis
        """

        return self.W_transform(data=data)[:, self.valid_indices]

    def eval(self, data: np.ndarray = None) -> None:
        """It projects and reconstructs

        Args:
            data (np.ndarray, optional): the data to be projected (Default value = None)

        Returns:
            np.ndarray: the approximated data
        """

        return self.W_transform(data=data) @ self.V_bar

    def save(self, save_path: str = None, model_name: str = None) -> None:
        """Complete saving

        Args:
            save_path (str, optional): path to the saving directory (Default value = None)
            model_name (str, optional): name for the model (Default value = None)

        Returns:
            : nothing
        """

        blacklist = ["optimizer"]
        for el in blacklist:
            setattr(self, el, None)

        path = os.path.join(save_path, model_name + ".pkl")
        try:
            with open(path, "wb") as fp:
                pickle.dump(self, fp, protocol=4)
        except Exception as e:
            print(e, e.args)
