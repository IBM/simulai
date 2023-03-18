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
from collections import OrderedDict
from itertools import product
from typing import Dict, Optional, Tuple, Union

import numpy as np


class StructuredMesh:
    """
    A class for generating structured meshes with a given dimensionality and boundary conditions.
    """

    def __init__(
        self, dim_bounds=None, dim_gdl=None, boundary_dim_gdl=None, dim_tags=None
    ):
        """
        Initializes a StructuredMesh object with the given dimension bounds, number of grid points, boundary conditions, and dimension tags.

        Parameters:
        ----------
        dim_bounds : list of tuple or array-like, optional
            A list of tuples or array-like objects, where each tuple or array-like represents the
            bounds of the corresponding dimension. The default is None.
        dim_gdl : list of int or array-like, optional
            A list of integers or array-like objects, where each integer or array-like represents
            the number of grid points of the corresponding dimension. The default is None.
        boundary_dim_gdl : list of int or array-like, optional
            A list of integers or array-like objects, where each integer or array-like represents
            the number of grid points on the boundary of the corresponding dimension. The default is None.
        dim_tags : list of str, optional
            A list of strings representing the tags of the mesh dimensions. The default is None.

        Returns:
        -------
        None.

        """
        self.n_dim = len(dim_tags)

        self.dim_tags = dim_tags
        self.mesh_tags = list()

        self.elements = OrderedDict()
        self.boundary_nodes_tags = list()
        self.boundary_elements = OrderedDict()

        self.dim_gdl_tags = {tag: gdl for tag, gdl in zip(dim_tags, dim_gdl)}

        if boundary_dim_gdl:
            self.boundary_dim_gdl_tags = {
                tag: gdl for tag, gdl in zip(dim_tags, boundary_dim_gdl)
            }
        else:
            self.boundary_dim_gdl_tags = self.dim_gdl_tags

        # Constructing axis
        for bounds, gdl, tag in zip(dim_bounds, dim_gdl, dim_tags):
            setattr(self, tag, np.linspace(*bounds, gdl + 1))

        mesh_matrices = np.meshgrid(*[getattr(self, tag) for tag in dim_tags])

        # Constructing the mesh matrices
        for tag, matrix in zip(dim_tags, mesh_matrices):
            matrix_tag = tag.capitalize() + "_f"
            self.mesh_tags.append(matrix_tag)
            setattr(self, matrix_tag, matrix)

        # Constructing the mesh elements
        for tag in self.dim_tags:
            gdl = self.dim_gdl_tags[tag]

            domain = getattr(self, tag).copy()

            subdomains = [domain[i : i + 2] for i in range(0, gdl, 1)]

            setattr(self, tag + "_subdomains", subdomains)

        for ii, el in enumerate(
            product(*[getattr(self, tag + "_subdomains") for tag in dim_tags])
        ):
            self.elements["el_" + str(ii)] = el

        for bounds, gdl, tag in zip(dim_bounds, boundary_dim_gdl, dim_tags):
            setattr(self, tag + "_b", np.linspace(*bounds, gdl + 1))

        # Constructing the boundaries
        for dim, dim_tag in enumerate(self.dim_tags):
            dim_tags = copy.copy(self.dim_tags)

            lower_bound = getattr(self, dim_tag + "_b").copy()[0]
            upper_bound = getattr(self, dim_tag + "_b").copy()[-1]

            lower_boundary = np.meshgrid(
                np.array([lower_bound]), *self._get_boundaries_curves(but=dim_tag)
            )
            upper_boundary = np.meshgrid(
                np.array([upper_bound]), *self._get_boundaries_curves(but=dim_tag)
            )

            dim_tags.remove(dim_tag)

            setattr(self, dim_tag + "_" + dim_tag + "_b0", lower_boundary[0])
            setattr(self, dim_tag + "_" + dim_tag + "_bL", upper_boundary[0])

            lower_boundary.pop(0)
            upper_boundary.pop(0)

            for ii, otag in enumerate(dim_tags):
                setattr(self, otag + "_" + dim_tag + "_b0", lower_boundary[ii])
                setattr(self, otag + "_" + dim_tag + "_bL", upper_boundary[ii])

                self.boundary_nodes_tags.append(otag + "_" + dim_tag + "_b0")
                self.boundary_nodes_tags.append(otag + "_" + dim_tag + "_bL")

        # Constructing the boundary elements
        for ii, bb in enumerate(self.boundary_nodes_tags):
            boundary_array = getattr(self, bb).copy()
            gdl = boundary_array.shape[0]

            tag = bb.split("_")[0]
            index = self.dim_tags.index(tag)

            subdomains = {
                "el" + str(i) + "_" + bb: (boundary_array[i : i + 2], index)
                for i in range(0, gdl - 1, 1)
            }

            self.boundary_elements[bb] = subdomains

    def _get_boundaries_curves(self, but=None):
        """
        Returns a list of numpy arrays that represent the boundary curves for each
        dimension of the mesh except for the one specified by the `but` parameter.

        Parameters
        ----------
        but : str, optional
            The dimension tag to exclude. Default is None.

        Returns
        -------
        list of numpy.ndarray
            A list of boundary curves for each dimension except `but`.
        """
        return [getattr(self, tag + "_b") for tag in self.dim_tags if tag != but]

    def internal_product(self, vector):
        """
        Computes the internal product of the given vector and the mesh, returning a list
        of tuples representing the resulting vectors.

        Parameters
        ----------
        vector : list or tuple
            A list or tuple of vectors to perform the internal product with.

        Returns
        -------
        list of tuple
            A list of tuples representing the resulting vectors after the internal product.
        Raises
        ------
        Exception
            If the input `vector` is neither a list nor a tuple.
        """
        if isinstance(vector, list):
            product_list = self.n_dim * (vector,)
        elif isinstance(vector, tuple):
            product_list = vector
        else:
            raise Exception("The internal product cannot be performed.")

        return list(product(*product_list))

    def internal_boundary_product(self, vector):
        """
        Calculates the internal boundary product of a given vector.

        Parameters
        ----------
        vector : numpy.ndarray or tuple
            The input vector for which the internal boundary product is to be calculated.

        Returns
        -------
        list
            The list of internal boundary product of the input vector.

        Raises
        ------
        Exception
            If the input is not a 1-D numpy array or a tuple.

        Examples
        --------
        # Example 1: Using a 1-D numpy array
        >>> vector = np.array([1, 2, 3])
        >>> obj.internal_boundary_product(vector)
        [(1,), (2,), (3,)]

        # Example 2: Using a tuple
        >>> vector = (1, 2, 3)
        >>> obj.internal_boundary_product(vector)
        [(1,), (2,), (3,)]

        # Example 3: Using a 2-D numpy array
        >>> vector = np.array([[1, 2, 3], [4, 5, 6]])
        >>> obj.internal_boundary_product(vector)
        An error occurred: The input must be a 1-D array or a tuple.
        """
        try:
            vector_ = np.array(vector)
            if len(vector_.shape) > 1:
                raise Exception("The input must be a 1-D array or a tuple.")
            else:
                pass
            if isinstance(vector_, np.ndarray):
                product_list = (vector_,)
            elif isinstance(vector_, tuple):
                product_list = vector_
            else:
                raise Exception("The input must be a numpy array or a tuple.")
            return list(product(*product_list))
        except Exception as e:
            print("An error occurred: ", e)

    def map_to_element(
        self,
        points: np.ndarray,
        reference_interval: Tuple[float, float],
        el: np.ndarray,
    ) -> np.ndarray:
        """
        Map a set of points from the reference interval to an element.

        Parameters
        ----------
        points : np.ndarray
            Points in the reference interval to be mapped.
        reference_interval : tuple of float
            Lower and upper bounds of the reference interval.
        el : np.ndarray
            Coordinates of the vertices of the element.

        Returns
        -------
        np.ndarray
            Mapped points in the element.

        """
        lower_bound, upper_bound = reference_interval

        local_el = np.array(el)
        local_points = np.array(points)

        dims_max = local_el.max(1)
        dims_min = local_el.min(1)

        points_mapped = dims_min + (dims_max - dims_min) * (
            local_points - lower_bound
        ) / (upper_bound - lower_bound)

        return points_mapped

    def map_to_boundary_element(
        self,
        points: Union[np.ndarray, Dict[str, np.ndarray]],
        reference_interval: Tuple[float, float],
        el: np.ndarray,
        tag: Optional[str] = None,
    ) -> np.ndarray:
        """
        Map a set of points from the reference interval to the boundary of an element.

        Parameters
        ----------
        points : np.ndarray or dict
            Points in the reference interval to be mapped.
        reference_interval : tuple of float
            Lower and upper bounds of the reference interval.
        el : np.ndarray
            Coordinates of the vertices of the element.
        tag : str, optional
            The tag of the boundary, by default None.

        Returns
        -------
        np.ndarray
            Mapped points on the boundary of the element.

        """
        lower_bound, upper_bound = reference_interval

        local_el = np.array(el)
        if tag is not None:
            local_points = np.array(points[tag])
        else:
            local_points = np.array(points)

        dims_max = local_el.max(0)
        dims_min = local_el.min(0)

        points_mapped = dims_min + (dims_max - dims_min) * (
            local_points - lower_bound
        ) / (upper_bound - lower_bound)

        return points_mapped.T
