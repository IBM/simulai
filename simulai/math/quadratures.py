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

import sys

import numpy as np
from scipy.special import jacobi


class GaussLegendre:
    def __init__(self, p_order=None):
        """
        Initializes the Quadrature class.

        Parameters
        ----------
        p_order : int or tuple, optional
            Order of the polynomial. If a tuple is given, the quadrature is assumed to be adaptative.

        """

        self.p_order = p_order
        self.alpha = 0
        self.beta = 0
        self.reference_interval = [-1, 1]

        # Considering a homogeneous p-order
        # Evaluate the weights and nodes of the
        # element
        if isinstance(p_order, int):
            self.poly = jacobi(self.p_order, self.alpha, self.beta)
            self.poly_der = self.poly.deriv(1)
            self.poly_roots = sorted(self.poly.roots)
            self.exec = self._exec_homogeneous
            self.weights = [
                2 / ((1 - root**2) * self.poly_der(root) ** 2)
                for root in self.poly_roots
            ]
            print("")
        elif isinstance(p_order, tuple):
            self.poly = tuple()
            self.poly_der = tuple()
            self.poly_roots = tuple()
            self.weights = tuple()

            for _p_order in p_order:
                poly = jacobi(_p_order, self.alpha, self.beta)
                poly_der = poly.deriv(1)
                poly_roots = sorted(poly.roots)

                self.exec = self._execute_adaptative

                weights = [
                    2 / ((1 - root**2) * poly_der(root) ** 2) for root in poly_roots
                ]

                self.poly += (poly,)
                self.poly_der += (poly_der,)
                self.poly_roots += (poly_roots,)
                self.weights += (weights,)
        else:
            pass

    def _exec_homogeneous(self):
        pass

    def _execute_adaptative(self):
        pass

    def generate_domain(self, mesh=None):
        """
        Generates domain for the given mesh.

        Parameters
        ----------
        mesh : object
            Mesh object.

        Returns
        -------
        tuple
            Tuple containing arrays for each dimension of the domain and weights.

        """

        nodes = mesh.internal_product(self.poly_roots)
        n_dim = mesh.n_dim

        weights = np.array(mesh.internal_product(self.weights)).prod(axis=1)[:, None]

        nodes_list = list()
        weights_list = list()

        for key, el in mesh.elements.items():
            sys.stdout.write(
                "\rMapping from the reference to the real mesh element {}".format(key)
            )
            sys.stdout.flush()

            nodes_mapped = mesh.map_to_element(nodes, self.reference_interval, el)

            nodes_list.append(nodes_mapped)
            weights_list.append(weights)

        nodes_array = np.vstack(nodes_list)
        weights_array = np.vstack(weights_list)

        dim_arrays = np.split(nodes_array, n_dim, axis=1)
        dim_arrays.append(weights_array)

        return tuple(dim_arrays)

    def generate_boundaries(self, mesh=None):
        """Generate boundary nodes and weights using the provided mesh.

        Parameters
        ----------
        mesh : object
            Mesh object containing boundary information.

        Returns
        -------
        boundaries_list : dict
            Dictionary of boundary nodes and weights, where the keys are the boundary tags and the values are tuples of
            (nodes, weights) arrays.
        """

        boundaries_list = dict()

        for boundary in mesh.boundary_nodes_tags:
            nodes_list = list()
            weights_list = list()

            for key, (el, tag) in mesh.boundary_elements[boundary].items():
                nodes = mesh.internal_boundary_product(self.poly_roots)

                weights = np.array(mesh.internal_boundary_product(self.weights)).prod(
                    axis=1
                )[:, None]

                sys.stdout.write(
                    "\rMapping from the reference to the real mesh element {} from {}".format(
                        key, boundary
                    )
                )
                sys.stdout.flush()

                if isinstance(self.p_order, tuple):
                    nodes_mapped = mesh.map_to_boundary_element(
                        nodes, self.reference_interval, el, tag
                    )
                    nodes_list.append(nodes_mapped)
                    weights_list.append(weights)

                else:
                    nodes_mapped = mesh.map_to_boundary_element(
                        nodes, self.reference_interval, el
                    )

                    nodes_list.append(nodes_mapped.T)
                    weights_list.append(weights)

            nodes_array = np.vstack(nodes_list)
            weights_array = np.vstack(weights_list)

            boundaries_list[boundary] = (nodes_array, weights_array)

        return boundaries_list
