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
from itertools import product
from collections import OrderedDict
import copy

class StructuredMesh:

    def __init__(self, dim_bounds=None, dim_gdl=None, boundary_dim_gdl=None, dim_tags=None):

        self.n_dim = len(dim_tags)

        self.dim_tags = dim_tags
        self.mesh_tags = list()

        self.elements = OrderedDict()
        self.boundary_nodes_tags = list()
        self.boundary_elements = OrderedDict()

        self.dim_gdl_tags = {tag: gdl for tag, gdl in zip(dim_tags, dim_gdl)}

        if boundary_dim_gdl:
            self.boundary_dim_gdl_tags = {tag: gdl for tag, gdl in zip(dim_tags, boundary_dim_gdl)}
        else:
            self.boundary_dim_gdl_tags = self.dim_gdl_tags

        # Constructing axis
        for bounds, gdl, tag in zip(dim_bounds, dim_gdl, dim_tags):

            setattr(self, tag, np.linspace(*bounds, gdl + 1))

        mesh_matrices = np.meshgrid(*[getattr(self, tag) for tag in dim_tags])

        # Constructing the mesh matrices
        for tag, matrix in zip(dim_tags, mesh_matrices):

            matrix_tag = tag.capitalize() + '_f'
            self.mesh_tags.append(matrix_tag)
            setattr(self, matrix_tag, matrix)

        # Constructing the mesh elements
        for tag in self.dim_tags:

            gdl = self.dim_gdl_tags[tag]

            domain = getattr(self, tag).copy()

            subdomains = [domain[i:i+2] for i in range(0, gdl, 1)]

            setattr(self, tag+'_subdomains', subdomains)

        for ii, el in enumerate(product(*[getattr(self, tag+'_subdomains')
                                          for tag in dim_tags])):

            self.elements['el_' + str(ii)] = el

        for bounds, gdl, tag in zip(dim_bounds, boundary_dim_gdl, dim_tags):
            setattr(self, tag + '_b', np.linspace(*bounds, gdl + 1))

        # Constructing the boundaries
        for dim, dim_tag in enumerate(self.dim_tags):

            dim_tags = copy.copy(self.dim_tags)

            lower_bound = getattr(self, dim_tag + '_b').copy()[0]
            upper_bound = getattr(self, dim_tag + '_b').copy()[-1]

            lower_boundary = np.meshgrid(np.array([lower_bound]), *self._get_boundaries_curves(but=dim_tag))
            upper_boundary = np.meshgrid(np.array([upper_bound]), *self._get_boundaries_curves(but=dim_tag))

            dim_tags.remove(dim_tag)

            setattr(self, dim_tag + '_' + dim_tag + '_b0', lower_boundary[0])
            setattr(self, dim_tag + '_' + dim_tag + '_bL', upper_boundary[0])

            lower_boundary.pop(0)
            upper_boundary.pop(0)

            for ii, otag in enumerate(dim_tags):

                setattr(self, otag + '_' + dim_tag + '_b0', lower_boundary[ii])
                setattr(self, otag + '_' + dim_tag + '_bL', upper_boundary[ii])

                self.boundary_nodes_tags.append(otag + '_' + dim_tag + '_b0')
                self.boundary_nodes_tags.append(otag + '_' + dim_tag + '_bL')

        # Constructing the boundary elements
        for ii, bb in enumerate(self.boundary_nodes_tags):

            boundary_array = getattr(self, bb).copy()
            gdl = boundary_array.shape[0]

            tag = bb.split('_')[0]
            index = self.dim_tags.index(tag)

            subdomains = {'el' + str(i) + '_' + bb: (boundary_array[i:i + 2], index)
                          for i in range(0, gdl - 1, 1)}

            self.boundary_elements[bb] = subdomains

    def _get_boundaries_curves(self, but=None):

        return [getattr(self, tag + '_b') for tag in self.dim_tags if tag != but]

    def internal_product(self, vector):

        if isinstance(vector, list):
            product_list = self.n_dim*(vector,)
        elif isinstance(vector, tuple):
            product_list = vector
        else:
            raise Exception("The internal product cannot be performed.")

        return list(product(*product_list))

    def internal_boundary_product(self, vector):

        vector_ = np.array(vector)

        if len(vector_.shape) > 1:
            vector_ = np.array(vector)[:, tag]
        else:
            pass

        if isinstance(vector_, np.ndarray):
            product_list = (vector_,)
        elif isinstance(vector_, tuple):
            product_list = vector_
        else:
            raise Exception("The internal product cannot be performed.")

        return list(product(*product_list))

    def map_to_element(self, points, reference_interval, el):

        lower_bound, upper_bound = reference_interval

        local_el = np.array(el)
        local_points = np.array(points)

        dims_max = local_el.max(1)
        dims_min = local_el.min(1)

        points_mapped = dims_min + (dims_max - dims_min)*(local_points - lower_bound)\
                        /(upper_bound - lower_bound)

        return points_mapped

    def map_to_boundary_element(self, points, reference_interval, el, tag=None):

        lower_bound, upper_bound = reference_interval

        local_el = np.array(el)
        if tag != None :
            local_points = np.array(points[tag])
        else:
            local_points = np.array(points)

        dims_max = local_el.max(0)
        dims_min = local_el.min(0)

        points_mapped = dims_min + (dims_max - dims_min)*(local_points - lower_bound)\
                        /(upper_bound - lower_bound)

        return points_mapped.T




