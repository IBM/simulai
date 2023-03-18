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
import inspect
import os
import pickle
import warnings
from collections import OrderedDict

import h5py
import numpy as np

import simulai
from simulai.abstract import BaseFramework
from simulai.batching import batchdomain_constructor
from simulai.io import DataPreparer
from simulai.rom import ROM


# Prototype of the class Pipeline
class Pipeline(BaseFramework):
    """
    Pipeline class is used to create a pipeline of algorithms to be used in a simulation.

    Methods:
    --------
    set_data_preparer(data_preparer: DataPreparer) -> None:
        Set the data preparer to be used in the pipeline
    set_rom(rom: ROM) -> None:
        Set the ROM to be used in the pipeline
    set_model(model: Model) -> None:
        Set the model to be used in the pipeline
    set_normalization(normalization: Normalization) -> None:
        Set the normalization to be used in the pipeline
    set_integration(integration: Integration) -> None:
        Set the integration to be used in the pipeline
    set_target(target: Dataset) -> None:
        Set the target dataset to be used in the pipeline
    set_reference(reference: Dataset) -> None:
        Set the reference dataset to be used in the pipeline
    set_batchwise(batchwise: bool) -> None:
        Set the batchwise flag to be used in the pipeline
    set_input_vars_list(input_vars_list: list) -> None:
        Set the input variables list to be used in the pipeline
    set_target_vars_list(target_vars_list: list) -> None:
        Set the target variables list to be used in the pipeline
    set_reference_vars_list(reference_vars_list: list) -> None:
        Set the reference variables list to be used in the pipeline
    set_input_data(input_data: np.ndarray) -> None:
        Set the input data to be used in the pipeline
    set_target_data(target_data: np.ndarray) -> None:
        Set the target data to be used in the pipeline
    set_reference_data(reference_data: np.ndarray) -> None:
        Set the reference data to be used in the pipeline
    set_input_data_prepared(input_data_prepared: np.ndarray) -> None:
        Set the input data prepared to be used in the pipeline
    set_target_data_prepared(target_data_prepared: np.ndarray) -> None:
        Set the target data prepared to be used in the pipeline
    set_reference_data_prepared(reference_data_prepared: np.ndarray) -> None:
        Set the reference data prepared to be used in the pipeline
    set_input_data_normalized(input_data_normalized: np.ndarray) -> None:
        Set the input data normalized to be used in the pipeline

    """

    def __init__(self, stages: list = list(), channels_last: bool = False) -> None:
        """

        :param stages:
        :type: stages: List[Tuple]
        """

        self.channels_last = channels_last

        super().__init__()
        assert isinstance(stages, list), "Error! stages is not a list"
        self.stages = OrderedDict(stages)

        self.triage_dict = {
            "data_preparer": "prepare_input_data",
            "rom": "fit",
            "model": "fit",
            "integration": "__call__",
            "normalization": "rescale",
        }

        self.wrappers_dict = {
            "data_preparer": self._data_preparer_wrapper,
            "rom": self._rom_wrapper,
            "model": self._model_wrapper,
            "normalization": self._normalization_wrapper,
            "integration": self._integration_wrapper,
        }

        # Some global directives:
        pipeline_layers = self.stages.keys()

        self.there_is_data_preparer = False
        self.there_is_rom = False
        self.there_is_model = False
        self.there_is_target = False
        self.there_is_reference = False
        self.is_batchwise = False

        if "data_preparer" in pipeline_layers:
            self.there_is_data_preparer = True
        if "rom" in pipeline_layers:
            self.there_is_rom = True
        if "model" in pipeline_layers:
            self.there_is_model = True

        self.execution_pipeline, self.pipeline_algorithms = self._classify_op(
            self.stages
        )

        # Multiple global attributes are initialized as empty objects and filled at
        # execution time
        self._input_data = None
        self.target_data = None
        self.reference_data = None
        self.fig_kwargs = None
        self.data = None
        self.model = None

        self.input_vars_list = None
        self.target_vars_list = None

        self.output = None
        self.rom = None
        # Bypass used when no data preparer is provided
        self.data_preparer = None
        self.data_generator = None
        self.normalization = None

        self.fit_kwargs = {}
        self.slicer = None

    @property
    def input_data(self):
        """
        this function is used to get the input data

        Returns:
        -------
        input_data: np.ndarray
            input data
        """
        return self._input_data

    @input_data.setter
    def input_data(self, v):
        """
        function to set the input data

        Parameters:
        ----------
        v: np.ndarray
            input data


        Exceptions:
        -----------
        assert: AssertionError
            if the input data is not a numpy.ndarray

        Returns:
        -------
        None
        """
        assert isinstance(v, np.ndarray) or isinstance(
            v, h5py.Dataset
        ), "Error! input_data is not a numpy.ndarray: {}".format(type(v))

        self._input_data = v

    @staticmethod
    def _construct_data_array(data, var_names_list):
        """
        Construct a structured array from a list of variables

        Parameters:
        -----------
        var_names_list: List[str]
            list of variable names

        data: np.ndarray
            data array

        Returns:
        --------
        data: np.ndarray
            structured array
        """
        # It considers the argument data as a structured array

        if isinstance(data, np.ndarray) and isinstance(data.dtype.names, tuple):
            return data.view(float)

        else:
            return data

    def _classify_op(self, op_dict):
        """
        Classify the operations in the pipeline

        Parameters:
        -----------
        op_dict: OrderedDict
            dictionary of operations

        Returns:
        --------
        execution_pipeline: OrderedDict
            dictionary of operations

        pipeline_algorithms: OrderedDict
            dictionary of algorithms
        """

        execution_pipeline = OrderedDict()
        pipeline_algorithms = {
            key: {"op": list(), "wrapper": None} for key in self.triage_dict
        }

        for node, op in op_dict.items():
            execution_method = self.triage_dict.get(op.purpose)
            wrapper_method = self.wrappers_dict.get(op.purpose)

            execution_pipeline[op] = execution_method

            pipeline_algorithms[op.purpose]["op"] = execution_method
            pipeline_algorithms[op.purpose]["wrapper"] = wrapper_method

        return execution_pipeline, pipeline_algorithms

    @staticmethod
    def _recover_array(data, var_names_list):
        """
        Recover a structured array from a list of variables

        Parameters:
        -----------
        data: np.ndarray
            data array

        var_names_list: List[str]
            list of variable names

        Returns:
        --------
        recovered_data: np.ndarray
            structured array
        """
        variables_list = list()

        for var in var_names_list:
            variables_list.append(data[var])

        variables_str = ",".join(var_names_list)
        formats_str = ",".join(len(var_names_list) * ["f8"])

        recovered_data = np.core.records.fromarrays(
            variables_list, names=variables_str, formats=formats_str
        )

        return recovered_data

    def _slice_by_interval(self, batch):
        """
        Slice the data by interval

        Parameters:
        -----------
        batch: List[int]
            list of batch indices

        Returns:
        --------
        batch: slice
            slice object
        """
        return slice(*batch)

    def _slice_by_set(self, batch):
        """
        slice the data by set

        Parameters:
        ----------
        batch: List[int]
            list of batch indices

        Returns:
        --------
        batch: List[int]
            list of batch indices
        """
        return batch

    def _data_preparer_wrapper(self, data_preparer):
        """
        Wrapper for the data preparer

        Parameters:
        -----------
        data_preparer: DataPreparer
            data preparer object

        Returns:
        --------
        result: np.ndarray
            prepared data

        Raises:
        -------
        raise Exception: Exception
            if the data preparer is not an instance of DataPreparer

        raise Exception: Exception
            if the data preparer is not a list of DataPreparer

        raise Exception: Exception
            if the data preparer is not an instance of DataPreparer
        """

        if isinstance(data_preparer, DataPreparer):
            data_preparers = [data_preparer]

        elif isinstance(data_preparer, list):
            data_preparers = data_preparer
            assert all(
                [
                    isinstance(data_preparer, DataPreparer)
                    for data_preparer in data_preparer
                ]
            )
        else:
            raise Exception("Error! data_preparer is not an instance of data_preparer")

        self.data_preparer = data_preparers

        data = getattr(self, "input_data")

        for data_preparer in self.data_preparer:
            self.input_data = data_preparer.prepare_input_data(data)
            if isinstance(self.target_data, np.ndarray):
                target_data = getattr(self, "target_data")
                self.target_data = data_preparer.prepare_input_data(target_data)
            else:
                pass

    def _get_operator(self, operator):
        """
        Get the operator

        Parameters:
        -----------
        operator: Operator
            operator object

        Returns:
        --------
        wrapper: function
            wrapper function
        """
        if self.normalization:

            def wrapper(data):
                evaluation = operator(data)

                output_dict = self.normalization.apply_descaling(
                    map_dict={"target": evaluation}
                )
                evaluation = output_dict["target"]

                output_dict = self.normalization.apply_rescaling(
                    map_dict={"input": evaluation}
                )
                evaluation = output_dict["input"]

                return evaluation

        else:

            def wrapper(data):
                return operator(data)

        return wrapper

    def _model_wrapper(self, model):
        """
        Wrapper for the model

        Parameters:
        -----------
        model: Model
            model object

        Returns:
        --------
        return: None
            model is fitted
        """
        if model.is_this_model_rough:
            model.fit(
                input_data=self.input_data,
                target_data=self.target_data,
                **self.fit_kwargs,
            )

        else:
            pass

        self.model = model

    @staticmethod
    def _get_kwargs(op):
        """
        Get the kwargs of the operator

        Parameters:
        -----------
        op: Operator
            operator object

        Returns:
        --------
        kwargs: List[str]
            list of kwargs
        """
        kwargs = inspect.getfullargspec(op).args
        kwargs.remove("self")

        return kwargs

    def _integration_wrapper(self, integration_op):
        """
        Wrapper for the integration operator

        Parameters:
        -----------
        integration_op: Operator
            operator object

        Returns:
        --------
        return: None
            integration operator is instantiated
        """
        # It checks if the post_process_op is instantiated or not
        self.right_operator = self.model.eval
        for key, var in self.extra_kwargs.items():
            setattr(self, key, var)

        self.initial_state = self.project_data(self.initial_state, self.input_vars_list)

        if inspect.isclass(integration_op):
            # It instantiates the class post_process_op
            init_kwargs_list = self._get_kwargs(integration_op.__init__)
            init_kwargs_dict = {key: getattr(self, key) for key in init_kwargs_list}

            execution_method_str = self.execution_pipeline.get(integration_op)

            postproc_op_instance = integration_op(**init_kwargs_dict)

            execution_method = getattr(postproc_op_instance, execution_method_str)
            exec_kwargs_list = self._get_kwargs(execution_method)
            exec_kwargs_dict = {key: getattr(self, key) for key in exec_kwargs_list}

            # It executes the main method of the instance
            output = execution_method(**exec_kwargs_dict)

            self.output = self.reconstruct_data(data=output)

        else:
            pass

    def _rom_wrapper(self, rom):
        """
        Wrapper for the ROM

        Parameters:
        -----------
        rom: ROM
            ROM object

        Returns:
        --------
        return: None
            ROM is fitted
        """
        # The target data may be previously provided or constructed after the
        # dimensionality reduction
        if self.is_batchwise:
            assert rom.kind == "batchwise", (
                "The ROM chosen is not proper " "to batchwise executions"
            )
        else:
            pass

        input_data = getattr(self, "input_data")
        target_data = getattr(self, "target_data")

        input_data = self._construct_data_array(input_data, self.input_vars_list)

        rom.fit(data=input_data)

        # Apply the conversion to the target data object when that exists
        if self.there_is_model:
            reduced_input_data = rom.project(data=input_data)
            if isinstance(target_data, np.ndarray):
                target_data = self._construct_data_array(
                    target_data, self.target_vars_list
                )
                # Apply the dimensionality reduction to the target data object when that exists
                reduced_target_data = rom.project(data=target_data)

            else:
                reduced_target_data = self.data_generator(data=reduced_input_data)

            self.target_data = reduced_target_data
            self.input_data = reduced_input_data
        else:
            pass

    def _normalization_wrapper(self, normalization_op):
        """
        normalization wrapper

        Parameters:
        -----------
        normalization_op: Operator
            operator object

        Returns:
        --------
        return: None
            normalization operator is instantiated
        """
        map_dict = dict()
        if isinstance(self.input_data, np.ndarray):
            map_dict.update({"input": self.input_data})
        else:
            pass

        if isinstance(self.target_data, np.ndarray):
            map_dict.update({"target": self.target_data})
        else:
            pass

        transformed_array_dict = normalization_op.rescale(map_dict=map_dict)

        if isinstance(self.input_data, np.ndarray):
            self.input_data = transformed_array_dict["input"]
        else:
            pass

        if isinstance(self.target_data, np.ndarray):
            self.target_data = transformed_array_dict["target"]
        else:
            pass

    def _batchwise_projection(
        self,
        data=None,
        variables_list=None,
        data_interval=None,
        batch_size=None,
        batch_indices=None,
    ):
        """
        batchwise projection

        Parameters:
        -----------
        data: np.ndarray
            data array
        variables_list: List[str]
            list of variables
        data_interval: Tuple[int, int]
            data interval
        batch_size: int
            batch size
        batch_indices: List[int]
            list of batch indices

        Returns:
        --------
        batch_list: List[np.ndarray]
            list of batches

        Raises:
        -------
        raise: Exception
            if there is a contradiction
        """

        if data_interval is not None:
            n_samples = data_interval[1] - data_interval[0]
            slicer = self._slice_by_interval

        elif batch_indices is not None:
            n_samples = len(batch_indices)
            slicer = self._slice_by_set

        else:
            raise Exception(
                "There is a contradiction. Or data_interval or batch_indices must be provided."
            )

        if isinstance(batch_size, simulai.metrics.MemorySizeEval):
            batch_size = batch_size(
                max_batches=n_samples, shape=data.shape
            )  # TODO data.shape[1:]

        elif batch_size == -1:
            batch_size = n_samples
        else:
            pass

        batches = batchdomain_constructor(
            data_interval=data_interval,
            batch_size=batch_size,
            batch_indices=batch_indices,
        )

        batches_list = list()
        for batch_id, batch in enumerate(batches):
            chunk_array = data[slicer(batch)]

            print(
                f"Projecting for the batch {batch_id+1}/{len(batches)} batch_size={chunk_array.shape[0]}"
            )

            if self.data_preparer:
                data_ = self.data_preparer.prepare_input_data(chunk_array)
            else:
                data_ = chunk_array

            data_numeric = self._construct_data_array(data_, variables_list)

            batches_list.append(self.rom.project(data_numeric))

        return np.vstack(batches_list)

    # Reconstructing using chunks of data in order to save memory
    def _batchwise_reconstruction(
        self,
        data=None,
        variables_list=None,
        data_interval=None,
        batch_size=None,
        batch_indices=None,
        dump_path=None,
    ):
        """
        batchwise reconstruction

        Parameters:
        -----------
        data: np.ndarray
            data array
        variables_list: List[str]
            list of variables
        data_interval: Tuple[int, int]
            data interval
        batch_size: int
            batch size
        batch_indices: List[int]
            list of batch indices
        dump_path: str
            path to dump the reconstruction

        Raises:
        -------
        rise: Exception
            if there is a contradiction
        """
        assert dump_path, (
            "It is necessary to provide a path to save the reconstruction"
            "output to a HDF5 file."
        )

        if os.path.exists(dump_path):
            warnings.warn(
                f"Reconstruction dump_path={dump_path} exists. It will be overwritten"
            )

        data_file = h5py.File(dump_path, "w")

        if data_interval is not None:
            samples_dim = data_interval[1] - data_interval[0]
            slicer = self._slice_by_interval

        elif batch_indices is not None:
            samples_dim = len(batch_indices)
            slicer = self._slice_by_set

        else:
            raise Exception(
                "There is a contradiction. Or data_interval or batch_indices must be provided."
            )

        # In case of using a memory limiter, it is necessary to evaluate the batch_size
        # using it
        if isinstance(batch_size, simulai.metrics.MemorySizeEval):
            batch_size = batch_size(
                max_batches=samples_dim, shape=(self.data_preparer.n_features,)
            )

        elif batch_size == -1:
            batch_size = samples_dim
        else:
            pass

        # Constructing the chunks intervals
        batches = batchdomain_constructor(
            data_interval=data_interval,
            batch_size=batch_size,
            batch_indices=batch_indices,
        )

        # If the data structure is a structured numpy array a list of variables is provided
        data_shape = (samples_dim,) + self.data_preparer.collapsible_shapes

        dset = data_file.create_dataset(
            "reconstructed_data", shape=data_shape, dtype=self.data_preparer.dtype
        )

        # Batchwise reconstruction loop
        for batch_id, batch in enumerate(batches):
            chunk_array = data[slicer(batch)]
            print(
                f"Reconstruction for the batch {batch_id+1}/{len(batches)} batch_size={chunk_array.shape[0]}"
            )

            data_numeric = self.rom.reconstruct(chunk_array)
            output_data_ = self.data_preparer.prepare_output_data(data_numeric)
            dset[slicer(batch)] = output_data_

        return dset

    def project_data(
        self,
        data=None,
        variables_list=None,
        data_interval=None,
        batch_size=1,
        batch_indices=None,
    ):
        """
        project data

        Parameters:
        -----------
        data: np.ndarray
            data array
        variables_list: List[str]
            list of variables
        data_interval: Tuple[int, int]
            data interval
        batch_size: int
            batch size
        batch_indices: List[int]
            list of batch indices

        Returns:
        --------
        return: np.ndarray
            projected data

        Raises:
        -------
        rise: Exception
            Data format not supported
        """

        if isinstance(data, np.ndarray):
            if variables_list:
                data_ = self.data_preparer.prepare_input_structured_data(data)
                data_numeric = self._construct_data_array(data_, variables_list)

                return self.rom.project(data_numeric)
            else:
                data_ = self.data_preparer.prepare_input_data(data)
                data_numeric = self._construct_data_array(data_, variables_list)
                return self.rom.project(data_numeric)

        elif isinstance(data, h5py.Dataset):
            assert data_interval, (
                "In using a h5py Dataset it is necessary" "to provide a data interval"
            )

            return self._batchwise_projection(
                data=data,
                variables_list=variables_list,
                data_interval=data_interval,
                batch_size=batch_size,
                batch_indices=batch_indices,
            )

        else:
            raise Exception(
                "Data format not supported. It must be np.ndarray" "or h5py.Dataset."
            )

    def reconstruct_data(
        self,
        data=None,
        variables_list=None,
        data_interval=None,
        batch_size=1,
        dump_path=None,
    ):
        """
        Reconstruction of data

        Parameters:
        -----------
        data: np.ndarray
            data array
        variables_list: List[str]
            list of variables
        data_interval: Tuple[int, int]
            data interval
        batch_size: int
            batch size
        dump_path: str
            path to dump the reconstruction

        Returns:
        --------
        Return: np.ndarray
            reconstructed data

        Raises:
        -------
        rise: Exception
            Data format not supported
        """

        if isinstance(data, np.ndarray) and not data_interval:
            print("Applying the global reconstruction strategy.")

            data_numeric = self.rom.reconstruct(data)

            return self.data_preparer.prepare_output_data(data_numeric)

        elif isinstance(data, np.ndarray) and data_interval and dump_path:
            print("Applying the batch-wise reconstruction strategy.")

            return self._batchwise_reconstruction(
                data=data,
                variables_list=variables_list,
                data_interval=data_interval,
                batch_size=batch_size,
                dump_path=dump_path,
            )

        else:
            raise Exception(
                "Data format not supported. It must be np.ndarray" "or h5py.Dataset."
            )

    def pipeline_loop(self, input_data, target_data, reference_data, extra_kwargs):
        """
        pipeline loop

        Parameters:
        -----------
        input_data: np.ndarray
            input data
        target_data: np.ndarray
            target data
        reference_data: np.ndarray
            reference data
        extra_kwargs: dict
            extra kwargs

        Returns:
        --------
        return: None
        """
        self.input_data = input_data
        self.target_data = target_data
        self.reference_data = reference_data
        self.extra_kwargs = extra_kwargs

        # These operations are considered already instantiated
        # it is also necessary to comprise the no instantiated cases
        for op, method_name in self.execution_pipeline.items():
            # op can be a class instance or a class itself

            wrapper_method = self.pipeline_algorithms[op.purpose]["wrapper"]
            wrapper_method(op)

            setattr(self, op.purpose, op)

            print(f"Executed operation {op.name.upper()}.")

    def batchwise_pipeline_loop(
        self,
        input_data,
        target_data,
        reference_data,
        extra_kwargs,
        data_interval=None,
        batch_size=None,
        batch_indices=None,
    ):
        """
        implementation of the batchwise pipeline loop

        Parameters:
        -----------
        input_data: np.ndarray
            input data
        target_data: np.ndarray
            target data
        reference_data: np.ndarray
            reference data
        extra_kwargs: dict
            extra kwargs
        data_interval: Tuple[int, int]
            data interval
        batch_size: int
            batch size
        batch_indices: List[int]
            list of batch indices

        Returns:
        --------
        return: None

        Asserts:
        --------
        rise: AssertionError
            There is a contradiction. Or data_interval or batch_indices must be provided.
        """
        self.input_data = input_data
        self.target_data = target_data
        self.reference_data = reference_data
        self.extra_kwargs = extra_kwargs

        if self.there_is_target:
            error_message = (
                "The input and target dimensions are not compatible"
                "with {} and {} samples".format(self.input_data.shape, self.target_data)
            )

            assert self.input_data.shape == self.target_data, error_message
        else:
            pass

        # Checking up if a list of batches was provided or if it is necessary to construct it
        assert (
            data_interval is not None or batch_indices is not None
        ), "There is a contradiction. Or data_interval or batch_indices must be provided."

        batches = batchdomain_constructor(
            data_interval=data_interval,
            batch_size=batch_size,
            batch_indices=batch_indices,
        )

        for batch_id, batch in enumerate(batches):
            self.input_data = input_data[self.slicer(batch)]

            print(
                f"Executing the mini-batch {batch_id+1}/{len(batches)} batch_size={self.input_data.shape[0]}"
            )

            if self.there_is_target:
                self.target_data = target_data[self.slicer(batch)]
            else:
                pass

            if self.there_is_reference:
                self.reference_data = reference_data[self.slicer(batch)]
            else:
                pass

            self.extra_kwargs = extra_kwargs

            # These operations are considered already instantiated
            # it is also necessary to comprise the no instantiated cases
            for op, method_name in self.execution_pipeline.items():
                # op can be a class instance or a class itself

                wrapper_method = self.pipeline_algorithms[op.purpose]["wrapper"]
                wrapper_method(op)

                setattr(self, op.purpose, op)

                print("Executed operation.")

    def exec(
        self,
        input_data=None,
        target_data=None,
        reference_data=None,
        data_generator=None,
        extra_kwargs=None,
        fit_kwargs=None,
        data_interval=None,
        batch_size=None,
        batch_indices=None,
    ):
        """
        implementation of the exec method

        Parameters:
        -----------
        input_data: np.ndarray
            input data
        target_data: np.ndarray
            target data
        reference_data: np.ndarray
            reference data
        data_generator: DataGenerator
            data generator
        extra_kwargs: dict
            extra kwargs
        fit_kwargs: dict
            fit kwargs
        data_interval: Tuple[int, int]
            data interval
        batch_size: int
            batch size
        batch_indices: List[int]
            list of batch indices

        Returns:
        --------
        return: None

        Raises:
        -------
        rise: Exception
            There is a contradiction. Or data_interval or batch_indices must be provided.
        raise: Exception
            Data format not supported. It must be np.ndarray or h5py.Dataset.
        """
        data_format = list()

        self.fit_kwargs = fit_kwargs

        if isinstance(input_data, np.ndarray):
            if input_data.dtype.names:
                self.input_vars_list = list(input_data.dtype.names)
            else:
                pass

            data_format.append("numpy")

        elif isinstance(input_data, h5py.Dataset):
            if data_interval is not None:
                n_samples = data_interval[1] - data_interval[0]
                self.slicer = self._slice_by_interval

            elif batch_indices is not None:
                n_samples = len(batch_indices)
                self.slicer = self._slice_by_set
            else:
                raise Exception(
                    "There is a contradiction. Or data_interval or batch_indices must be provided."
                )

            if isinstance(batch_size, simulai.metrics.MemorySizeEval):
                batch_size = batch_size(
                    max_batches=n_samples, shape=input_data.shape
                )  # TODO input_data.shape[1:]

            elif batch_size == -1:
                batch_size = n_samples
            else:
                pass

            if input_data.dtype.names:
                self.input_vars_list = list(input_data.dtype.names)
            else:
                pass

            assert batch_size, (
                "The argument batch_size must be" "provided when using HDF5 as input"
            )

            data_format.append("hdf5")

        else:
            data_format.append(None)
            raise Exception("This data format is not supported.")

        # When a machine-learning (or other fitting method) is employed
        # target data, or a method for generating it, must be provided
        if self.there_is_model:
            if isinstance(target_data, np.ndarray):
                self.target_vars_list = list(target_data.dtype.names)
                data_format.append("numpy")

            elif isinstance(target_data, h5py.Dataset):
                assert batch_size, (
                    "The argument batch_size must be"
                    "provided when using HDF5 as input"
                )
                self.target_vars_list = [
                    "var_" + str(ii) for ii in range(target_data.shape[1])
                ]
                data_format.append("numpy")
            else:
                assert data_generator
                self.data_generator = data_generator
                data_format.append("numpy")

            self.there_is_target = True

        else:
            pass

        # This way of executing the pipeline is used when the data are ingested
        # at a time in a Numpy array

        data_format = set(data_format)
        assert len(data_format) == 1, "Incompatible input and output formats"

        data_format = list(data_format)[0]

        if data_format == "numpy":
            print("Executing a global pipeline.")
            self.pipeline_loop(input_data, target_data, reference_data, extra_kwargs)

        # For ingestion via HDF5, the pipeline process is repeated for each
        # mini-batch
        elif data_format == "hdf5":
            print("Executing a batchwise pipeline.")
            self.is_batchwise = True
            self.batchwise_pipeline_loop(
                input_data,
                target_data,
                reference_data,
                extra_kwargs,
                data_interval=data_interval,
                batch_size=batch_size,
                batch_indices=batch_indices,
            )

        else:
            raise Exception("The data format was not understood")

    def eval(self, data=None, with_projection=True, with_reconstruction=True):
        """
        evaluation method is used to evaluate the model

        Parameters:
        -----------
        data: np.ndarray
            data
        with_projection: bool
            with projection
        with_reconstruction: bool
            with reconstruction

        Returns:
        --------
        return: np.ndarray
            evaluation of the model
        """
        if self.rom and with_projection:
            data_ = self.project_data(data, self.input_vars_list)
        else:
            data_ = data

        if self.normalization:
            output_dict = self.normalization.apply_rescaling(map_dict={"input": data_})
            data_ = output_dict["input"]
        else:
            pass

        evaluation_ = self.model.eval(data_)

        if self.normalization:
            output_dict = self.normalization.apply_descaling(
                map_dict={"target": evaluation_}
            )
            evaluation_ = output_dict["target"]
        else:
            pass

        if with_reconstruction:
            evaluation = self.reconstruct_data(data=evaluation_)
        else:
            evaluation = evaluation_

        return evaluation

    def predict(
        self, post_process_op=None, extra_kwargs=None, with_reconstruction=True
    ):
        """
        predict method is used to predict the model

        Parameters:
        -----------
        post_process_op: object
            post process operator
        extra_kwargs: dict
            extra keyword arguments
        with_reconstruction: bool
            with reconstruction

        Returns:
        --------
        return: np.ndarray
            prediction of the model
        """
        # It checks if the post_process_op is instantiated or not

        initial_state = extra_kwargs["initial_state"]
        extra_kwargs_copy = copy.copy(extra_kwargs)

        if self.rom:
            extra_kwargs_copy["initial_state"] = self.project_data(
                initial_state, self.input_vars_list
            )
        else:
            pass

        if self.normalization:
            initial_state = extra_kwargs_copy["initial_state"]
            output_dict = self.normalization.apply_rescaling(
                map_dict={"input": initial_state}
            )
            extra_kwargs_copy["initial_state"] = output_dict["input"]

        else:
            pass

        recurrent_operator = self._get_operator(self.model.eval)
        op_instance = post_process_op(recurrent_operator)
        output = op_instance(**extra_kwargs_copy)

        if self.normalization:
            output_dict = self.normalization.apply_descaling(map_dict={"input": output})
            output = output_dict["input"]
        else:
            pass

        if with_reconstruction:
            output = self.reconstruct_data(data=output)
        else:
            pass

        return output

    def _save(self, save_path=None, model_name=None):
        """
        _save method is used to save the model

        Parameters:
        -----------
        save_path: str
            save path
        model_name: str
            model name

        Returns:
        --------
        return: None
        """
        fp = open(os.path.join(save_path, model_name), "wb")

        try:
            print(f"Trying to save {self} to a file.")
            pickle.dump(self, fp, protocol=4)

        except Exception as e:
            print("START ---- Exception message")
            print(e, e.args)
            print("END ---- Exception message")
            print(f"Object {self} is not pickable." f" Trying to save with another way")

    def save(self, save_path=None, model_name=None):
        """
        Save method is used to save the model

        Parameters:
        -----------
        save_path: str
            save path
        model_name: str
            model name

        Returns:
        --------
        return: None

        Raises:
        -------
        AssertionError
            model is not an attribute of self.model
        """
        try:
            self._save(save_path=save_path, model_name=model_name)
        except Exception:
            assert self.model is not None, f"model is not an attribute of {self.model}"

            self.model.save(save_path, model_name)

    def test(self, metric=None, data=None):
        """
        test method is used to test the model

        Parameters:
        -----------
        metric: object
            metric
        data: np.ndarray
            data

        Returns:
        --------
        error: float
            error
        """
        data = self._construct_data_array(data, self.input_vars_list)

        error = metric(data, self.output)

        return error
