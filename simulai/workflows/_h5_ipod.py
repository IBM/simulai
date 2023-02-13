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

import contextlib
import json
import os
import pickle

import h5py
import numpy as np

from simulai.io import MemorySizeEval, Reshaper, Sampling
from simulai.metrics import MeanEvaluation
from simulai.rom import IPOD
from simulai.simulation import Pipeline
from simulai.utilities import make_temp_directory, makedirs_to_file
from simulai.workflows import compute_datasets_to_reference_norm


def construct_IPOD_pipeline(
    dataset,
    n_components=10,
    batch_size=100,
    train_interval=None,
    sampling_choices_fraction=1.0,
    sampler_save_path=None,
    data_preparer=None,
):
    with contextlib.ExitStack() as stack:
        if sampler_save_path is None:
            save_path = stack.enter_context(make_temp_directory())

        dump_sampler = os.path.join(save_path, f"exec_data_sampled.h5")
        sampler = Sampling(choices_fraction=sampling_choices_fraction, shuffling=True)
        sampler.prepare_input_structured_data(
            dataset,
            data_interval=train_interval,
            batch_size=batch_size,
            dump_path=dump_sampler,
        )

        with h5py.File(dump_sampler, "r") as fp_sampled:
            sampled_dataset = fp_sampled.get("data")

            mean = MeanEvaluation()(
                dataset=sampled_dataset,
                data_interval=[0, sampled_dataset.shape[0]],
                batch_size=batch_size,
                data_preparer=data_preparer,
            )

            rom_config = {"n_components": n_components, "mean_component": True}

            if data_preparer is None:
                data_preparer = Reshaper()

            pipeline = Pipeline(
                stages=[
                    ("data_preparer", data_preparer),
                    ("rom", IPOD(config=rom_config, data_mean=mean)),
                ],
            )

            pipeline.exec(
                input_data=sampled_dataset,
                data_interval=[0, sampled_dataset.shape[0]],
                batch_size=batch_size,
            )

    return pipeline


def pipeline_project_then_reconstruct(
    dataset,
    pipeline=None,
    reconstructed_dump_path=None,
    batch_size=100,
    project_interval=None,
    variables_names=None,
):
    if project_interval is None:
        project_interval = [0, dataset.shape[0]]

    if variables_names is None:
        variables_names = list(dataset.dtype.names)

    # The projected data usually can be allocated in memory
    projected = pipeline.project_data(
        data=dataset,
        data_interval=project_interval,
        variables_list=variables_names,
        batch_size=batch_size,
    )

    if reconstructed_dump_path is not None:
        pipeline_reconstruct(
            projected_data=projected,
            variables_names=variables_names,
            pipeline=pipeline,
            reconstructed_dump_path=reconstructed_dump_path,
            batch_size=batch_size,
        )

    return projected


def pipeline_reconstruct(
    projected_data,
    variables_names,
    pipeline,
    reconstructed_dump_path,
    n_modes=None,
    data_interval=None,
    batch_size=None,
):
    makedirs_to_file(reconstructed_dump_path)

    if data_interval is None:
        data_interval = [0, projected_data.shape[0]]

    if n_modes is not None:
        n_modes = min(n_modes, projected_data.shape[1])
        projected_data = projected_data[:, 0:n_modes]

    if batch_size is None:
        batch_size = 100

    pipeline.reconstruct_data(
        data=projected_data,
        data_interval=data_interval,
        variables_list=variables_names,
        batch_size=batch_size,
        dump_path=reconstructed_dump_path,
    )


def dataset_ipod(
    data_path: str = None,
    data_key: str = None,
    train_interval=None,
    train_interval_fraction: float = 0.9,
    sampling_fraction: float = 1.0,
    save_path: str = None,
    n_components: int = None,
    batch_size: int = None,
    memory_percent: float = 0.1,
    reconstructed_dump_path=None,
    project_reconstruct_interval="full_interval",
    error_norm_dump_path=None,
    data_preparer=None,
):
    if isinstance(batch_size, int):
        batch_sizer = batch_size
    else:
        memory_limiter = MemorySizeEval(memory_tol_percent=memory_percent)
        batch_sizer = memory_limiter

    with h5py.File(data_path, "r") as fp:
        dataset = fp.get(data_key)

        n_samples = dataset.shape[
            0
        ]  # Total number of samples (time-steps) in the dataset
        n_samples_train = int(
            train_interval_fraction * n_samples
        )  # Rounded value of samples (used for practical purposes)

        if train_interval is None:
            train_interval = [0, n_samples_train]

        if project_reconstruct_interval is None:
            project_reconstruct_interval = [n_samples_train, n_samples]
        elif (
            isinstance(project_reconstruct_interval, str)
            and project_reconstruct_interval == "full_interval"
        ):
            project_reconstruct_interval = [0, n_samples]
        elif (
            isinstance(project_reconstruct_interval, str)
            and project_reconstruct_interval == "test_interval"
        ):
            project_reconstruct_interval = [n_samples_train, n_samples]
        elif (
            isinstance(project_reconstruct_interval, list)
            and len(project_reconstruct_interval) == 2
        ):
            pass
        else:
            raise RuntimeError(
                f"Invalid option project_reconstruct_interval={project_reconstruct_interval}"
            )

        pipeline = construct_IPOD_pipeline(
            dataset,
            n_components=n_components,
            batch_size=batch_sizer,
            train_interval=train_interval,
            sampling_choices_fraction=sampling_fraction,
            data_preparer=data_preparer,
        )

        dataset_name = os.path.splitext(os.path.basename(data_path))[0]
        pipeline.save(
            save_path=save_path, model_name=f"pipeline_{dataset_name}_{data_key}.pkl"
        )

        projected = pipeline_project_then_reconstruct(
            dataset,
            pipeline=pipeline,
            reconstructed_dump_path=reconstructed_dump_path,
            batch_size=batch_sizer,
            project_interval=project_reconstruct_interval,
        )

        np.save(
            os.path.join(save_path, f"time_series_{dataset_name}_{data_key}.npy"),
            projected,
        )

        if reconstructed_dump_path is not None and error_norm_dump_path is not None:
            with h5py.File(reconstructed_dump_path, "r") as fpr:
                reconstructed = fpr.get("reconstructed_data")
                norm_to_test_reference = compute_datasets_to_reference_norm(
                    reconstructed,
                    dataset,
                    reference_data_interval=project_reconstruct_interval,
                    batch_size=batch_sizer,
                )

                makedirs_to_file(error_norm_dump_path)
                with open(error_norm_dump_path, "w") as fn:
                    json.dump(norm_to_test_reference, fn, indent=0)


def pipeline_projection_error(
    data_path: str = None,
    key: str = "ssh",
    pipeline_path: str = None,
    projected_data_path: str = None,
    projected_data_interval=None,
    n_sub_components: int = None,
    energy_preserved: float = None,
    memory_percent: float = 0.1,
    reconstructed_dump_path: str = None,
    save_path: str = None,
    error_filename: str = None,
    scaled: bool = False,
):
    batch_sizer = MemorySizeEval(memory_tol_percent=memory_percent)

    with h5py.File(data_path, "r") as fp:
        dataset = fp.get(key)

        with contextlib.ExitStack() as stack:
            if reconstructed_dump_path is None or os.path.isdir(
                reconstructed_dump_path
            ):
                projected = np.load(projected_data_path)
                variables_names = list(dataset.dtype.names)
                with open(pipeline_path, "rb") as f_pipeline:
                    pipeline = pickle.load(f_pipeline)
                if scaled:
                    projected = projected * np.reshape(
                        np.sqrt(pipeline.rom.singular_values[0 : projected.shape[1]]),
                        [1, -1],
                    )

                if n_sub_components is not None:
                    pass
                elif energy_preserved is not None:
                    cum_explained_variance_ratio = np.cumsum(
                        pipeline.rom.pca.explained_variance_ratio_
                    )
                    keep_modes = cum_explained_variance_ratio >= energy_preserved
                    if not np.any(keep_modes):
                        raise RuntimeError(
                            f"Maximum explained energy is {cum_explained_variance_ratio[-1]}. "
                            f"Impossible to preserve {energy_preserved} energy"
                        )
                    n_sub_components = (
                        np.argmax(cum_explained_variance_ratio > energy_preserved) + 1
                    )

                if reconstructed_dump_path is None:
                    reconstructed_dump_path = stack.enter_context(make_temp_directory())

                dataset_name = os.path.splitext(os.path.basename(data_path))[0]
                reconstructed_dump_path = os.path.join(
                    reconstructed_dump_path,
                    f"reconstructed_{n_sub_components}_{dataset_name}_{key}.h5",
                )

                print(f"Number of sub-components used: {n_sub_components}")

                pipeline_reconstruct(
                    projected,
                    variables_names,
                    pipeline,
                    reconstructed_dump_path,
                    n_modes=n_sub_components,
                    batch_size=batch_sizer,
                )
            else:
                print(f"Reusing reconstructed file={reconstructed_dump_path}")

            with h5py.File(reconstructed_dump_path, "r") as fr:
                reconstructed = fr.get("reconstructed_data")
                reconstruction_error = compute_datasets_to_reference_norm(
                    data=reconstructed,
                    reference_data=dataset,
                    reference_data_interval=projected_data_interval,
                    batch_size=batch_sizer,
                )

                if error_filename is None:
                    append_name = os.path.splitext(
                        os.path.basename(reconstructed_dump_path)
                    )[0]
                    error_filename = f"error_{append_name}.json"
                json_file = os.path.join(save_path, error_filename)
                makedirs_to_file(json_file)
                with open(json_file, "w") as f:
                    json.dump(reconstruction_error, f)
