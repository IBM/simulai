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

import warnings

MPI_GLOBAL_AVAILABILITY = True

try:
    from mpi4py import MPI
except:
    MPI_GLOBAL_AVAILABILITY = False
    warnings.warn(f'Trying to import MPI in {__file__}.')
    warnings.warn('mpi4py is not installed. If you want to execute MPI jobs, we recommend you install it.')

# Pipeline for executing independent MPI jobs
class PipelineMPI:

    def __init__(self, exec: callable=None, extra_params:dict=None, collect:bool=None, show_log:bool=True) -> None:

        self.exec = exec
        self.show_log = show_log

        if extra_params is not None:
            self.extra_params = extra_params
        else:
            self.extra_params = {}

        self.collect = collect

        self.comm = MPI.COMM_WORLD
        self.n_procs = self.comm.Get_size()

        self.status = (self.n_procs - 1)*[False]
        self.status_dict = dict()

    # Check if the provided datasets
    def _check_kwargs_consistency(self, kwargs: dict=None) -> int:

        types = [type(value) for value in kwargs.values()]
        lengths = [len(value) for value in kwargs.values()]

        assert all([t==list for t in types]), f"All the elements in kwargs must be list," \
                                              f" but received {types}."

        assert len(set(lengths)) == 1, f"All the elements in kwargs must be the same length," \
                                       f" but received {lengths}"

        print("kwargs is alright.")

        return lengths[0]

    # The workload can be executed serially in each worker node
    def _split_kwargs(self, kwargs:dict, rank:int, size:int, total_size:int) -> (dict, int):

        # Decrement rank and size by 1, because they are usually 0-indexed in Python
        size -= 1
        rank -= 1

        # Calculate batch size and remainder using divmod() function
        batch_size, remainder = divmod(total_size, size)

        # If rank is less than remainder, calculate kwargs_batch using batch size + 1
        if rank < remainder:
            kwargs_batch = {key: value[rank*(batch_size+1):(rank+1)*(batch_size+1)] for key, value in kwargs.items()}
            return kwargs_batch, batch_size+1
        # If rank is not less than remainder, calculate kwargs_batch using batch size
        else:
         kwargs_batch = {key: value[remainder*(batch_size+1) + (rank-remainder)*batch_size:(rank-remainder+1)*batch_size] for key, value in kwargs.items()}

        return kwargs_batch, batch_size


    def _attribute_dict_output(self, dicts:list=None) -> None:

        root = dict()
        for e in dicts:
            root.update(e)

        for key, value in root.items():
            self.status_dict[key] = value

    @staticmethod
    def inner_type(obj: list=None):

        types_list = [type(o) for o in obj]
        assert len(set(types_list)) == 1, "Composed types are not supported."

        return types_list[0]

    def _exec_wrapper(self, kwargs:dict, total_size:int) -> None:

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        size_ = size

        # Rank 0 is the 'master' node
        # The worker nodes execute their workload and send a message to
        # master

        if rank != 0:

            print(f"Executing rank {rank}.")
            kwargs_batch, batch_size = self._split_kwargs(kwargs, rank , size_, total_size)

            kwargs_batch_list = [{key:value[j] for key, value in kwargs_batch.items()}
                                               for j in range(batch_size)]

            out = list()
            for i in kwargs_batch_list:

                print(f"Executing batch {i['key']} in rank {rank}")
                # Concatenate the rank to the extra parameters
                i.update(self.extra_params)
                # Appending the result of the operation self.exec to the partial list
                out.append(self.exec(**i))

            if self.collect is True:
                msg = out
            else:
                msg = 1

            if self.show_log:
                print(f"Sending the output {msg} to rank 0")

            comm.send(msg, dest=0)

            print(f"Execution concluded for rank {rank}.")

        # The master awaits the responses of each worker node
        elif rank == 0:

            for r in range(1, size):

                msg = comm.recv(source=r)
                self.status[r - 1] = msg

                if self.inner_type(msg) == dict:

                    self._attribute_dict_output(dicts=msg)

                if self.show_log:
                    print(f"Rank 0 received {msg} from rank {r}")

        comm.barrier()

    @property
    def success(self):
        return all(self.status)

    def run(self, kwargs:dict=None) -> None:

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        total_size = 0

        # Checking if the datasets dimensions are in accordance with the expected ones
        if rank == 0:
            total_size = self._check_kwargs_consistency(kwargs=kwargs)

        total_size = comm.bcast(total_size, root=0)

        comm.barrier()

        # Executing a wrapper containing the parallelized operation
        self._exec_wrapper(kwargs, total_size)

        comm.barrier()

