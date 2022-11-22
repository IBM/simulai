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
import sys
import inspect
import importlib
from typing import Union

from simulai.templates import NetworkTemplate


def load_pkl(path:str=None) -> Union[object, None]:

    """It loads a pickle file into a Python object

    :param path: path to the pickle file
    :type path: str
    :return: the loaded object, if possible
    :rtype: object, None
    """

    import pickle

    filename = os.path.basename(path)
    ext = filename.split('.')[-1]

    if ext == "pkl":

        try:
            with open(path, "rb") as fp:
                model = pickle.load(fp)

            return model

        except:
            raise Exception(f"The file {path} could not be opened.")

    else:
        raise (f"The file format {ext} is not supported. It must be pickle.")

# This class creates a directory containing all the necessary to save and
# restore a NetworkTemplate object
class SPFile:

    def __init__(self, compact:bool=False) -> None:

        """SimulAI Persistency File
        It saves PyTorch Module-like objects in a directory containing the model template and
        its coefficients dictionary

        :param compact: compact the directory to a tar file or not ?
        :type compact: bool
        :return: nothing
        """

        self.compact = compact

    def _leading_size(self, first_line:str=None) -> int:

        n = len(first_line) - len(first_line.lstrip())

        return n

    def _process_code(self, code:str=None) -> str:

        code_lines = code.split('\n')
        first_line = code_lines[0]
        leading_size = self._leading_size(first_line=first_line)

        code_lines_ = [item[leading_size:] for item in code_lines]

        return '\n'.join(code_lines_)

    def write(self, save_dir:str=None, name:str=None,
                    template:callable=None, model:NetworkTemplate=None, device:str=None) -> None:

        """
        :param save_dir: the absolute directory for the saved model
        :type save_dir: str
        :param name: a name for the model
        :type name: str
        :param template: a function for instantiate a raw version of the model
        :type template: callable
        :param device: the device in which the saved model must be located (gpu or cpu)
        :type device: str
        :returns: nothing
        """

        model_dir = os.path.join(save_dir, name)

        # Saving the template code
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        template_filename = os.path.join(model_dir, name+'_template.py')
        tfp = open(template_filename, 'w')

        code = inspect.getsource(template)
        code_ = self._process_code(code=code)
        tfp.write(code_)

        # Saving the model coefficients
        model.save(save_dir=model_dir, name=name, device=device)

    def read(self, model_path:str=None, device:str=None) -> NetworkTemplate:

        """
        :param model_path: the complete path to the model
        :type model_path: str
        :returns: the model restored to memory
        :rtype: NetworkTemplate (child of torch.nn.Module)
        """

        name = os.path.basename(model_path)
        save_dir = model_path

        sys.path.append(model_path)

        module = importlib.import_module(name+'_template')
        Model = getattr(module, 'model')()

        Model.load(save_dir=save_dir, name=name, device=device)

        return Model


