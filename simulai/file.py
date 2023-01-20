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

    """Load a pickle file into a Python object.
     
     Parameters:
     -----------
     path : str, optional
         The path to the pickle file.

     Returns:
     -------
     object or None
         The loaded object, if possible. None if the file cannot be loaded
         
     Raises:
     ------
     Exception :
         if the provided path is not a file or cannot be opened
     """

    import pickle
    
    filename = os.path.basename(path)
    file_extension = filename.split('.')[-1]

    if file_extension == "pkl":
         if os.path.isfile(path):
             try:
                 with open(path, "rb") as fp:
                     model = pickle.load(fp)

                 return model
             except:
                 raise Exception(f"The file {path} could not be opened.")
         else:
             raise Exception(f"The file {path} is not a file.")
    else:
        raise Exception(f"The file format {file_extension} is not supported. It must be pickle.")
        

# This class creates a directory containing all the necessary to save and
# restore a NetworkTemplate object
class SPFile:

    def __init__(self, compact:bool=False) -> None:
        """
        Class for handling persistence of Pytorch Module-like objects.
        
        SimulAI Persistency File
        It saves PyTorch Module-like objects in a directory containing the model template and
        its coefficients dictionary
        
         Parameters:
         -----------
         compact : bool, optional
             Compress the directory to a tar file or not. Default : False
    
        """
        self.compact = compact

    def _leading_size(self, first_line:str=None) -> int:
        """
         Returns the number of leading white spaces in the given line
         
         Parameters:
         -----------
         first_line : str, optional
             The line for which to find the number of leading white spaces.
         
         Returns:
         --------
         int
             number of leading white spaces.
         """
        leading_whitespaces = len(first_line) - len(first_line.lstrip())
        return leading_whitespaces


    def _process_code(self, code:str=None) -> str:
        """
         Returns the code string with leading white spaces removed from each line
         
         Parameters:
         -----------
         code : str, optional
             The code string which to remove the leading whitespaces
         
         Returns:
         --------
         str
             The code string with leading white spaces removed.
        """
        code_lines = code.split('\n')
        first_line = code_lines[0]
        leading_size = self._leading_size(first_line=first_line)

        code_lines_ = [item[leading_size:] for item in code_lines]

        return '\n'.join(code_lines_)

    def write(self, save_dir:str=None, name:str=None,
                    template:callable=None, model:NetworkTemplate=None, device:str=None) -> None:

        """
        Writes the model and its instantiating function to a directory.
         
         Parameters:
         -----------
         save_dir : str, optional
             The absolute directory path to save the model
         name : str, optional
             A name for the model.
         template : callable, optional
             A function for instantiating the model.
         model : NetworkTemplate, optional
             The model to be saved.
         device : str, optional
             The device on which the model is located (gpu or cpu).
         
         Returns:
         --------
         None
        """
        model_dir = os.path.join(save_dir, name)

        # Saving the template code
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        template_filename = os.path.join(model_dir, name+'_template.py')
        tfp = open(template_filename, 'w')

        code = inspect.getsource(template)
        code = self._process_code(code=code)
        tfp.write(code)

        # Saving the model coefficients
        model.save(save_dir=model_dir, name=name, device=device)

    def read(self, model_path:str=None, device:str=None, template_name:str=None) -> NetworkTemplate:
        """
        Reads a model from the specified file path, imports it as a module, and initializes it as an object of the corresponding class.
         
         Parameters:
         -----------
         model_path : str, optional
             Complete path to the model.
         device : str, optional
             Device to load the model onto.
         template_name: str, optional
             Name of the class within the imported module to initialize as an object. 

         Returns:
         --------
         NetworkTemplate (child of torch.nn.Module)
             The model restored to memory.
        """
        name = os.path.basename(model_path)
        save_dir = model_path

        sys.path.append(model_path)

        module = importlib.import_module(name+'_template')

        callables = {attr:getattr(module, attr) for attr in dir(module) if callable(getattr(module, attr))}

        if len(callables) > 1:
            if  template_name is None:
                raise Exception(f"There are {len(callables)} models in the module, please provide a value for name.")
            else:
                Model = callables[template_name]()

        elif len(callables) == 1:
            Model = list(callables.values())[0]()

        else:
            raise Exception("There is no model template in the module.")

        Model.load(save_dir=save_dir, name=name, device=device)

        return Model