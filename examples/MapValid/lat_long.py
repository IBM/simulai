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
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

parser = ArgumentParser(description="Reading input arguments")
parser.add_argument("--data_path", type=str, help="The path tot the datasets.")

#  Reading input arguments
args = parser.parse_args()

data_path = args.data_path

root = Dataset(data_path, "r")

lat = root["lat"][:]
long = root["lon"][:]

Lat, Long = np.meshgrid(lat, long, indexing="ij")

grid_filename = os.path.join(os.path.dirname(data_path), "nldas_grid.npz")
np.savez(grid_filename, Lat=Lat, Long=Long)
