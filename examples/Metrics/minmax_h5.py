from simulai.metrics import MinMaxEvaluation

import h5py
data_path = '/tmp/snapshots_short.h5'

fp = h5py.File(data_path, "r")
dataset = fp.get("tasks")
N = dataset["u"].shape[0]

minmax = MinMaxEvaluation()
minmax
