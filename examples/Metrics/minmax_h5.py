from simulai.metrics import MinMaxEvaluation

import h5py
data_path = '/tmp/snapshots_short.h5'

fp = h5py.File(data_path, "r")
dataset = fp.get("tasks")
N = dataset["u"].shape[0]

minmax = MinMaxEvaluation()
max_value, min_value = minmax.eval_h5(dataset=dataset, data_interval=[0, 4000], keys=['u', 'w', 'T'], batch_size=1_00, axis=None)
