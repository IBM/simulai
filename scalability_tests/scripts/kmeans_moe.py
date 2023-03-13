from itertools import groupby
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from examples.utils.lorenz_solver import lorenz_solver
from simulai.math.integration import LSODA, ClassWrapper
from simulai.models import KMeansWrapper, MoEPool
from simulai.regression import DenseNetwork

# Set up
dt = 0.0025
T_max = 100
rho = 28
beta = 8 / 3
beta_str = "8/3"
sigma = 10

# Generating datasets
initial_state = np.array([1, 0, 0])[None, :]

lorenz_data, _, _ = lorenz_solver(
    rho=rho,
    dt=dt,
    T=T_max,
    sigma=sigma,
    initial_state=initial_state,
    beta=beta,
    beta_str=beta_str,
    data_path="on_memory",
    solver="RK45",
)

n_clusters = 5

kmeans = KMeansWrapper(n_clusters=n_clusters)
kmeans.fit(lorenz_data)

config = {
    "layers_units": [50, 50, 50],  # Hidden layers
    "activations": "tanh",
    "input_size": 3,
    "output_size": 10,
    "name": "net",
}

experts_list = list()

for i in range(n_clusters):
    experts_list.append(DenseNetwork(**config))

moe_pool = MoEPool(experts_list=experts_list, gating_network=kmeans, input_size=3, devices="gpu")

estimative = moe_pool.forward(input_data=lorenz_data)
print(moe_pool)
print(estimative.shape)



