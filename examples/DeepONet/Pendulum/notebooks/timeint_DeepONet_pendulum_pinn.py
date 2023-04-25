#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import os

import matplotlib.pyplot as plt
import numpy as np

from simulai.file import SPFile
from simulai.optimization import Optimizer
from simulai.residuals import SymbolicOperator

# In[3]:


# #### Basic configuration

# In[4]:


save_path = "."


# In[5]:


Q = 1_000
N = int(5e4)


# In[6]:


initial_state_test = np.array([1, 1])


# In[7]:


t_intv = [0, 1]
s_intv = np.stack([[-3, -3], [3, 3]], axis=0)


# #### The system of ODEs we aim at solving.
# #### Damped Gravitational Pendulum:
# $$
#     \frac{d\,s_1}{d\,t} = s_2\\
#     \frac{d\,s_2}{d\,t} = \frac{-b\,s_2}{m} - \frac{g\,sin(s_1)}{L}
# $$
#

# In[8]:


f_s1 = "D(s1, t) - s2"
f_s2 = "D(s2, t) + b*s2/m + g*sin(s1)/L"


# In[9]:


U_t = np.random.uniform(low=t_intv[0], high=t_intv[1], size=Q)
U_s = np.random.uniform(low=s_intv[0], high=s_intv[1], size=(N, 2))


# In[10]:


np.save("initial_states.npy", U_s)


# In[11]:


branch_input_train = np.tile(U_s[:, None, :], (1, Q, 1)).reshape(N * Q, -1)
trunk_input_train = np.tile(U_t[:, None], (N, 1))


# In[12]:


branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))
trunk_input_test = np.sort(U_t[:, None], axis=0)


# In[13]:


initial_states = U_s


# In[14]:


input_labels = ["t"]
output_labels = ["s1", "s2"]


# In[15]:


n_inputs = len(input_labels)
n_outputs = len(output_labels)


# In[16]:


lambda_1 = 0.0  # Penalty for the L¹ regularization (Lasso)
lambda_2 = 0.0  # Penalty factor for the L² regularization
n_epochs = 300_000  # Maximum number of iterations for ADAM
lr = 1e-3  # Initial learning rate for the ADAM algorithm


# In[17]:


def model():
    from simulai.models import ImprovedDeepONet
    from simulai.regression import SLFNN, ConvexDenseNetwork

    n_latent = 100
    n_inputs_b = 2
    n_inputs_t = 1
    n_outputs = 2

    # Configuration for the fully-connected trunk network
    trunk_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_t,
        "output_size": n_latent * n_outputs,
        "name": "trunk_net",
    }

    # Configuration for the fully-connected branch network
    branch_config = {
        "layers_units": 7 * [100],  # Hidden layers
        "activations": "tanh",
        "input_size": n_inputs_b,
        "output_size": n_latent * n_outputs,
        "name": "branch_net",
    }

    # Instantiating and training the surrogate model
    trunk_net = ConvexDenseNetwork(**trunk_config)
    branch_net = ConvexDenseNetwork(**branch_config)

    encoder_trunk = SLFNN(input_size=n_inputs_t, output_size=100, activation="tanh")
    encoder_branch = SLFNN(input_size=n_inputs_b, output_size=100, activation="tanh")

    # It prints a summary of the network features
    trunk_net.summary()
    branch_net.summary()

    pendulum_net = ImprovedDeepONet(
        trunk_network=trunk_net,
        branch_network=branch_net,
        encoder_trunk=encoder_trunk,
        encoder_branch=encoder_branch,
        var_dim=n_outputs,
        devices="gpu",
        model_id="pendulum_net",
    )

    return pendulum_net


# In[18]:


pendulum_net = model()


# In[19]:


residual = SymbolicOperator(
    expressions=[f_s1, f_s2],
    input_vars=input_labels,
    output_vars=output_labels,
    function=pendulum_net,
    inputs_key="input_trunk",
    constants={"b": 0.05, "g": 9.81, "L": 1, "m": 1},
    device="gpu",
    engine="torch",
)


# In[20]:


penalties = [1, 1]
batch_size = 10_000


# In[21]:


optimizer_config = {"lr": lr}


# In[22]:


input_data = {"input_branch": branch_input_train, "input_trunk": trunk_input_train}


# In[23]:


optimizer = Optimizer(
    "adam",
    params=optimizer_config,
    lr_decay_scheduler_params={
        "name": "ExponentialLR",
        "gamma": 0.9,
        "decay_frequency": 5_000,
    },
)


# In[24]:


params = {
    "lambda_1": lambda_1,
    "lambda_2": lambda_2,
    "residual": residual,
    "initial_input": {"input_trunk": np.zeros((N, 1)), "input_branch": initial_states},
    "initial_state": initial_states,
    "weights_residual": [1, 1],
    "weights": penalties,
}


# In[ ]:


optimizer.fit(
    op=pendulum_net,
    input_data=input_data,
    n_epochs=n_epochs,
    loss="opirmse",
    params=params,
    device="gpu",
    batch_size=batch_size,
)


# #### Saving model

# In[ ]:


print("Saving model.")
saver = SPFile(compact=False)
saver.write(
    save_dir=save_path, name="pendulum_deeponet", model=pendulum_net, template=model
)


# In[ ]:


from scipy.integrate import odeint

# #### Pendulum numerical solver

# In[ ]:


class Pendulum:
    def __init__(self, rho: float = None, b: float = None, m: float = None) -> None:
        self.rho = rho
        self.b = b
        self.m = m

    def eval(self, state: np.ndarray = None, t: float = None) -> np.ndarray:
        x = state[0]
        y = state[1]

        x_residual = y
        y_residual = -self.b * y / self.m - self.rho * np.sin(x)

        return np.array([x_residual, y_residual])

    def run(self, initial_state, t):
        solution = odeint(self.eval, initial_state, t)

        return np.vstack(solution)


# In[ ]:


Q = 1000
N = int(100)
dt = 1 / Q

t = np.arange(0, 100, dt)

initial_state_0 = np.array([1, 1])

s_intv = np.stack([[-2, -2], [2, 2]], axis=0)
U_s = np.random.uniform(low=s_intv[0], high=s_intv[1], size=(N, 2))
U_s = np.vstack([U_s, np.array([[1, 1]])])

solver = Pendulum(rho=9.81, m=1, b=0.05)

saver = SPFile(compact=False)
pendulum_net = saver.read(model_path=save_path)


# In[ ]:


for j in range(N + 1):
    exact_data = solver.run(U_s[j], t)

    initial_state_test = U_s[j]

    n_outputs = 2
    n_times = 100

    branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))
    trunk_input_test = np.linspace(0, 1, Q)[:, None]

    approximated_data = rober_net.eval(
        trunk_data=trunk_input_test, branch_data=branch_input_test
    )
    data_ = torch.from_numpy(branch_input_test.astype("float32")).to("cuda")
    # print(rober_net.branch_network.gate(input_data=data_).cpu().detach().numpy())

    eval_list = list()

    for i in range(0, n_times):
        branch_input_test = np.tile(initial_state_test[None, :], (Q, 1))

        approximated_data = rober_net.eval(
            trunk_data=trunk_input_test, branch_data=branch_input_test
        )
        initial_state_test = approximated_data[-1]

        eval_list.append(approximated_data)

    evaluation = np.vstack(eval_list)
    time = np.linspace(0, n_times, evaluation.shape[0])

    l2_norm = L2Norm()

    error_s1 = 100 * l2_norm(
        data=evaluation[:, 0], reference_data=exact_data[:, 0], relative_norm=True
    )
    error_s2 = 100 * l2_norm(
        data=evaluation[:, 1], reference_data=exact_data[:, 1], relative_norm=True
    )

    print(f"State {j}, {U_s[j]}.")
    print(f"Approximation errors, s1: {error_s1} %, s2: {error_s2} ")

    if j % 1 == 0:
        plt.plot(time, evaluation[:, 0], label="Approximated")
        plt.plot(time, exact_data[:, 0], label="Exact", ls="--")
        plt.xlabel("t (s)")
        plt.ylabel("Angle")

        plt.xticks(np.arange(0, 100, 20))
        plt.legend()
        plt.ylim(1.5 * exact_data[:, 0].min(), 1.5 * exact_data[:, 0].max())
        plt.savefig(f"{model_name}_s1_time_int_{j}.png")
        plt.close()

        plt.plot(time, evaluation[:, 1], label="Approximated")
        plt.plot(time, exact_data[:, 1], label="Exact", ls="--")
        plt.xlabel("t (s)")
        plt.ylabel("Angular Speed")

        plt.xticks(np.arange(0, 100, 20))
        plt.legend()
        plt.ylim(1.5 * exact_data[:, 1].min(), 1.5 * exact_data[:, 1].max())
        plt.savefig(f"{model_name}_s2_time_int_{j}.png")
        plt.close()
