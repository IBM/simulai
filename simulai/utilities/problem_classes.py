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

import numpy as np
from scipy import linalg


class NonlinearOscillator:

    def __init__(self, forcing : bool =False, p : int =3, alpha1=-0.1, alpha2=-2, beta1=2, beta2=-0.1):

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2

        self.p = p

        if forcing is True:
            self.eval = self._eval_with_forcing
        elif forcing is False:
            self.eval = self._eval_without_forcing
        else:
            raise Exception(f"It is expected forcing to be bool but received {type(forcing)}")

    def _eval_without_forcing(self, state):

        state_ = state[0]

        x = state_[0]
        y = state_[1]

        f = self.alpha1 * (x ** self.p) + self.beta1 * (y ** self.p)
        g = self.alpha2 * (x ** self.p) + self.beta2 * (y ** self.p)

        return f, g

    def _eval_with_forcing(self, state):

        state_ = state[0]

        x = state_[0]
        y = state_[1]
        u = state_[2]
        v = state_[3]

        f = self.alpha1 * (x ** self.p) + self.beta1 * (y ** self.p) + u
        g = self.alpha2 * (x ** self.p) + self.beta2 * (y ** self.p) + v

        return f, g

    def __call__(self, state):

        f, g = self.eval(state)

        return np.array([f, g])


class LorenzSystem:

    def __init__(self, rho, sigma, beta, forcing : bool =False, use_t : bool =False):
        """

        :param rho:
        :param sigma:
        :param beta:
        """

        self.rho = rho
        self.beta = beta
        self.sigma = sigma

        if forcing is True:
            self.eval = self._eval_with_forcing
        elif forcing is False and use_t is False:
            self.eval = self._eval_without_forcing
        elif forcing is False and use_t is True:
            self.eval = self._eval_without_forcing_t
        else:
            raise Exception(f"It is expected forcing to be bool but received {type(forcing)}")

    def _eval_without_forcing(self, state, *args):

        state_ = state[0]

        x = state_[0]
        y = state_[1]
        z = state_[2]

        f = self.sigma * (y - x)
        g = x*(self.rho - z) - y
        h = x*y - self.beta*z

        return np.array([f, g, h])

    def _eval_without_forcing_t(self, t, state):

        state_ = state

        x = state_[0]
        y = state_[1]
        z = state_[2]

        f = self.sigma * (y - x)
        g = x*(self.rho - z) - y
        h = x*y - self.beta*z

        return np.array([f, g, h])

    def _eval_with_forcing(self, state):

        state_ = state[0]

        x = state_[0]
        y = state_[1]
        z = state_[2]
        u = state_[3]
        v = state_[4]
        w = state_[5]

        f = self.sigma * (y - x) + u
        g = x*(self.rho - z) - y + v
        h = x*y - self.beta*z + w

        return np.array([f, g, h])

    def __call__(self, state):

        f, g, h = self.eval(state)

        return np.array([f, g, h])

    def jacobian(self, state, e, w, dt):

        x = state[0]
        y = state[1]
        z = state[2]

        e1 = e[0]
        e2 = e[1]
        e3 = e[2]

        D = np.array([
                        [-self.sigma, self.sigma, 0],
                        [-z + self.rho,    -1,   -x],
                        [y,       x,     -self.beta]
                     ])

        J = np.eye(3) + dt*D
        w_prev = w
        w = linalg.orth(J*w_prev)

        de1 = np.log(np.linalg.norm(w[:, 0], 2))
        de2 = np.log(np.linalg.norm(w[:, 1], 2))
        de3 = np.log(np.linalg.norm(w[:, 2], 2))

        e1 += de1
        e2 += de2
        e3 += de3

        w1 = w[:, 0] / np.linalg.norm(w[:, 0], 2)
        w2 = w[:, 1] / np.linalg.norm(w[:, 1], 2)
        w3 = w[:, 2] / np.linalg.norm(w[:, 2], 2)

        return np.array([e1, e2, e3]), np.hstack([w1[:, None], w2[:, None], w3[:, None]])
