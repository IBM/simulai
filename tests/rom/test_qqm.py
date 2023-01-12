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


from unittest import TestCase
import numpy as np

from simulai.file import load_pkl
from simulai.rom import QQM

class TestQQM(TestCase):

    def setUp(self) -> None:

        self.t = np.linspace(0, 2 * np.pi, 10000)

        # input: s = [t, sin(t), cos(t)]
        self.s = np.hstack([self.t[:, None], np.sin(self.t)[:, None], np.cos(self.t)[:, None]])

        # output: e = [1, sin(2*t), cos(2*t)]
        self.e = np.hstack([np.ones(self.t.shape)[:, None], np.sin(2 * self.t)[:, None], np.cos(2 * self.t)[:, None]])

    def test_qqm_sparsa(self):

        # Using the default solver (SpaRSA)
        qqm = QQM(n_inputs=3, lambd=1e-3, alpha_0=100, epsilon=1e-2, use_mean=True)

        # Using the default solver (SpaRSA) without sparsity hard-limiting
        qqm.fit(input_data=self.s, target_data=self.e)

        print("\n Vanilla SpaRSA")

        assert isinstance(qqm.V_bar, np.ndarray)

    def test_qqm_sparsa_hard_limiting(self):

        # Using the default solver (SpaRSA) with sparsity hard-limiting
        qqm = QQM(n_inputs=3, lambd=1e-3, sparsity_tol=1e-6, epsilon=1e-2, alpha_0=100, use_mean=True)

        qqm.fit(input_data=self.s, target_data=self.e)

        assert isinstance(qqm.V_bar, np.ndarray)

    def test_qqm_pinv(self):

        qqm = QQM(n_inputs=3, lambd=1e-3, alpha_0=100, use_mean=True)

        # Using the Moore-Penrose generalized inverse
        qqm.fit(input_data=self.s, target_data=self.e, pinv=True)

        print("\n Moore-Penrose pseudoinverse")
        assert isinstance(qqm.V_bar, np.ndarray)

    def test_qqm_save_load(self):

        qqm = QQM(n_inputs=3, lambd=1e-3, alpha_0=100, use_mean=True)

        # Using the Moore-Penrose generalized inverse
        qqm.fit(input_data=self.s, target_data=self.e, pinv=True)

        qqm.save(save_path='/tmp', model_name='qqm')

        qqm_reload = load_pkl('/tmp/qqm.pkl')

        assert isinstance(qqm_reload.V_bar, np.ndarray)