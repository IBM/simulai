# Welcome to SimulAI

[![image](https://badge.fury.io/py/simulai-toolkit.svg)](https://badge.fury.io/py/simulai-toolkit)

[![Documentation Status](https://readthedocs.org/projects/simulai-toolkit/badge/?version=latest)](https://simulai-toolkit.readthedocs.io/en/latest/?badge=latest)

[![image](../assets/coverage.svg)](tests/)

![image](../../assets/logo_1.png){width="500px"}

An extensible Python package with data-driven pipelines for
physics-informed machine learning.

[//]: ![image](../assets/simulai_diagram.svg)

The SimulAI toolkit provides easy access to state-of-the-art models and
algorithms for physics-informed machine learning. Currently, it includes
the following methods described in the literature:

-   [Physics-Informed Neural Networks](#references) (PINNs)
-   [Deep Operator Networks](#references) (DeepONets)
-   [Variational Encoder-Decoders](#reference) (VED)
-   [Operator Inference](#references) (OpInf)
-   [Koopman Autoencoders](#references) (experimental)
-   [Echo State Networks](#references) (experimental GPU support)
-   Transformers
-   U-Nets

In addition to the methods above, many more techniques for model
reduction and regularization are included in SimulAI. See
[documentation](https://ibm.github.io/simulai).

# Installing

Python version requirements: 3.9 \<= python \<= 3.11

## Using pip

For installing the most recent stable version from PyPI:

``` shell
pip install simulai-toolkit
```

For installing from the latest commit sent to GitHub (just for testing
and developing purposes):

``` shell
pip uninstall simulai_toolkit
pip install -U git+https://github.com/IBM/simulai@$(git ls-remote git@github.com:IBM/simulai.git  | head -1 | awk '{print $1;}')#egg=simulai_toolkit
```

# Contributing code to SimulAI

If you are interested in directly contributing to this project, please
see [CONTRIBUTING](CONTRIBUTING.rst).

# Using MPI

Some methods implemented on SimulAI support multiprocessing with MPI.

In order to use it, you will need a valid MPI distribution, e.g. MPICH,
OpenMPI. As an example, you can use `conda` to install MPICH as follows:

``` shell
conda install -c conda-forge mpich gcc
```

## Issues with macOS

If you have problems installing `gcc` using the command above, we
recommend you to install it using [Homebrew](https://brew.sh).

# Using Tensorboard

[Tensorboard](https://www.tensorflow.org/tensorboard) is supported for
monitoring neural network training tasks. For a tutorial about how to
set it see [this
example](https://github.com/IBM/simulai/blob/main/examples/Dense/miscellaneous/notebooks/lorenz_96_chaotic.ipynb).

# Documentation

Please, refer to the SimulAI API
[documentation](https://simulai-toolkit.readthedocs.io) before using the
toolkit.

# Examples

Additionally, you can refer to examples in the [respective
folder](examples/).

# License

This software is licensed under Apache license 2.0. See
[LICENSE](LICENSE).

## Contributing code to SimulAI

If you are interested in directly contributing to this project, please
see
[CONTRIBUTING](https://github.com/IBM/simulai/blob/main/CONTRIBUTING.rst).

# How to cite SimulAI in your publications

If you find SimulAI to be useful, please consider citing it in your
published work:

``` python
@misc{simulai,
  author = {IBM},
  title = {SimulAI Toolkit},
  subtitle = {A Python package with data-driven pipelines for physics-informed machine learning},
  note = "https://github.com/IBM/simulai",
  doi = {10.5281/zenodo.7351516},
  year = {2022},
}
```

or, via Zenodo:

``` python
@software{joao_lucas_de_sousa_almeida_2023_7566603,
      author       = {João Lucas de Sousa Almeida and
                      Leonardo Martins and
                      Tarık Kaan Koç},
      title        = {IBM/simulai: 0.99.13},
      month        = jan,
      year         = 2023,
      publisher    = {Zenodo},
      version      = {0.99.25},
      doi          = {10.5281/zenodo.7566603},
      url          = {https://doi.org/10.5281/zenodo.7566603}
    }
```

# Publications

João Lucas de Sousa Almeida, Pedro Roberto Barbosa Rocha, Allan Moreira
de Carvalho and Alberto Costa Nogueira Jr. A coupled Variational
Encoder-Decoder - DeepONet surrogate model for the Rayleigh-Bénard
convection problem. In When Machine Learning meets Dynamical Systems:
Theory and Applications, AAAI, 2023.

João Lucas S. Almeida, Arthur C. Pires, Klaus F. V. Cid, and Alberto C.
Nogueira Jr. Non-intrusive operator inference for chaotic systems. IEEE
Transactions on Artificial Intelligence, pages 1--14, 2022.

Pedro Roberto Barbosa Rocha, Marcos Sebastião de Paula Gomes, Allan
Moreira de Carvalho, João Lucas de Sousa Almeida and Alberto Costa
Nogueira Jr. Data-driven reduced-order model for atmospheric CO2
dispersion. In AAAI 2022 Fall Symposium: The Role of AI in Responding to
Climate Challenges, 2022.

Pedro Roberto Barbosa Rocha, João Lucas de Sousa Almeida, Marcos
Sebastião de Paula Gomes, Alberto Costa Nogueira, Reduced-order modeling
of the two-dimensional Rayleigh--Bénard convection flow through a
non-intrusive operator inference, Engineering Applications of Artificial
Intelligence, Volume 126, Part B, 2023, 106923, ISSN 0952-1976,
<https://doi.org/10.1016/j.engappai.2023.106923>.
(<https://www.sciencedirect.com/science/article/pii/S0952197623011077>)

# References

Jaeger, H., Haas, H. (2004). \"Harnessing Nonlinearity: Predicting
Chaotic Systems and Saving Energy in Wireless Communication,\"
*Science*, **304** (5667): 78--80. <DOI:%6010.1126/science.1091277>
\<<https://doi.org/10.1126/science.1091277>\>\`\_.

Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. E. (2021).
\"Learning nonlinear operators via DeepONet based on the universal
approximation theorem of operators,\" *Nature Machine Intelligence*,
**3** (1): 218--229. ISSN: 2522-5839.
<DOI:%6010.1038/s42256-021-00302-5>
\<<https://doi.org/10.1038/s42256-021-00302-5>\>\`\_.

Eivazi, H., Le Clainche, S., Hoyas, S., Vinuesa, R. (2022) \"Towards
extraction of orthogonal and parsimonious non-linear modes from
turbulent flows\" *Expert Systems with Applications*, **202**. ISSN:
0957-4174. <DOI:%6010.1016/j.eswa.2022.117038>
\<<https://doi.org/10.1016/j.eswa.2022.117038>\>\`\_.

Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019).
\"Physics-informed neural networks: A deep learning framework for
solving forward and inverse problems involving nonlinear partial
differential equations,\" *Journal of Computational Physics*, **378**
(1): 686-707. ISSN: 0021-9991. <DOI:%6010.1016/j.jcp.2018.10.045>
\<<https://doi.org/10.1016/j.jcp.2018.10.045>\>\`\_.

Lusch, B., Kutz, J. N., Brunton, S.L. (2018). \"Deep learning for
universal linear embeddings of nonlinear dynamics,\" *Nature
Communications*, **9**: 4950. ISSN: 2041-1723.
<DOI:%6010.1038/s41467-018-07210-0>
\<<https://doi.org/10.1038/s41467-018-07210-0>\>\`\_.

McQuarrie, S., Huang, C. and Willcox, K. (2021). \"Data-driven
reduced-order models via regularized operator inference for a
single-injector combustion process,\" *Journal of the Royal Society of
New Zealand*, **51**(2): 194-211. ISSN: 0303-6758.
<DOI:%6010.1080/03036758.2020.1863237>
\<<https://doi.org/10.1080/03036758.2020.1863237>\>\`\_. 


