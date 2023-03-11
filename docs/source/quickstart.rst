Quick Start 
===========
.. image:: https://zenodo.org/badge/561364034.svg
   :target: https://zenodo.org/badge/latestdoi/561364034
.. image:: https://badge.fury.io/py/simulai-toolkit.svg
   :target: https://badge.fury.io/py/simulai-toolkit
.. image:: https://readthedocs.org/projects/simulai-toolkit/badge/?version=latest
   :target: https://simulai-toolkit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: ../../assets/coverage.svg
   :target: ../../tests/
    
.. image:: ../../assets/logo.png

A Python package with data-driven pipelines for physics-informed machine learning.

.. image:: ../../assets/simulai_diagram.svg

The SimulAI toolkit provides easy access to state-of-the-art models and algorithms for physics-informed machine learning. Currently, it includes the following methods described in the literature:

1. `Physics-Informed Neural Networks <#references>`_ (PINNs)
2. `Deep Operator Networks <#references>`_ (DeepONets)
3. `Variational Encoder-Decoders <#reference>`_ (VED)
4. `Operator Inference <#references>`_ (OpInf)
5. `Koopman Autoencoders <#references>`_ (experimental)
6. `Echo State Networks <#references>`_ (experimental GPU support)

In addition to the methods above, many more techniques for model reduction and regularization are included in SimulAI. 

Installing Guide
----------------

.. code-block:: console
   
   Python version requirements: 3.8 <= python <= 3.10

Using pip
---------
.. code-block:: console

    $ pip install simulai-toolkit

Using MPI
---------

Some methods implemented on SimulAI support multiprocessing with MPI.

In order to use it, you will need a valid MPI distribution, e.g. MPICH, OpenMPI. As an example, you can use ``conda`` to install MPICH as follows: 

.. code-block:: console

    $ conda install -c conda-forge mpich gcc

Issues with macOS
-----------------

If you have problems installing ``gcc`` using the command above, we recommend you to install it using `Homebrew <https://brew.sh>`_.

Using Tensorboard
-----------------

`Tensorboard <https://www.tensorflow.org/tensorboard>`_ is supported for monitoring neural network training tasks. For a tutorial about how to set it see `this example <https://github.com/IBM/simulai/blob/main/examples/Dense/miscellaneous/notebooks/lorenz_96_chaotic.ipynb>`_.

Examples
--------
Additionally, you can refer to examples in the `respective folder. <https://github.com/IBM/simulai/blob/main/AUTHORS.rst>`_.

License
-------

This software is licensed under Apache 2.0. See `LICENSE <https://github.com/IBM/simulai/blob/main/LICENSE>`_.

Contributing code to SimulAI
----------------------------

If you are interested in directly contributing to this project, please see `CONTRIBUTING <https://github.com/IBM/simulai/blob/main/CONTRIBUTING.rst>`_.


How to cite SimulAI in your publications
----------------------------------------

If you find SimulAI to be useful, please consider citing it in your published work:

.. code-block:: python

    @misc{simulai,
      author = {IBM},
      title = {SimulAI Toolkit},
      subtitle = {A Python package with data-driven pipelines for physics-informed machine learning},
      note = "https://github.com/IBM/simulai",
      doi = {10.5281/zenodo.7351516},
      year = {2022},
    }

or, via Zenodo: 

.. code-block:: python

    @software{joao_lucas_de_sousa_almeida_2023_7566603,
          author       = {João Lucas de Sousa Almeida and
                          Leonardo Martins and
                          Tarık Kaan Koç},
          title        = {IBM/simulai: 0.99.13},
          month        = jan,
          year         = 2023,
          publisher    = {Zenodo},
          version      = {0.99.13},
          doi          = {10.5281/zenodo.7566603},
          url          = {https://doi.org/10.5281/zenodo.7566603}
        }

Publications
------------
João Lucas de Sousa Almeida, Pedro Roberto Barbosa Rocha, Allan Moreira de Carvalho and Alberto Costa Nogueira Jr. A coupled Variational
Encoder-Decoder - DeepONet surrogate model for the Rayleigh-Bénard convection problem. In When Machine Learning meets Dynamical Systems:
Theory and Applications, AAAI, 2023.

João Lucas S. Almeida, Arthur C. Pires, Klaus F. V. Cid, and Alberto C.
Nogueira Jr. Non-intrusive operator inference for chaotic systems. IEEE Transactions on Artificial Intelligence, pages 1–14, 2022.

Pedro Roberto Barbosa Rocha, Marcos Sebastião de Paula Gomes,
Allan Moreira de Carvalho, João Lucas de Sousa Almeida and Alberto Costa
Nogueira Jr. Data-driven reduced-order model for atmospheric CO2 dispersion. In AAAI 2022 Fall Symposium: The Role of AI in Responding to
Climate Challenges, 2022.


References
----------
Jaeger, H., Haas, H. (2004).
"Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication,"
*Science*, **304** (5667): 78–80.
DOI:`10.1126/science.1091277 <https://doi.org/10.1126/science.1091277>`_.

Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. E. (2021).
"Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators,"
*Nature Machine Intelligence*, **3** (1): 218–229.
ISSN: 2522-5839.
DOI:`10.1038/s42256-021-00302-5 <https://doi.org/10.1038/s42256-021-00302-5>`_.

Eivazi, H., Le Clainche, S., Hoyas, S., Vinuesa, R. (2022)
"Towards extraction of orthogonal and parsimonious non-linear modes from
turbulent flows"
*Expert Systems with Applications*, **202**.
ISSN: 0957-4174.
DOI:`10.1016/j.eswa.2022.117038 <https://doi.org/10.1016/j.eswa.2022.117038>`_.

Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019).
"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,"
*Journal of Computational Physics*, **378** (1): 686-707.
ISSN: 0021-9991.
DOI:`10.1016/j.jcp.2018.10.045 <https://doi.org/10.1016/j.jcp.2018.10.045>`_.

Lusch, B., Kutz, J. N., Brunton, S.L. (2018).
"Deep learning for universal linear embeddings of nonlinear dynamics,"
*Nature Communications*, **9**: 4950.
ISSN: 2041-1723.
DOI:`10.1038/s41467-018-07210-0 <https://doi.org/10.1038/s41467-018-07210-0>`_.

*McQuarrie, S., Huang, C. and Willcox*, K. (2021).
"Data-driven reduced-order models via regularized operator inference for a single-injector combustion process," 

*Journal of the Royal Society of New Zealand, **51** (2): 194-211.
ISSN: 0303-6758.
DOI:`10.1080/03036758.2020.1863237 <https://doi.org/10.1080/03036758.2020.1863237>`_.