=======
SimulAI
=======
.. image:: figs/coverage.svg

.. image:: figs/logo_2.svg

A Python package with data-driven pipelines for dynamical systems machine learning.

.. image:: figs/simulai_diagram.svg

The SimulAI toolkit provides easy access to state-of-the-art models and algorithms for physics-informed machine learning. Currently, it includes the following methods described in the literature:

- Physics-Informed Neural Networks (PINNs)
- Deep Operator Networks (DeepONets)
- Variational Encoder-Decoders (VED)
- Koopman AutoEncoders (experimental)
- Operator Inference (OpInf)

In addition to the methods above, many more techniques for model reduction and regularization are included in SimulAI. See `documentation <https://simulai.readthedocs.io/>`_.

Installing
==========

Python version requirements: [3.6, 3.10)

Using pip
---------

``pip install simulai-toolkit``

Using conda
-----------

``conda install -c conda-forge simulai-toolkit``

Contributing code to SimulAI
============================

It is strongly recommended that you have a Miniconda distribution installed. You can contribute code to SimulAI by following the steps below:

1. Fork this repository
2. Git clone the forked repository
3. Create a conda environment

``conda create -n simulai -f environment.yml``

4. Activate the newly created environment

``conda activate simulai``

Unit-testing
------------

1. ``cd`` to the root directory
2. Set PYTHONPATH environment variable

``export PYTHONPATH=.``

3. Run `pytest`

``pytest tests/``

Using MPI
=========

SimulAI supports multiprocessing with MPI.

In order to use it, you will need a valid MPI distribution, e.g. MPICH, OpenMPI.

``conda install -c conda-forge mpich gcc``

Issues with macOS
-----------------

If you have problems installing ``gcc`` using the command above, we recommend you to install it using `Homebrew <https://brew.sh>`_.

Documentation
=============

Please, refer to the SimulAI API `documentation <https://simulai.readthedocs.io>`_ before using the toolkit.

Examples
========

Additionally, you can refer to examples in the ``examples`` folder.

License
=======

This software is licensed under Apache 2.0.

References
==========

The following references in the literature.

Citing SimulAI
==============

If you find SimulAI to be useful, please consider citing it in your published work:

.. code-block:: python

    @software{simulai,
      author = {IBM},
      title = {SimulAI},
      url = {https://github.ibm.com/simulai/simulai},
      version = {},
      date = {},
    }
