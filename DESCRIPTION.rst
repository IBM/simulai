The SimulAI toolkit is a Python package with data-driven pipelines that provide easy access to state-of-the-art models and algorithms for physics-informed machine learning. Currently, it includes the following methods described in the literature:

- Physics-Informed Neural Networks (PINNs)
- Deep Operator Networks (DeepONets)
- Variational Encoder-Decoders (VED)
- Koopman AutoEncoders (experimental)
- Operator Inference (OpInf)
- Echo State Networks (Reservoir Computing)

In addition to the methods above, many more techniques for model reduction and regularization are included in SimulAI. See `documentation <https://simulai-toolkit.readthedocs.io/>`_.

Installing
==========

Python version requirements: [3.6, 3.10)

Using pip
---------

``pip install simulai-toolkit``

Using MPI
=========

Some methods implemented on SimulAI support multiprocessing with MPI.

In order to use it, you will need a valid MPI distribution, e.g. MPICH, OpenMPI.

``conda install -c conda-forge mpich gcc``

You need to run the following command to install SimulAI with MPI support:

``pip install simulai-toolkit[mpi]``

Issues with macOS
-----------------

If you have problems installing ``gcc`` using the command above, we recommend you to install it using `Homebrew <https://brew.sh>`_.

Documentation
=============

Please, refer to the SimulAI API `documentation <https://simulai-toolkit.readthedocs.io>`_ before using the toolkit.

Examples
========

Additionally, you can refer to examples of how to use SimulAI in the `examples folder <https://github.com/IBM/simulai/tree/main/examples>`_.

License
=======

This software is licensed under Apache 2.0.

