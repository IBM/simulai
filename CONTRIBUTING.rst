.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/IBM/simulai/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Be sure to take a look at the template here: https://github.com/IBM/simulai/tree/main/docs/_templates/ISSUES_TEMPLATE.rst

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Create new examples
~~~~~~~~~~~~~~~~~~~

You can create new examples executed in notebooks and include them under `examples`. 

Write Documentation
~~~~~~~~~~~~~~~~~~~

simulai could always use more documentation, whether as part of the
official simulai docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/IBM/simulai/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `simulai` for local development.

1. Fork the `simulai` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/simulai.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv simulai
    $ cd simulai/
    $ python -m pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 simulai
    $ pytest simulai/test
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. We strongly recommend you to create a dedicated branch when modifying core features. It can be easily tested by everyone
   and afterwards merged to **main**. 
2. The pull request should include tests. In this way, test locally (we plan to automate this task shortly) before submitting a pull request
   using **pytest**, as::
   
   $ pytest --durations=0 tests/
    
3. As not all the tests are really necessary for
   a given modification in the source code, we recommend the usage of the pytest plugin 
   **testmon** (https://github.com/tarpas/pytest-testmon), which will select the correct tests to be
   executed at each commit/pull request::
    
    $ pytest --durations=0 --testmon tests/
   
4. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
5. The pull request should work for Python 3.6, 3.7, 3.8, and 3.9. Check
   https://travis-ci.org/ltizzei/simulai/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests, e.g.::

$ pytest simulai/test/math

Or::

$ pytest --durations=0 simulai/test/math 

For estimating execution times.

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bumpver --update <TYPE>  # <TYPE> options: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.
