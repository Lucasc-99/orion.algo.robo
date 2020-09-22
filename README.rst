===============
orion.algo.robo
===============


.. |pypi| image:: https://img.shields.io/pypi/v/orion.algo.robo
    :target: https://pypi.python.org/pypi/orion.algo.robo
    :alt: Current PyPi Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/orion.algo.robo.svg
    :target: https://pypi.python.org/pypi/orion.algo.robo
    :alt: Supported Python Versions

.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: BSD 3-clause license

.. |rtfd| image:: https://readthedocs.org/projects/orion.algo.robo/badge/?version=latest
    :target: https://orion.algo-robo.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |codecov| image:: https://codecov.io/gh/Lucasc-99/orion.algo.robo/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Lucasc-99/orion.algo.robo
    :alt: Codecov Report

.. |travis| image:: https://travis-ci.org/Lucasc-99/orion.algo.robo.svg?branch=master
    :target: https://travis-ci.org/Lucasc-99/orion.algo.robo
    :alt: Travis tests


----

This `orion.algo`_ plugin was generated with `Cookiecutter`_ along with `@Epistimio`_'s `cookiecutter-orion.algo`_ template.

See Orion : https://github.com/Epistimio/orion


Installation
------------

Install RoBO: instructions from https://github.com/automl/RoBO/blob/master/README.md

NOTE: RoBO installation is very difficult on MAC

RoBO uses the libraries george and pyrfr.
Additionally, make sure that libeigen and swig are installed

sudo apt-get install libeigen3-dev swig

Download RoBO and then change into the new directory:

git clone https://github.com/automl/RoBO
cd RoBO/
Install the required dependencies.

for req in $(cat requirements.txt); do pip install $req; done
Finally install RoBO by:

python setup.py install



You can install "orion.algo.robo" via `pip`_ from `PyPI`_::

    $ pip install git+https://github.com/Lucasc-99/orion.algo.robo.git



Contributing
------------
Contributions are very welcome. Tests can be run with `tox`_, please ensure
the coverage at least stays the same before you submit a pull request.

License
-------

Distributed under the terms of the BSD-3-Clause license,
"orion.algo.robo" is free and open source software.


Issues
------

If you encounter any problems, please `file an issue`_ along with a detailed description.

.. _`Cookiecutter`: https://github.com/audreyr/cookiecutter
.. _`@Epistimio`: https://github.com/Epistimio
.. _`GNU GPL v3.0`: http://www.gnu.org/licenses/gpl-3.0.txt
.. _`cookiecutter-orion.algo`: https://github.com/Epistimio/cookiecutter-orion.algo
.. _`file an issue`: https://github.com/Lucasc-99/cookiecutter-orion.algo.robo/issues
.. _`orion`: https://github.com/Epistimio/orion
