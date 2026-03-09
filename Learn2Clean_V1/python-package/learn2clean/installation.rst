Installation guide
==================

|Documentation Status| |PyPI version| |Build Status| |GitHub Issues| |codecov| |License|

Compatibilities 
---------------

* *It is compatible with:* **Python 3.6**. & **64-bit version only** (32-bit python is not supported)
* *Operating system:* **MacOS**


Preparation 
-----------

First, make sure you have `setuptools <https://pypi.python.org/pypi/setuptools>`__ installed. Check that the following requirements are installed: 

* `gcc <https://gcc.gnu.org/>`__ 

.. code-block:: console

    $ sudo apt-get install build-essential
    
* `cmake <https://cmake.org/>`__  

.. code-block:: console

    $ pip install cmake
    
    
    
Installation
------------

Install from pip 
~~~~~~~~~~~~~~~~

Learn2clean is now available on **PyPI**, so you only need to run the following command:

.. code-block:: console

    $ pip install learn2clean


Install from the Github
~~~~~~~~~~~~~~~~~~~~~~~

* **The sources for Learn2clean can be downloaded** from the `Github repo`_.

    * You can either clone the public repository:

    .. code-block:: console

        $ git clone git://github.com/LaureBerti/Learn2Clean

    * Or download the `tarball`_:

    .. code-block:: console

        $ curl  -OL https://github.com/LaureBerti/Learn2Clean/tarball/master


* Once you have a copy of the source, **you can install it** using setup.py :
    
    .. code-block:: console

        $ cd python-package/
        $ python setup.py install


.. _Github repo: https://github.com/LaureBerti/Learn2Clean

.. _tarball: https://github.com/LaureBerti/Learn2Clean/tarball/master

.. |Documentation Status| image:: https://readthedocs.org/projects/learn2clean/badge/?version=latest
   :target: https://learn2clean.readthedocs.io/en/latest/
.. |PyPI version| image:: https://badge.fury.io/py/learn2clean.svg
   :target: https://pypi.python.org/pypi/learn2clean
.. |Build Status| image:: https://travis-ci.org/LaureBerti/Learn2Clean.svg?branch=master
   :target: https://travis-ci.org/LaureBerti/Learn2Clean
.. |GitHub Issues| image:: https://img.shields.io/github/issues/LaureBerti/Learn2Clean.svg
   :target: https://github.com/LaureBerti/Learn2Clean/issues
.. |codecov| image:: https://codecov.io/gh/LaureBerti/Learn2Clean/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/LaureBerti/Learn2Clean
.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/LaureBerti/Learn2Clean/blob/master/LICENSE
