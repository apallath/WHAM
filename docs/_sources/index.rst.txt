.. WHAM documentation master file

###############################
WHAM
###############################

:Release: |release|

WHAM is a Python package for constructing free energy profiles from
umbrella sampling simulation data.

Installation
**************************

**Source code** is available from
`https://github.com/apallath/WHAM`_

Obtain the sources with `git`_

.. code-block:: bash

  git clone https://github.com/apallath/WHAM.git

.. _`https://github.com/apallath/WHAM`: https://github.com/apallath/WHAM
.. _git: https://git-scm.com/

1. Install requirements

.. code-block:: bash

  pip install -r requirements.txt

2. Build C extensions

.. code-block:: bash

  python setup.py build_ext --inplace

2. Install package [in editable state]

.. code-block:: bash

  pip install [-e] .


Running tests
**************************

.. code-block:: bash

  cd tests
  pytest


Usage
**************************

Binless WHAM is generally a better choice for accuracy, and implements
more features than binned WHAM (such as reweighting, binning 2D profiles
given a related order parameter, and integrating these profiles to obtain
free energy profiles in terms of a related unbiased order parameter).
However, binned WHAM is faster and uses less memory than
binless WHAM.

Log-likelihood maximization is a better approach than self-consistent iteration, which can suffer
from slow convergence.

Choose between the two different WHAM formulations and solution
approaches based on your needs.

Look at the documentation of the statistics module to understand how to
use statistical checks to verify the consistency of WHAM
calculations.

For examples demonstrating free energy profile calculations, see the `examples` directory.


References
**************************

- Shirts, M. R., & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. Journal of Chemical Physics, 129(12). `[1]`_
- Zhu, F., & Hummer, G. (2012). Convergence and error estimation in free energy calculations using the weighted histogram analysis method. Journal of Computational Chemistry, 33(4), 453–465. `[2]`_
- Tan, Z., Gallicchio, E., Lapelosa, M., & Levy, R. M. (2012). Theory of binless multi-state free energy estimation with applications to protein-ligand binding. Journal of Chemical Physics, 136(14). `[3]`_

.. _`[1]`: https://doi.org/10.1063/1.2978177
.. _`[2]`: https://doi.org/10.1002/jcc.21989
.. _`[3]`: https://doi.org/10.1063/1.3701175

================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   WHAM/WHAM
   WHAM/WHAM.lib


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
