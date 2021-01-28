# WHAM

Optimized python code for constructing free energy profiles from umbrella sampling simulation data.

Link to [documentation](https://apallath.github.io/WHAM).

## Status

[![Actions Status](https://img.shields.io/github/workflow/status/apallath/WHAM/build_test_WHAM)](https://github.com/apallath/WHAM/actions)
[![Open Issues](https://img.shields.io/github/issues-raw/apallath/WHAM)](https://github.com/apallath/WHAM/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed-raw/apallath/WHAM)](https://github.com/apallath/WHAM/issues)

## Code

[![Python](https://img.shields.io/github/languages/top/apallath/WHAM)](https://www.python.org/downloads/release/python-370/)
[![Google Python Style](https://img.shields.io/badge/Code%20Style-Google%20Python%20Style-brightgreen)](https://google.github.io/styleguide/pyguide.html)

## Details

### Binned formulation (`WHAM.binned`)
- Implemented using self-consistent iteration (baseline/debugging)
- Implemented using log-likelihood maximization for superlinear convergence

### Binless formulation/MBAR (`WHAM.binless`) [preferred]
- Implemented using self-consistent iteration (baseline/debugging)
- Implemented using log-likelihood maximization for superlinear convergence

Both log-likelihood maximization approaches can use multiple nonlinear optimization algorithms. Read the documentation to see which algorithms are available.

Binless WHAM supports reweighting and binning 2D free energy profiles given a related (unbiased) order parameter.

## Installation

1. Install requirements

```sh
pip install -r requirements.txt
```

2. Build C extensions

```sh
python setup.py build_ext --inplace
```

2. Install package

```sh
pip install .
```

## Tests
Integration tests are in the directory `tests/` and unit tests are in the directory `tests/tests_unit`. To run all tests at once, navigate to the directory `tests/` and run

```sh
pytest
```

## References:
- Shirts, M. R., & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. Journal of Chemical Physics, 129(12). [DOI](https://doi.org/10.1063/1.2978177)
- Zhu, F., & Hummer, G. (2012). Convergence and error estimation in free energy calculations using the weighted histogram analysis method. Journal of Computational Chemistry, 33(4), 453–465. [DOI](https://doi.org/10.1002/jcc.21989)
- Tan, Z., Gallicchio, E., Lapelosa, M., & Levy, R. M. (2012). Theory of binless multi-state free energy estimation with applications to protein-ligand binding. Journal of Chemical Physics, 136(14). [DOI](https://doi.org/10.1063/1.3701175)
