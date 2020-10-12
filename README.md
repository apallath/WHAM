# pyWHAM

Optimized python code for constructing free energy profiles from umbrella sampling simulation data.

## Binned formulation
- Implemented using log-likelihood maximization for superlinear convergence

## Binless formulation (MBAR) (preferred)
- Implemented using self-consistent iteration (baseline/debugging)
- Implemented using log-likelihood maximization for superlinear convergence

Both log-likelihood maximization approaches can use multiple nonlinear optimization algorithms. Read the documentation to see which algorithms are available.

## References:
- Shirts, M. R., & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. Journal of Chemical Physics, 129(12). [DOI](https://doi.org/10.1063/1.2978177)
- Zhu, F., & Hummer, G. (2012). Convergence and error estimation in free energy calculations using the weighted histogram analysis method. Journal of Computational Chemistry, 33(4), 453–465. [DOI](https://doi.org/10.1002/jcc.21989)
- Tan, Z., Gallicchio, E., Lapelosa, M., & Levy, R. M. (2012). Theory of binless multi-state free energy estimation with applications to protein-ligand binding. Journal of Chemical Physics, 136(14). [DOI](https://doi.org/10.1063/1.3701175)
