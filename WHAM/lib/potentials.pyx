"""Contains definitions of commonly used Umbrella potentials."""
import numpy as np


def harmonic(kappa, xstar):
    r"""
    Returns umbrella potential
    $U(x) = \frac{\kappa}{2} (x - x^*)^2$

    Args:
        kappa (float)
        xstar (float)
    """
    def potential(x):
        return kappa / 2 * (x - xstar) ** 2
    return potential


def linear(phi):
    r"""
    Returns umbrella potential
    $U(x) = \phi x$

    Args:
        phi (float)
    """
    def potential(x):
        return phi * x
    return potential


def harmonic_DD(kappa, xstar):
    r"""
    Returns umbrella potential
    $U(x) = \frac{1}{2} \sum_{i=0}^{D - 1} \kappa_i (x_i - x_i^*)^2$
    """
    # x has shape (D,) or (N, D)
    # kappa has shape (D,)
    # x* has shape (D,)
    # returned potential has shape (1,) or (N,)

    # (D,) x (D,) --> (1, D) x (D, 1) = (1, 1)
    # (N, D) x (D,) --> (N, D) x (D, 1) = (N, 1)
    kappa = np.array(kappa)

    if kappa.shape[0] != 1:
        kappa = kappa.reshape(1, -1)

    def potential(x):
        return np.matmul(kappa / 2, (x - xstar) ** 2)
    return potential


def linear_DD(phi):
    r"""
    Returns umbrella potential
    $U(x) = \sum_{i=0}^{D - 1} \phi_i x_i
    """
    # x has shape (D,) or (N, D)
    # kappa has shape (D,)
    # x* has shape (D,)
    # returned potential has shape (1,) or (N,)

    # (D,) x (D,) --> (1, D) x (D, 1) = (1, 1)
    # (N, D) x (D,) --> (N, D) x (D, 1) = (N, 1)
    def potential(x):
        return np.matmul(phi, x)
    return potential
