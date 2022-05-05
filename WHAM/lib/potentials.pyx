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

    $kappa$ has shape (D,)
    $x^*$ has shape (D,)
    $x$ has shape (N, D)
    $U(x) has shape (N,)
    """
    # Reshape kappa and xstar for matrix multiply
    kappa = np.array(kappa).reshape(-1, 1)  # reshape to (D, 1)
    xstar = np.array(xstar).reshape(1, -1)  # reshape to (1, D)

    # Check that dimensions match
    assert kappa.shape[0] == xstar.shape[1], "kappa should have dimensons (D, 1) and x* should have dimensions (1, D), but kappa has dimensions %s and x* has dimensions %s" % (kappa.shape, xstar.shape)

    def potential(x):
        # x must be 2-dimensional
        assert len(x.shape) == 2, "x must be 2-dimensional"
        # x must have shape (N, D)
        assert x.shape[1] == xstar.shape[1], "x and x* must have same dimensions, but x is %d-dimensional and x* is %d-dimensional" % (x.shape[1], xstar.shape[1])
        # (N, D) * (D, 1) = (N, 1) --> (N,)
        return np.matmul((x - xstar) ** 2, kappa / 2).squeeze(-1)
    return potential


def linear_DD(phi):
    r"""
    Returns umbrella potential
    $U(x) = \sum_{i=0}^{D - 1} \phi_i x_i

    $phi$ has shape (D,)
    $x$ has shape (N, D)
    $U(x) has shape (N,)
    """
    # Reshape phi for matrix multiply
    phi = np.array(phi).reshape(-1, 1)  # reshape to (D, 1)

    def potential(x):
        # x must be 2-dimensional
        assert len(x.shape) == 2, "x must be 2-dimensional"
        # x must have shape (N, D)
        assert x.shape[1] == phi.shape[0], "x and phi must have same dimensions, but x is %d-dimensional and phi is %d-dimensional" % (x.shape[1], phi.shape[0])
        # (N, D) * (D, 1) = (N, 1) --> (N,)
        return np.matmul(x, phi).squeeze(-1)
    return potential
