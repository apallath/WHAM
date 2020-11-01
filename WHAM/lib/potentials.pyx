"""Contains definitions of commonly used Umbrella potentials"""


def harmonic(kappa, xstar):
    """
    Returns umbrella potential
    U(x) = kappa/2 (x - xstar)^2
    """
    def harmonic_potential(x):
        return kappa / 2 * (x - xstar) ** 2
    return harmonic_potential


def linear(phi):
    """
    Returns umbrella potential
    U(x) = phi * x
    """
    def harmonic_potential(x):
        return phi * x
    return harmonic_potential
