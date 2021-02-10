"""
Implementation of binless WHAM (described in Shirts M., & Chodera J. D. (2008)) using
    - Negative log-likelihood maximization, inspired from Zhu, F., & Hummer, G. (2012)
    - Self-consistent iteration.
"""
import numpy as np

import WHAM.lib.potentials
import WHAM.lib.timeseries
from WHAM.lib import numeric

import scipy.optimize

from tqdm import tqdm

# Automatic differentiation, for log-likelihood maximization
import autograd.numpy as anp
from autograd import value_and_grad

# Cython optimization for self-consistent iteration
cimport numpy as np

from functools import partial


cdef class Calc1D:
    """Class containing methods to compute free energy profiles
    from umbrella sampling data using binless WHAM.
    Handles empty bins when computing free energy profiles by setting
    bin free energy to inf.

    Note:
        Binless WHAM implements several features which are not part of binned WHAM, such as reweighting,
        and binning 2D free energy profile given a related (unbiased) order parameter.f

    Important:
        Usage patterns for this module differ from binned WHAM. For a complete usage example, look at tests/test_binless.py.

    Example:
        >>> calc = Calc1D()
        >>> status = calc.compute_point_weights(x_l, ...)
        >>> status
        True
        >>> G_l = calc.G_l
        >>> g_i = calc.g_i
        >>> betaF_x = calc.bin_betaF_profile(x_l, G_l, x_bin, ...)
        >>> betaF_xy = calc.bin_2D_betaF_profile(x_l, y_l, G_l, x_bin, y_bin, ...)

    Attributes:
        x_l (ndarray): 1-dimensional array containing unrolled order parameter
            which is being biased.
        G_l (ndarray): 1-dimensional array containing weights corresponding to
            each (unrolled) order parameter.
        g_i (ndarray): 1-dimensional array of size N (=no of windows) containing
            free energies for each window.
    """

    # Counter for log-likelihood minimizer
    cdef int _min_ctr

    # Output arrays
    cdef public np.ndarray x_l
    cdef public np.ndarray G_l
    cdef public np.ndarray g_i

    def __cinit__(self):
        self._min_ctr = 0
        self.x_l = None
        self.G_l = None
        self.g_i = None

    def NLL(self, g_i, x_l, N_i, W_il):
        """Computes the negative log-likelihood objective function to minimize.

        Args:
            g_i (ndarray of shape (S,)) Array of total free energies associated with the windows
                0, 1, 2, ..., S-1.
            x_l (ndarray of shape (Ntot,)): Array containing each sample.
            N_i (ndarray of shape (S,)): Array of total sample counts for the windows
                0, 1, 2, ..., S-1.
            W_il (ndarray of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
                0, 1, 2, ..., S-1.

        Returns:
            A(g) (np.float): Negative log-likelihood objective function.
        """
        with anp.errstate(divide='ignore'):
            Ntot = anp.sum(N_i)
            term1 = -anp.sum(N_i / Ntot * g_i)
            term2 = 1 / Ntot * anp.sum(numeric.alogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis] / Ntot, axis=0))

            A = term1 + term2

            return A

    def _min_callback(self, g_i, args, logevery=100000000):
        if self._min_ctr % logevery == 0:
            print("{:10d} {:.5f}".format(self._min_ctr, self.NLL(g_i, *args)))
        self._min_ctr += 1

    def minimize_NLL_solver(self, x_l, N_i, W_il, g_i=None, opt_method='L-BFGS-B', logevery=100000000):
        """Computes optimal g_i by minimizing the negative log-likelihood
        for jointly observing the bin counts in the independent windows in the dataset.

        Note:
            Any optimization method which scipy.optimize supports can be used. L-BFGS-B is used
            by default. Gradient information required for L-BFGS-B is computed using autograd.

        Args:
            x_l (ndarray of shape (Ntot,)): Array containing each sample.
            N_i (ndarray of shape (S,)): Array of total sample counts for the windows
                0, 1, 2, ..., S-1.
            W_il (ndarray of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
                0, 1, 2, ..., S-1.
            g_i (ndarray of shape (S,)): Total free energy initial guess.
            opt_method (string): Optimization algorithm to use (default: L-BFGS-B).
            logevery (int): Interval to log negative log-likelihood (default=100000000, i.e. no logging).

        Returns:
            tuple(bG_l, g_i, status)
                - bG_l (ndarray of shape (Ntot,)): Free energy for each sample point,
                - g_i (ndarray of shape (S,)): Total free energy for each window,
                - status (bool): Solution status.
        """
        if g_i is None:
            g_i = np.random.rand(len(N_i))  # TODO: Smarter initial guess

        # Optimize
        print("      Iter NLL")
        self._min_ctr = 0
        res = scipy.optimize.minimize(value_and_grad(self.NLL), g_i, jac=True,
                                      args=(x_l, N_i, W_il),
                                      method=opt_method,
                                      callback=partial(self._min_callback, args=(x_l, N_i, W_il), logevery=logevery))
        self._min_callback(res.x, args=(x_l, N_i, W_il), logevery=1)

        g_i = res.x
        g_i = g_i - g_i[0]

        G_l = numeric.clogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0)

        return G_l, g_i, res.success

    cpdef self_consistent_solver(self, np.ndarray x_l, np.ndarray N_i, np.ndarray W_il,
                                 np.ndarray g_i=np.zeros(1), float tol=1e-7, int maxiter=100000, int logevery=100000000):
        """Computes optimal parameters g_i by solving the coupled MBAR equations self-consistently
        until convergence. Optimized using Cython.

        Args:
            x_l (ndarray of shape (Ntot,)): Array containing each sample.
            N_i (ndarray of shape (S,)): Array of total sample counts for the windows
                0, 1, 2, ..., S-1.
            W_il (ndarray of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
                0, 1, 2, ..., S-1.
            g_i (ndarray of shape (S,)): Total free energy initial guess.
            tol (float): Relative tolerance to stop solver iterations at (defaul=1e-7).
            maxiter (int): Maximum number of iterations to run solver for (default=100000).
            logevery (int): Interval to log self-consistent solver error (default=100000000, i.e. no logging).

        Returns:
            tuple(bG_l, g_i, status)
                - bG_l (ndarray of shape (Ntot,)): Free energy for each sample point,
                - g_i (ndarray of shape (S,)): Total free energy for each window,
                - status (bool): Solution status.
        """
        cdef float EPS = 1e-24

        cdef int S = len(N_i)
        cdef int Ntot = len(x_l)

        if np.allclose(g_i, np.zeros(1)):
            g_i = EPS * np.ones(S, dtype=np.float)
        else:
            g_i[g_i == 0] = EPS

        # Solution obtained with required tolerance level?
        cdef bint status = False

        cdef int iter
        cdef float tol_check
        cdef np.ndarray g_i_mat, N_i_mat, G_l, G_l_mat, g_i_prev, increment

        for iter in range(maxiter):
            G_l = numeric.clogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0)

            g_i_prev = g_i

            g_i = -numeric.clogsumexp(W_il - G_l[np.newaxis, :], axis=1)
            g_i = g_i - g_i[0]

            # Tolerance check
            g_i[g_i == 0] = EPS  # prevent divide by zero
            g_i_prev[g_i_prev == 0] = EPS  # prevent divide by zero
            increment = np.abs((g_i - g_i_prev) / g_i_prev)

            tol_check = increment[np.argmax(increment)]

            if logevery > 0:
                if iter % logevery == 0:
                    print("Self-consistent solver error = {:.2e}.".format(tol_check))

            if increment[np.argmax(increment)] < tol:
                if logevery > 0:
                    print("Self-consistent solver error = {:.2e}.".format(tol_check))
                status = True
                break

        return G_l, g_i, status

    ###############################
    # Main computation call       #
    ###############################
    def compute_point_weights(self, x_l, N_i, u_i, beta, solver='log-likelihood', **solverkwargs):
        """Computes WHAM weights corresponding to each order parameter sample
        and total window free energies. This is the main computation call
        for the Calc1D class.

        Args:
            x_l (ndarray): Array containing each sample (unrolled).
            N_i (ndarray): Counts for each window.
            u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
                acting on the i'th window.
            beta: beta, in inverse units to the units of u_i(x).
            solver (string): Solution technique to use ['log-likelihood', 'self-consistent', default='log-likelihood'].
            **solverkwargs: Arguments for solver.
        """
        self.x_l = x_l

        S = len(u_i)
        Ntot = len(x_l)

        # Compute weights for each data point
        W_il = np.zeros((S, Ntot))
        for i in range(S):
            W_il[i, :] = -beta * u_i[i](x_l)

        # Get free energy profile through binless WHAM/MBAR
        if solver == 'log-likelihood':
            G_l, g_i, status = self.minimize_NLL_solver(x_l, N_i, W_il, **solverkwargs)
        elif solver == 'self-consistent':
            G_l, g_i, status = self.self_consistent_solver(x_l, N_i, W_il, **solverkwargs)
        else:
            raise ValueError("Requested solution technique not a recognized option.")

        self.G_l = G_l
        self.g_i = g_i

        return status

    ###############################
    # Checks                      #
    ###############################
    def check_data(self):
        """Verifies that x_l is not None, else raises RuntimeError.

        Raises:
            RuntimeError"""
        if self.x_l is None:
            raise RuntimeError("Data points not available.")

    def check_weights(self):
        """Verifies that g_i and G_l are not None, else raises RuntimeError.

        Raises:
            RuntimeError"""
        if self.g_i is None:
            raise RuntimeError("Window free energies not available.")
        if self.G_l is None:
            raise RuntimeError("Point weights not available.")

    ###############################
    # Post processing             #
    ###############################
    def reweight(self, beta, u_bias, g_i_bias=0):
        """Reweights sample weights to a biased ensemble. Does not change computed
        WHAM weights.

        Caution:
            This is a post-processing calculation, and needs to be performed after
            computing weights through `compute_point_weights`.

        Args:
            beta: beta: beta, in inverse units to the units of u_i(x).
            u_bias: Biasing function applied to order parameter x_l.
            g_i_bias: Total window free energy for window corresponding to biasing
                function. This determines how much to shift free energy profile down to match
                window. Default = 0, which means no shifting.

        Returns:
            G_l_bias (ndarray): Biased weights for each data point x_l.
        """
        self.check_data()
        self.check_weights()

        G_l_bias = self.G_l + beta * u_bias(self.x_l) - g_i_bias
        return G_l_bias

    def bin_betaF_profile(self, x_bin, G_l=None, bin_style='left'):
        """Bins weights corresponding to each sample into a 1D free energy
        profile. If weights are not passed as an argument, then computed WHAM weights are
        used for binning. Passing weights allows for computing reweighted
        free energy profiles.

        Caution:
            This calculation requires that the order parameter samples [self.x_l] (which the
            weights correspond to) are available.

        Args:
            x_bin (list): Array of bin left-edges/centers of length M. Used only for computing final PMF.
            G_l (ndarray): Array of weights corresponding to each data point/order parameter x_l.
            bin_style (string): 'left' or 'center'.

        Returns:
            betaF_bin (ndarray): Free energy profile, binned as per x_bin.
        """
        self.check_data()
        x_l = self.x_l
        if G_l is None:
            self.check_weights()
            G_l = self.G_l

        M = len(x_bin)

        # Compute bin edges
        bin_edges = np.zeros(M + 1)

        if bin_style == 'left':
            bin_edges[:-1] = x_bin
            # add an artificial edge after the last center
            bin_edges[-1] = x_bin[-1] + (x_bin[-1] - x_bin[-2])
        elif bin_style == 'center':
            # add an artificial edge before the first center
            bin_edges[0] = x_bin[0] - (x_bin[1] - x_bin[0]) / 2
            # add an artificial edge after the last center
            bin_edges[-1] = x_bin[-1] + (x_bin[-1] - x_bin[-2]) / 2
            # bin edges are at the midpoints of bin centers
            bin_edges[1:-1] = (x_bin[1:] + x_bin[:-1]) / 2
        else:
            raise ValueError('Bin style not recognized.')

        delta_x_bin = bin_edges[1:] - bin_edges[:-1]

        # Construct consensus free energy profiles by constructing weighted histogram
        # using individual point weights G_l obtained from solver
        betaF_bin = np.zeros(M)
        for b in range(1, M + 1):
            sel_mask = np.logical_and(x_l >= bin_edges[b - 1], x_l < bin_edges[b])

            if np.any(sel_mask != False):
                G_l_bin = G_l[sel_mask]
                betaF_bin[b - 1] = np.log(delta_x_bin[b - 1]) - numeric.clogsumexp(-G_l_bin)
            else:
                betaF_bin[b - 1] = np.inf

        return betaF_bin

    def bin_2D_betaF_profile(self, y_l, x_bin, y_bin, G_l=None, x_bin_style='left', y_bin_style='left'):
        """Bins weights corresponding to each sample into a 2D free energy
        profile, given a related order parameter y_l.
        If weights are not passed as an argument, then computed WHAM weights are
        used for binning. Passing weights allows for computing reweighted
        free energy profiles.

        Caution:
            This calculation requires that the order parameter samples [self.x_l] (which the
            weights correspond to) are available.

        Args:
            y_l (ndarray): Second dimension order parameter values.
            x_bin (list): Array of x-bin left-edges/centers of length M_x.
            y_bin (list): Array of y-bin left-edges/centers of length M_y.
            G_l (ndarray): Array of weights corresponding to each data point/order parameter x_l.
            x_bin_style (string): 'left' or 'center'.
            y_bin_style (string): 'left' or 'center'.

        Returns:
            tuple(betaF_bin, tuple(delta_x_bin, delta_y_bin))
                - betaF_bin (ndarray): 2-D free energy profile of shape (M_x, M_y), binned as per x_bin (1st dim) and y-bin (2nd dim).
                - delta_x_bin: Array of length M_x containing bin intervals along x.
                - delta_y_bin: Array of length M_y containing bin intervals along y.
        """
        self.check_data()
        x_l = self.x_l
        if G_l is None:
            self.check_weights()
            G_l = self.G_l

        M_x = len(x_bin)
        M_y = len(y_bin)

        # Compute x-bin edges
        x_bin_edges = np.zeros(M_x + 1)

        if x_bin_style == 'left':
            x_bin_edges[:-1] = x_bin
            # add an artificial edge after the last center
            x_bin_edges[-1] = x_bin[-1] + (x_bin[-1] - x_bin[-2])
        elif x_bin_style == 'center':
            # add an artificial edge before the first center
            x_bin_edges[0] = x_bin[0] - (x_bin[1] - x_bin[0]) / 2
            # add an artificial edge after the last center
            x_bin_edges[-1] = x_bin[-1] + (x_bin[-1] - x_bin[-2]) / 2
            # bin edges are at the midpoints of bin centers
            x_bin_edges[1:-1] = (x_bin[1:] + x_bin[:-1]) / 2
        else:
            raise ValueError('x-bin style not recognized.')

        delta_x_bin = x_bin_edges[1:] - x_bin_edges[:-1]

        # Compute y-bin edges
        y_bin_edges = np.zeros(M_y + 1)

        if y_bin_style == 'left':
            y_bin_edges[:-1] = y_bin
            # add an artificial edge after the last center
            y_bin_edges[-1] = y_bin[-1] + (y_bin[-1] - y_bin[-2])
        elif y_bin_style == 'center':
            # add an artificial edge before the first center
            y_bin_edges[0] = y_bin[0] - (y_bin[1] - y_bin[0]) / 2
            # add an artificial edge after the last center
            y_bin_edges[-1] = y_bin[-1] + (y_bin[-1] - y_bin[-2]) / 2
            # bin edges are at the midpoints of bin centers
            y_bin_edges[1:-1] = (y_bin[1:] + y_bin[:-1]) / 2
        else:
            raise ValueError('y-bin style not recognized.')

        delta_y_bin = y_bin_edges[1:] - y_bin_edges[:-1]

        # Construct consensus free energy profiles by constructing weighted histogram
        # using individual point weights G_l obtained from solver
        betaF_bin = np.zeros((M_x, M_y))

        for b_x in range(1, M_x + 1):
            for b_y in range(1, M_y + 1):
                sel_mask = np.logical_and.reduce((x_l >= x_bin_edges[b_x - 1],
                                                  x_l < x_bin_edges[b_x],
                                                  y_l >= y_bin_edges[b_y - 1],
                                                  y_l < y_bin_edges[b_y]))

                if np.any(sel_mask != False):
                    G_l_bin = G_l[sel_mask]
                    betaF_bin[b_x - 1, b_y - 1] = np.log(delta_x_bin[b_x - 1]) + np.log(delta_y_bin[b_y - 1]) - numeric.clogsumexp(-G_l_bin)
                else:
                    betaF_bin[b_x - 1, b_y - 1] = np.inf

        return betaF_bin, (delta_x_bin, delta_y_bin)

    def bin_second_betaF_profile(self, y_l, x_bin, y_bin, G_l=None, x_bin_style='left', y_bin_style='left'):
        """Bins weights corresponding to each sample into a 2D free energy
        profile of x_l related order parameter y_l, then integrates out x_l
        to get a free energy profile in terms of y_l.
        If weights are not passed as an argument, then computed WHAM weights are
        used for binning. Passing weights allows for computing reweighted
        free energy profiles.

        Caution:
            This calculation requires that the order parameter samples [self.x_l] (which the
            weights correspond to) are available.

        Args:
            y_l (ndarray): Second dimension order parameter values.
            x_bin (list): Array of x-bin left-edges/centers of length M_x.
            y_bin (list): Array of y-bin left-edges/centers of length M_y.
            G_l (ndarray): Array of weights corresponding to each data point/order parameter x_l.
            x_bin_style (string): 'left' or 'center'.
            y_bin_style (string): 'left' or 'center'.

        Returns:
            betaF_y (ndarray): Free energy profile of length M_y,
                binned as per y-bin (2nd dim).
        """
        betaF_xy, (delta_x_bin, delta_y_bin) = self.bin_2D_betaF_profile(y_l, x_bin, y_bin, G_l=G_l, x_bin_style=x_bin_style, y_bin_style=y_bin_style)
        betaF_xy = np.nan_to_num(betaF_xy)
        print(betaF_xy.shape)
        print(delta_x_bin.shape)
        betaF_y = np.zeros(len(y_bin))
        for yi in range(len(y_bin)):
            betaF_y[yi] = -numeric.clogsumexp(-betaF_xy[:, yi], b=delta_x_bin ** 2, axis=0)
        return betaF_y

    ###############################
    # Alternate API call          #
    ###############################
    def compute_betaF_profile(self, x_it, x_bin, u_i, beta, bin_style='left', solver='log-likelihood', **solverkwargs):
        """Computes the binned free energy profile and window total free energies.

        Args:
            x_it (list): Nested list of length S, x_it[i] is an array containing timeseries
                data from the i'th window.
            x_bin (list): Array of bin left-edges/centers of length M. Used only for computing final PMF.
            u_i (list): List of length S, u_i[i] is the umbrella potential function u_i(x)
                acting on the i'th window.
            beta: beta, in inverse units to the units of u_i(x).
            bin_style (string): 'left' or 'center'.
            solver (string): Solution technique to use ['log-likelihood', 'self-consistent', default='log-likelihood'].
            **solverkwargs: Arguments for solver.

        Returns:
            betaF_bin (ndarray): Free energy profile, binned as per x_bin.
            status (bool): Solver status.
        """
        # Unroll x_it into a single array
        x_l = x_it[0]
        for i in range(1, len(x_it)):
            x_l = np.hstack((x_l, x_it[i]))

        # Compute window counts
        N_i = np.array([len(arr) for arr in x_it])

        status = self.compute_point_weights(x_l, N_i, u_i, beta, solver=solver, **solverkwargs)
        betaF_bin = self.bin_betaF_profile(x_bin, bin_style='left')

        return betaF_bin, status
