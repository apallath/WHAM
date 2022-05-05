"""
Implementation of binless WHAM (described in Shirts M., & Chodera J. D. (2008)) using
    - Negative log-likelihood maximization, inspired by Zhu, F., & Hummer, G. (2012), with automatic differentiation.
    - Self-consistent iteration.
"""
from functools import partial
import logging

from autograd import value_and_grad
import autograd.numpy as anp
import numpy as np
import scipy.optimize
from tqdm import tqdm

import WHAM.lib.potentials
import WHAM.lib.timeseries
from WHAM.lib import numeric

# Cython imports
cimport numpy as np

# Logging
logger = logging.getLogger(__name__)

################################################################################
#
# Core solver functionality
#
################################################################################

cdef class CalcBase:
    """
    Class containing basic binless WHAM solver functionality.

    Attributes:
        x_l (ndarray): 1-dimensional or 2-dimensional array containing unrolled order parameter
            which is being biased.
        G_l (ndarray): 1-dimensional array containing WHAM-computed weights corresponding to
            each (unrolled) order parameter.
        g_i (ndarray): 1-dimensional array of size N (=no of windows) containing
            WHAM-computed free energies for each window.

    Caution:
        All data must be at the same temperature.

    Note:
        Binless WHAM handles empty bins when computing free energy profiles by setting bin free energy to inf.
    """
    # Counter for log-likelihood minimizer
    cdef int _min_ctr

    # Output arrays
    cdef public np.ndarray x_l
    cdef public np.ndarray G_l
    cdef public np.ndarray g_i

    def __cinit__(self):
        self._min_ctr = 0
        self.x_l = None  # shape (N, D) or (N,)
        self.G_l = None  # shape (N,)
        self.g_i = None  # shape (S,)

    ############################################################################
    # IMP: Required for serialization.
    # Removing this will break pickling

    def __getstate__(self):
        return (self.x_l, self.G_l, self.g_i)

    def __setstate__(self, state):
        x_l, G_l, g_i = state
        self.x_l = x_l
        self.G_l = G_l
        self.g_i = g_i

    ############################################################################

    def NLL(self, g_i, N_i, W_il, autograd=True):
        """Computes the negative log-likelihood objective function to minimize.

        Args:
            g_i (ndarray of shape (S,)) Array of total free energies associated with the windows
                0, 1, 2, ..., S-1.
            N_i (ndarray of shape (S,)): Array of total sample counts for the windows
                0, 1, 2, ..., S-1.
            W_il (ndarray of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
                0, 1, 2, ..., S-1.
            autograd (bool): Use autograd to compute gradient.

        Returns:
            A(g) (np.float): Negative log-likelihood objective function.
        """
        if autograd:
            with anp.errstate(divide='ignore'):
                Ntot = anp.sum(N_i)
                term1 = -anp.sum(N_i / Ntot * g_i)
                term2 = 1 / Ntot * anp.sum(numeric.alogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis] / Ntot, axis=0))

                A = term1 + term2

                return A
        else:
            raise NotImplementedError()

    def _min_callback(self, g_i, args, logevery=100000000):
        if self._min_ctr % logevery == 0:
            logger.info("{:10d} {:.5f}".format(self._min_ctr, self.NLL(g_i, *args)))
        self._min_ctr += 1

    def minimize_NLL_solver(self, N_i, W_il, g_i=None, opt_method='L-BFGS-B', logevery=100000000, autograd=True):
        """Computes optimal g_i by minimizing the negative log-likelihood
        for jointly observing the bin counts in the independent windows in the dataset.

        Note:
            Any optimization method which scipy.optimize supports can be used. L-BFGS-B is used
            by default. Gradient information required for L-BFGS-B can either be computed
            analytically or using autograd.

        Args:
            N_i (ndarray of shape (S,)): Array of total sample counts for the windows
                0, 1, 2, ..., S-1.
            W_il (ndarray of shape (S, Ntot)): Array of weights, W_il = -beta * U_i(x_l) for the windows
                0, 1, 2, ..., S-1.
            g_i (ndarray of shape (S,)): Total free energy initial guess.
            opt_method (string): Optimization algorithm to use (default: L-BFGS-B).
            logevery (int): Interval to log negative log-likelihood (default=100000000, i.e. no logging).
            autograd (bool): Use autograd to compute gradient.

        Returns:
            tuple(bG_l, g_i, status)
                - bG_l (ndarray of shape (Ntot,)): Free energy for each sample point,
                - g_i (ndarray of shape (S,)): Total free energy for each window,
                - status (bool): Solution status.
        """
        if g_i is None:
            g_i = np.random.rand(len(N_i))

        # Optimize
        logger.debug("      Iter NLL")
        self._min_ctr = 0

        if autograd:
            res = scipy.optimize.minimize(value_and_grad(self.NLL), g_i, jac=True,
                                          args=(N_i, W_il, True),
                                          method=opt_method,
                                          callback=partial(self._min_callback, args=(N_i, W_il, True), logevery=logevery))
        else:
            raise NotImplementedError()

        self._min_callback(res.x, args=(N_i, W_il, autograd), logevery=1)

        g_i = res.x
        g_i = g_i - g_i[0]

        G_l = numeric.clogsumexp(g_i[:, np.newaxis] + W_il, b=N_i[:, np.newaxis], axis=0)

        return G_l, g_i, res.success

    def self_consistent_solver(self, N_i, W_il, g_i=np.zeros(1), tol=1e-7, maxiter=100000, logevery=100000000):
        """Computes optimal parameters g_i by solving the coupled MBAR equations self-consistently
        until convergence.

        Args:
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
        cdef int Ntot = N_i.sum()

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
                    logger.info("Self-consistent solver error = {:.8e}.".format(tol_check))

            if increment[np.argmax(increment)] < tol:
                if logevery > 0:
                    logger.info("Self-consistent solver error = {:.8e}.".format(tol_check))
                status = True
                break

        return G_l, g_i, status

    ############################################################################
    # Computations on point data
    ############################################################################

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
        Ntot = x_l.shape[0]  # (N, D) or (N, 1)

        # Compute weights for each data point
        W_il = np.zeros((S, Ntot))
        for i in range(S):
            W_il[i, :] = -beta * u_i[i](x_l)

        # Get free energy profile through binless WHAM/MBAR
        if solver == 'log-likelihood':
            G_l, g_i, status = self.minimize_NLL_solver(N_i, W_il, **solverkwargs)
        elif solver == 'self-consistent':
            G_l, g_i, status = self.self_consistent_solver(N_i, W_il, **solverkwargs)
        else:
            raise ValueError("Requested solution technique not a recognized option.")

        self.G_l = G_l
        self.g_i = g_i

        return status

    def reweight(self, beta, u_bias):
        """Reweights sample weights to a biased ensemble. Does not change computed
        WHAM weights.

        Caution:
            This is a post-processing calculation, and needs to be performed after
            computing weights through `compute_point_weights` or through the main
            API call `compute_betaF_profile`.

        Args:
            beta: beta: beta, in inverse units to the units of u_i(x).
            u_bias: Biasing function applied to order parameter x_l.

        Returns:
            G_l_bias (ndarray): Biased weights for each data point x_l.
        """
        self.check_data()
        self.check_weights()

        # Compute g_bias: total window free energy for applied bias potential
        w_bias_l = -beta * u_bias(self.x_l)

        g_bias = -numeric.clogsumexp(w_bias_l - self.G_l)

        G_l_bias = self.G_l + beta * u_bias(self.x_l) - g_bias

        return G_l_bias

    ############################################################################
    # Checks
    ############################################################################

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


################################################################################
#
# Biasing one order parameter
#
################################################################################

cdef class Calc1D(CalcBase):
    """Class containing methods to compute free energy profiles from 1D umbrella sampling data
    (i.e. data from biasing a single order parameter) using binless WHAM.

    Attributes:
        x_l (ndarray): 1-dimensional array containing unrolled order parameter
            which is being biased.
        G_l (ndarray): 1-dimensional array containing WHAM-computed weights corresponding to
            each (unrolled) order parameter.
        g_i (ndarray): 1-dimensional array of size N (=no of windows) containing
            WHAM-computed free energies for each window.

    Caution:
        All data must be at the same temperature.

    Note:
        Binless WHAM handles empty bins when computing free energy profiles by setting bin free energy to inf.

    Example:
        >>> calc = Calc1D()
        >>> status = calc.compute_point_weights(x_l, ...)
        >>> status
        True
        >>> G_l = calc.G_l
        >>> g_i = calc.g_i
        >>> betaF_x, _ = calc.bin_betaF_profile(x_l, G_l, x_bin, ...)
        >>> betaF_xy, _ = calc.bin_2D_betaF_profile(x_l, y_l, G_l, x_bin, y_bin, ...)

        For comprehensive examples, check out the Jupyter notebooks in the `examples/` folder.
    """

    ############################################################################
    # Binning point weights
    ############################################################################

    def bin_betaF_profile(self, x_bin, G_l=None, bin_style='left'):
        """Bins weights corresponding to each sample into a 1D free energy profile.
        If point weights G_l are not passed as an argument, then the computed WHAM weights are
        used for binning. You can pass custom weights G_l to compute reweighted
        free energy profiles.

        Caution:
            This calculation uses the order parameter samples [self.x_l]. These will be
            available if you have called the compute function `compute_point_weights`
            or the main API call `compute_betaF_profile`. If you haven't done so, you must initialize
            the Calc1D object's x_l variable before calling this function.

        Args:
            x_bin (list): Array of bin left-edges/centers of length M. Used only for computing final PMF.
            G_l (ndarray): Array of weights corresponding to each data point/order parameter x_l.
            bin_style (string): 'left' or 'center'.

        Returns:
            betaF_bin (ndarray): Free energy profile, binned as per x_bin.
            betaF_bin_counts (ndarray): Bin counts
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
        betaF_bin_counts = np.zeros(M)

        for b in range(1, M + 1):
            sel_mask = np.logical_and(x_l >= bin_edges[b - 1], x_l < bin_edges[b])
            betaF_bin_counts[b - 1] = np.sum(sel_mask)

            if np.any(sel_mask != False):
                G_l_bin = G_l[sel_mask]
                betaF_bin[b - 1] = np.log(delta_x_bin[b - 1]) - numeric.clogsumexp(-G_l_bin)
            else:
                betaF_bin[b - 1] = np.inf

        return betaF_bin, betaF_bin_counts

    ############################################################################
    # One-step API call to compute free energy profile
    ############################################################################

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
        betaF_bin, betaF_bin_counts = self.bin_betaF_profile(x_bin, bin_style='left')

        return betaF_bin, betaF_bin_counts, status

    ############################################################################
    # 2D reweighting API calls
    ############################################################################

    def bin_2D_betaF_profile(self, y_l, x_bin, y_bin, G_l=None, x_bin_style='left', y_bin_style='left'):
        """Bins weights corresponding to each sample into a 2D free energy profile in order parameters x_l (which is biased) and y_l (a related unbiased order parameter).
        If point weights G_l are not passed as an argument, then the computed WHAM weights are
        used for binning. You can pass custom weights G_l to compute reweighted
        free energy profiles.

        Caution:
            This calculation uses the order parameter samples [self.x_l]. These will be
            available if you have called the compute function `compute_point_weights`
            or the main API call `compute_betaF_profile`. If you haven't done so, you must initialize
            the Calc1D object's x_l variable before calling this function.

        Args:
            y_l (ndarray): Second dimension order parameter values.
            x_bin (list): Array of x-bin left-edges/centers of length M_x.
            y_bin (list): Array of y-bin left-edges/centers of length M_y.
            G_l (ndarray): Array of weights corresponding to each data point/order parameter x_l.
            x_bin_style (string): 'left' or 'center'.
            y_bin_style (string): 'left' or 'center'.

        Returns:
            tuple(betaF_bin, tuple(betaF_bin_counts, delta_x_bin, delta_y_bin))
                - betaF_bin (ndarray): 2-D free energy profile of shape (M_x, M_y), binned using x_bin (1st dim) and y-bin (2nd dim).
                - betaF_bin_counts (ndarray): 2-D bin counts of shape (M_x, M_y)
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
        betaF_bin_counts = np.zeros((M_x, M_y))

        for b_x in range(1, M_x + 1):
            for b_y in range(1, M_y + 1):
                sel_mask = np.logical_and.reduce((x_l >= x_bin_edges[b_x - 1],
                                                  x_l < x_bin_edges[b_x],
                                                  y_l >= y_bin_edges[b_y - 1],
                                                  y_l < y_bin_edges[b_y]))
                betaF_bin_counts[b_x - 1, b_y - 1] = np.sum(sel_mask)

                if np.any(sel_mask != False):
                    G_l_bin = G_l[sel_mask]
                    betaF_bin[b_x - 1, b_y - 1] = np.log(delta_x_bin[b_x - 1]) + np.log(delta_y_bin[b_y - 1]) - numeric.clogsumexp(-G_l_bin)
                else:
                    betaF_bin[b_x - 1, b_y - 1] = np.inf

        return betaF_bin, (betaF_bin_counts, delta_x_bin, delta_y_bin)

    def bin_second_betaF_profile(self, y_l, x_bin, y_bin, G_l=None, x_bin_style='left', y_bin_style='left'):
        """Bins weights corresponding to each sample into a into a 2D free energy profile in x_l (which is biased) and y_l (a related unbiased order parameter), then integrates out x_l
        to get a free energy profile in y_l. If point weights G_l are not passed as an argument, then the computed WHAM weights are
        used for binning. You can pass custom weights G_l to compute reweighted
        free energy profiles.

        Caution:
            This calculation uses the order parameter samples [self.x_l]. These will be
            available if you have called the compute function `compute_point_weights`
            or the main API call `compute_betaF_profile`. If you haven't done so, you must initialize
            the Calc1D object's x_l variable before calling this function.

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
        betaF_xy, (betaF_bin_counts, delta_x_bin, delta_y_bin) = self.bin_2D_betaF_profile(y_l, x_bin, y_bin, G_l=G_l, x_bin_style=x_bin_style, y_bin_style=y_bin_style)
        betaF_xy = np.nan_to_num(betaF_xy)
        logger.debug(betaF_xy.shape)
        logger.debug(delta_x_bin.shape)
        betaF_y = np.zeros(len(y_bin))
        for yi in range(len(y_bin)):
            betaF_y[yi] = -numeric.clogsumexp(-betaF_xy[:, yi], b=delta_x_bin ** 2, axis=0)
        return betaF_y


################################################################################
#
# Biasing D order parameters
#
################################################################################


cdef class CalcDD(CalcBase):
    """Class containing methods to compute free energy profiles from D-dimensional umbrella sampling data
    (i.e. data from biasing D order parameters) using binless WHAM.

    Attributes:
        x_l (ndarray): DxN matrix containing unrolled order parameter
            which is being biased, where (D=no of dimensions, N=no of data points).
        G_l (ndarray): 1-dimensional array of length N (=no of data points) containing WHAM-computed weights corresponding to
            each (unrolled) order parameter.
        g_i (ndarray): 1-dimensional array of size S (=no of windows) containing
            WHAM-computed free energies for each window.

    Caution:
        All data must be at the same temperature.

    Note:
        Binless WHAM handles empty bins when computing free energy profiles by setting bin free energy to inf.

    Example:
        >>> calc = CalcDD()
        >>> status = calc.compute_point_weights(X_l, ...)
        >>> status
        True
        >>> G_l = calc.G_l
        >>> g_i = calc.g_i
        >>> betaF_x, _ = calc.bin_betaF_profile(x_l, G_l, X_bin, ...)

        For comprehensive examples, check out the Jupyter notebooks in the `examples/` folder.
    """

    ############################################################################
    # Binning point weights
    ############################################################################

    def bin_betaF_profile(self, x_bin, G_l=None):
        """Bins weights corresponding to each sample into a D-dimensional free energy profile.
        If point weights G_l are not passed as an argument, then the computed WHAM weights are
        used for binning. You can pass custom weights G_l to compute reweighted
        free energy profiles.

        Caution:
            This calculation uses the order parameter samples [self.x_l]. These will be
            available if you have called the compute function `compute_point_weights`
            or the main API call `compute_betaF_profile`. If you haven't done so, you must initialize
            the Calc1D object's x_l variable before calling this function.

        Args:
            x_bin (ndarray of dimensions (Nbin, D)): List of arrays of bin left-edges of length M. Used only for computing final PMF.
            G_l (ndarray): Array of weights corresponding to each data point/order parameter x_l.

        Returns:
            betaF_bin (ndarray): Free energy profile, binned as per x_bin.
            betaF_bin_counts (ndarray): Bin counts
        """
        raise NotImplementedError()

    ############################################################################
    # One-step API call to compute free energy profile
    ############################################################################

    def compute_betaF_profile(self, x_it, bin_list, u_i, beta, bin_style='left', solver='log-likelihood', **solverkwargs):
        raise NotImplementedError()
