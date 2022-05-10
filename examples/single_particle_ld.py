"""
Classes for performing langevin dynamics simulations on 2D potential energy surfaces.
"""
import multiprocessing
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from openmm import unit
from openmm import openmm
from tqdm import tqdm

################################################################################
#
# Potentials
#
################################################################################


class Potential2D(openmm.CustomExternalForce):
    """
    Abstract class defining basic 2D potential behavior.

    Child classes can be used as CustomExternalForce in OpenMM.
    A harmonic restraining potential of magnitude 1000 kJ/mol is applied on the
    z coordinates about z=0.

    Note:
        Child classes must call super.__init__() only after initializing the force
        attribute in x and y variables.

    Attributes:
        force (str): `OpenMM-compatible custom force expression`_.

    .. _OpenMM-compatible custom force expression:
       http://docs.openmm.org/latest/userguide/theory/03_custom_forces.html#writing-custom-expressions
    """
    def __init__(self):
        # Apply restraining potential along z direction
        # Child classes will add terms for x and y and initialize this force expression
        self.force += " + 1000 * z^2"

        # Print force expression
        print("[Potential] Initializing potential with expression:\n" + self.force)

        # Initialize force expression
        super().__init__(self.force)

    def potential(self, x: float, y: float):
        """
        Computes the potential at a given point (x, y).

        Args:
            x (float): x-coordinate of the point to compute potential at.
            y (float): y-coordinate of the point to compute potential at.

        Returns:
            V (float): Value of potential at (x, y).
        """
        # Child classes will implement this method.
        raise NotImplementedError()


class TwoBasinPotential2D(Potential2D):
    r"""
    2-basin potential varuing in the $x$ coordinate.
    $$U(x, y) = M \left( (x^2 - 1)^2 + y^2 \right)$$
    """
    def __init__(self, magnitude=1):
        self.magnitude = magnitude

        self.force = '''{magnitude} * ((x^2 - 1)^2 + y^2)'''.format(magnitude=self.magnitude)

        super().__init__()

    def potential(self, x, y):
        return self.magnitude * ((x ** 2 - 1) ** 2 + y ** 2)


class SlipBondPotential2D(Potential2D):
    r"""
    2-basin potential varying in both the $x$ and the $y$ coordinates.
    $$U(x, y) = \left( \left(\frac{(y - y\_0)^2}{y\_scale} - y\_shift \right)^2 + \frac{(x - y - xy\_0)^2}{xy\_scale} \right)$$
    """
    def __init__(self, y_0=1, y_scale=5, y_shift=4, xy_0=0, xy_scale=2):
        self.y_0 = y_0
        self.y_scale = y_scale
        self.y_shift = y_shift
        self.xy_0 = xy_0
        self.xy_scale = xy_scale

        constvals = {"y_0": self.y_0,
                     "y_scale": self.y_scale,
                     "y_shift": self.y_shift,
                     "xy_0": self.xy_0,
                     "xy_scale": self.xy_scale}

        self.force = '''((y - {y_0})^2 / {y_scale} - {y_shift})^2 + (x - y - {xy_0})^2 / {xy_scale}'''.format(**constvals)

        super().__init__()

    def potential(self, x, y):
        """Computes the slip bond potential at a given point (x, y)."""
        return ((y - self.y_0) ** 2 / self.y_scale - self.y_shift) ** 2 + (x - y - self.xy_0) ** 2 / self.xy_scale


################################################################################
#
# Biases
#
################################################################################


class HarmonicBias(openmm.CustomExternalForce):
    """

    """
    def __init__(self, kappa_x=0, x_0=0, kappa_y=0, y_0=0):
        self.kappa_x = kappa_x
        self.x_0 = x_0
        self.kappa_y = kappa_y
        self.y_0 = y_0

        constvals = {"kappa_x": self.kappa_x,
                     "x_0": self.x_0,
                     "kappa_y": self.kappa_y,
                     "y_0": self.y_0}

        self.force = "{kappa_x} / 2 * (x - {x_0})^2 + {kappa_y} / 2 * (y - {y_0})^2 + 1000 * z^2".format(**constvals)

        # Print force expression
        print("[Bias] Initializing bias with expression:\n" + self.force)

        super().__init__(self.force)


################################################################################
#
# Simulation
#
################################################################################


class SingleParticleSimulation:
    """
    Performs langevin dynamics simulation of a particle on a potential energy surface.

    Attributes:
        potential (openmm.CustomExternalForce): Underlying potential energy surface.
        mass (int): Mass of particle on the surface, in dalton (default = 1).
        temp (float): Temperature, in Kelvin (default = 300).
        friction (float): Friction factor, in ps^-1 (default = 100).
        timestep (float): Timestep, in fs (default = 10).
        init_state (openmm.State): Initial state for re-runs (default = None).
        init_coord (np.ndarray): Initial coordinates of the particle on the surface (default = [[0, 0, 0]]).
        gpu (bool): If True, uses GPU for simulation (default = False).
        cpu_threads (int): If gpu is False, number of CPU threads to use for simulation. If None, the max cpu count is used. (default = None).
        seed (int): Seed for reproducibility (default = None).
        traj_in_mem (bool): If True, stores trajectory in memory. The trajectory can be accessed by the object's `traj` attribute (default=False).

    Arguments:
        nsteps (int): Number of steps to run simulation for (default = 1000)
        chkevery (int): Checkpoint interval (default = 500).
        trajevery (int): Trajectory output interval (default = 1).
        energyevery (int): Energy output interval (default = 1).
        chkfile (str): File to write checkpoints to (default = "./chk_state.pkl"). If this file already exists, it will be overwritten.
        trajfile (str): File to write trajectory data to (default = "./traj.dat"). If this file already exists, it will be overwritten.
        energyfile (str): File to write energies to (default = "./energies.dat"). If this file already exists, it will be overwritten.
    """
    def __init__(self,
                 potential: openmm.CustomExternalForce,
                 mass: int = 1,
                 temp: float = 300,
                 friction: float = 100,
                 timestep: float = 10,
                 init_state: openmm.State = None,
                 init_coord: np.ndarray = np.array([0, 0, 0]).reshape((1, 3)),
                 gpu: bool = False,
                 cpu_threads: int = None,
                 seed: int = None,
                 traj_in_mem: bool = False,
                 bias: openmm.CustomExternalForce = None):
        # Properties
        self.mass = mass * unit.dalton  # mass of particles
        self.temp = temp * unit.kelvin  # temperature
        self.friction = friction / unit.picosecond  # LD friction factor
        self.timestep = timestep * unit.femtosecond   # LD timestep

        self.init_state = init_state
        self.gpu = gpu
        self.traj_in_mem = traj_in_mem

        # Init simulation objects
        self.system = openmm.System()
        self.system.addParticle(self.mass)
        self.potential = potential
        self.potential.addParticle(0, [])  # no parameters associated with each particle
        self.system.addForce(self.potential)

        # Add bias
        if bias is not None:
            self.bias = bias
            self.bias.addParticle(0, [])  # no parameters associated with each particle
            self.system.addForce(bias)

        self.integrator = openmm.LangevinIntegrator(self.temp,
                                                    self.friction,
                                                    self.timestep)

        if seed is not None:
            self.integrator.setRandomNumberSeed(seed)

        if self.gpu:
            platform = openmm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            print("Running simulation on GPU.")
        else:
            platform = openmm.Platform.getPlatformByName('CPU')
            if cpu_threads is None:
                cpu_threads = multiprocessing.cpu_count()
            properties = {'Threads': str(cpu_threads)}
            print("Running simulation on {} CPU threads.".format(cpu_threads))

        self.context = openmm.Context(self.system, self.integrator, platform, properties)

        # Init state
        if init_state is None:
            self.context.setPositions(init_coord)
            if seed is not None:
                self.context.setVelocitiesToTemperature(self.temp, randomSeed=seed)
            else:
                self.context.setVelocitiesToTemperature(self.temp)
        else:
            self.context.setState(init_state)

    def __call__(self,
                 nsteps: int = 1000,
                 chkevery: int = 500,
                 trajevery: int = 1,
                 energyevery: int = 1,
                 chkfile="./chk_state.pkl",
                 trajfile="./traj.dat",
                 energyfile="./energies.dat"):

        if self.traj_in_mem:
            self.traj = None

        for i in tqdm(range(nsteps)):
            # Checkpoint
            if i > 0 and i % chkevery == 0:
                self._dump_state(chkfile, i)

            # Store positions
            if i % trajevery == 0:
                self._write_trajectory(trajfile, i)

            # Store energy
            if i % energyevery == 0:
                self._write_energies(energyfile, i)

            # Integrator step
            self.integrator.step(1)

        # Finalize
        self._dump_state(chkfile, i)

    def _dump_state(self, ofilename, i):
        t = i * self.timestep / unit.picosecond
        print("Checkpoint at {:10.7f} ps".format(t))

        state = self.context.getState(getPositions=True, getVelocities=True)

        with open(ofilename, "wb") as fh:
            pickle.dump(state, fh)

    def _write_trajectory(self, ofilename, i):
        t = i * self.timestep / unit.picosecond
        pos = self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Store trajectory in memory
        if self.traj_in_mem:
            if i == 0:
                self.traj = pos
            else:
                self.traj = np.vstack((self.traj, pos))

        # Write trajectory to disk
        if i == 0:
            with open(ofilename, "w") as of:
                of.write("# t[ps]    x [nm]    y [nm]    z[nm]\n")

        with open(ofilename, "a") as of:
            of.write("{:10.5f}\t{:10.7f}\t{:10.7f}\t{:10.7f}\n".format(t, pos[0, 0], pos[0, 1], pos[0, 2]))

    def _write_energies(self, ofilename, i):
        t = i * self.timestep / unit.picosecond
        PE = self.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
        KE = self.context.getState(getEnergy=True).getKineticEnergy() / unit.kilojoule_per_mole

        if i == 0:
            with open(ofilename, "w") as of:
                of.write("# t[ps]    PE [kJ/mol]    KE [kJ/mol]\n")

        with open(ofilename, "a") as of:
            of.write("{:10.5f}\t{:10.7f}\t{:10.7f}\n".format(t, PE, KE))


################################################################################
#
# Reading trajectory data
#
################################################################################


class TrajectoryReader:
    """Utility class for reading large trajectories.

    Args:
        traj_file (str): Path to trajectory file.
        comment_char (str): Character marking the beginning of a comment line.
        format (str): Format of each line (options = 'txyz' or 'xyz'; default = 'txyz')
    """
    def __init__(self, traj_file, comment_char='#', format='txyz'):
        self.traj_file = traj_file
        self.comment_char = comment_char
        self.format = format

    def read_traj(self, skip=1):
        """
        Reads trajectory.

        Args:
            skip (int): Number of frames to skip between reads (default = 1).

        Returns:
            tuple(T, traj) if self.format == 'txyz'
            traj if self.format == 'xyz'
        """
        if self.format == 'txyz':
            return self._read_traj_txyz(skip)
        elif self.format == 'xyz':
            return self._read_traj_xyz(skip)
        else:
            raise ValueError('Invalid format {}'.format(format))

    def _read_traj_txyz(self, skip):
        times = []
        traj = []

        count = 0
        with open(self.traj_file, 'r') as trajf:
            for line in trajf:
                if line.strip()[0] != self.comment_char:
                    if count % skip == 0:
                        txyz = [float(c) for c in line.strip().split()]
                        times.append(txyz[0])
                        traj.append([txyz[1], txyz[2], txyz[3]])

        return np.array(times), np.array(traj)

    def _read_traj_xyz(self, skip):
        traj = []

        count = 0
        with open(self.traj_file, 'r') as trajf:
            for line in trajf:
                if line.strip()[0] != self.comment_char:
                    if count % skip == 0:
                        xyz = [float(c) for c in line.strip().split()]
                        traj.append([xyz[0], xyz[1], xyz[2]])

        return np.array(traj)


################################################################################
#
# Visualization
#
################################################################################


class VisualizePotential2D:
    """
    Class defining functions to generate scatter plots and animated trajectories
    of a particle on a 2D potential surface.

    Args:
        potential2D (pyib.md.potentials.Potential2D): 2D potential energy surface.
        temp (float): Temperature (required, as free energies are plotted in kT).
        xrange (tuple of length 2): Range of x-values to plot.
        yrange (tuple of length 2): Range of y-values to plot.
        contourvals (int or array-like): Determines the number and positions of the contour lines / regions. Refer to the `matplotlib documentation`_ for details.
        clip (float): Value of free energy (in kT) to clip contour plot at.
        mesh: Number of mesh points in each dimension for contour plot.
        cmap: Matplotlib colormap.

    .. _matplotlib documentation: https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.contour.html
    """
    def __init__(self,
                 potential2D: Potential2D,
                 temp: float,
                 xrange: tuple,
                 yrange: tuple,
                 contourvals=None,
                 clip=None,
                 mesh: int = 200,
                 cmap: str = 'jet'):
        self.potential2D = potential2D
        self.kT = 8.3145 / 1000 * temp
        self.xrange = xrange
        self.yrange = yrange
        self.contourvals = contourvals
        self.clip = clip
        self.mesh = mesh
        self.cmap = cmap

    def plot_potential(self):
        """
        Plots the potential within (xrange[0], xrange[1]) and (yrange[0], yrange[1]).
        """
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)

        if self.clip is not None:
            V = v.reshape(self.mesh, self.mesh) / self.kT
            V = V.clip(max=self.clip)
        else:
            V = v.reshape(self.mesh, self.mesh) / self.kT

        fig, ax = plt.subplots(dpi=150)
        if self.contourvals is not None:
            cs = ax.contourf(xx, yy, V, self.contourvals, cmap=self.cmap)
        else:
            cs = ax.contourf(xx, yy, V, cmap=self.cmap)
        cbar = fig.colorbar(cs)
        cbar.set_label(r"Potential ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return (fig, ax)

    def plot_projection_x(self):
        """
        Plots the x-projection of potential within (xrange[0], xrange[1])
        and (yrange[0], yrange[1]).
        """
        # Compute 2D free energy profile
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)
        V = v.reshape(self.mesh, self.mesh) / self.kT

        # Integrate over y-coordinate to get free-energy along x-coordinate
        Fx = -logsumexp(-V, axis=0)
        Fx = Fx - np.min(Fx)
        x = np.linspace(self.xrange[0], self.xrange[1], self.mesh)

        # Plot
        fig, ax = plt.subplots(dpi=150)
        ax.plot(x, Fx)
        ax.set_ylabel(r"Potential ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylim([0, None])
        return (fig, ax, x, Fx)

    def plot_projection_y(self):
        """
        Plots the y-projection of potential within (xrange[0], xrange[1])
        and (yrange[0], yrange[1]).
        """
        # Compute 2D free energy profile
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)
        V = v.reshape(self.mesh, self.mesh) / self.kT

        # Integrate over y-coordinate to get free-energy along x-coordinate
        Fy = -logsumexp(-V, axis=1)
        Fy = Fy - np.min(Fy)
        y = np.linspace(self.yrange[0], self.yrange[1], self.mesh)

        # Plot
        fig, ax = plt.subplots(dpi=150)
        ax.plot(y, Fy)
        ax.set_ylabel(r"Potential ($k_B T$)")
        ax.set_xlabel("$y$")
        ax.set_ylim([0, None])
        return (fig, ax, y, Fy)

    def scatter_traj(self, traj, every=1, s=1, c='black'):
        """
        Scatters entire trajectory onto potential energy surface.
        Args:
            traj (numpy.ndarray): Array of shape (T, 3) containing the (x, y, z) coordinates at each timestep.
            outimg (str): Filename of the output image.
            every (int): Interval to plot point at (default = 1).
            s (int): Size of points (default = 1).
            c (string): Color of points (default = 'black').
        """
        fig, ax = self.plot_potential()
        ax.scatter(traj[::every, 0], traj[::every, 1], s=s, c=c)
        return (fig, ax)

    def scatter_traj_projection_x(self, traj, every=1, s=1, c='black'):
        """
        Scatters x-projection of entire trajectory onto potential energy surface.
        Args:
            traj (numpy.ndarray): Array of shape (T, 3) containing the (x, y, z) coordinates at each timestep.
            outimg (str): Filename of the output image.
            every (int): Interval to plot point at (default = 1).
            s (int): Size of points (default = 1).
            c (str): Color of points (default = 'black').
        """
        fig, ax, x, Fx = self.plot_projection_x()
        for i in tqdm(range(0, traj.shape[0], every)):
            xpt = traj[i, 0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
        return (fig, ax)

    def scatter_traj_projection_y(self, traj, every=1, s=1, c='black'):
        """
        Scatters x-projection of entire trajectory onto potential energy surface.
        Args:
            traj (numpy.ndarray): Array of shape (T, 3) containing the (x, y, z) coordinates at each timestep.
            outimg (str): Filename of the output image.
            every (int): Interval to plot point at (default = 1).
            s (int): Size of points (default = 1).
            c (str): Color of points (default = 'black').
        """
        fig, ax, y, Fy = self.plot_projection_y()
        for i in tqdm(range(0, traj.shape[0], every)):
            xpt = traj[i, 1]
            xloc = np.argmin((y - xpt)**2)
            ypt = Fy[xloc]
            ax.scatter(xpt, ypt, s=s, c=c)
        return (fig, ax)
