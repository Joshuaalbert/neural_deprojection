import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from amuse.community.fi.interface import FiMap
from amuse.io import read_set_from_file
from amuse.units import units, nbody_system, constants


# C and O abundances in code
xC = 3.31e-4 * 10.**(8.15-8.55)
xO = 6.76e-4 * 10.**(8.50-8.87)

# C and O mass fractions
rho_frac_C = 12.*xC/(1. + xC*11. + xO*15.)
rho_frac_O = 16.*xO/(1. + xC*11. + xO*15.)


def column_density (gas_particles, N=480, box_size=10.|units.pc, mode='total'):
    '''
    Make a column density plot of a distribution of AMUSE gas particles. 
    Column density is volume density integrated along a line.

    gas_particles: gas particle set to make plot of, must at least have mass attribute
        (amuse particle set)
    N: number of pixels on each side (int)
    box_size: physical size of plotted region (scalar, unit length)
    mode: what to plot (str)

    current plotting modes:
    CO: carbon monoxide mass column density
    H2: molecular hydrogen mass column density
    H+: ionized hydrogen mass column density

    default: total gas mass column density
    '''

    converter = nbody_system.nbody_to_si(gas_particles.mass.sum(),
        gas_particles.position.lengths().max())

    mapper = FiMap(converter)

    # Compute column density over parallell lines-of-sight; effectively look from infinitely far
    mapper.parameters.projection_mode = 'parallel'

    # Physical size of imaged region
    mapper.parameters.image_width = box_size

    # Number of image pixels
    mapper.parameters.image_size = (N, N)

    # Coordinates of image center
    mapper.parameters.image_target = [0., 0., 0.] | units.parsec

    # Camera position (unit vector)
    mapper.parameters.projection_direction = [0,0,-1]

    # Image orientation
    mapper.parameters.upvector = [0,1,0]

    # Particles to include in image
    mapper.particles.add_particles(gas_particles)

    # Quantity to make image of
    # xCO, xmol, and xion are essentially 'how much of available material is locked up in
    # 'state', for molecular CO, molecular hydrogen, and ionized hydrogen, respectively
    if mode == 'CO':
        mapper.particles.weight = gas_particles.mass.value_in(units.MSun) * \
            gas_particles.xCO*(rho_frac_C + rho_frac_O)/2.
    elif mode == 'H2':
        mapper.particles.weight = gas_particles.mass.value_in(units.MSun) * \
            gas_particles.xmol
    elif mode == 'H+':
        mapper.particles.weight = gas_particles.mass.value_in(units.MSun) * \
            gas_particles.xion
    else:
        mapper.particles.weight = gas_particles.mass.value_in(units.MSun)

    # Make image; every pixel contains the total mass within that pixel
    image = mapper.image.pixel_value.T

    mapper.stop()


    # Dividing by pixel size gives mass column density
    bs = box_size.value_in(units.parsec)
    surface_density = image / (bs/N)**2

    if np.max(surface_density) == 0.:
        print ("[WARNING] image identically 0")


    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.imshow(surface_density, norm=LogNorm(), 
        extent=[-bs/2., bs/2., -bs/2., bs/2.], origin='lower', 
        vmin=np.max(surface_density)/1e2, vmax=np.max(surface_density))
    # Since the color scale is logarithmic and calibrated on the max, this does not work if
    # the image is identically zero!


    cbar = fig.colorbar(cax)
    cbar.set_label('Column Density [M$_\\odot$ pc$^{-2}$]')

    ax.set_xlabel('x [pc]')
    ax.set_ylabel('y [pc]')

    if mode == 'CO':
        ax.set_title('CO Column Density')
    elif mode == 'H2':
        ax.set_title('H$_2$ Column Density')
    elif mode == 'H+':
        ax.set_title('H$^+$ Column Density')
    else:
        ax.set_title('Gas Column Density')


if __name__ == '__main__':

    from amuse.datamodel import Particles

    ''' Check whether particle is in the correct position
    test_particle = Particles(1, mass=1.|units.MSun, u=1.|units.kms**2, radius=0.5|units.pc,
        vx=0.|units.kms, vy=0.|units.kms, vz=0.|units.kms,
        x=1.|units.pc, y=2.|units.pc, z=3.|units.pc)

    column_density(test_particle)
    '''

    folder = '/data2/wilhelm/sim_archive/gmc_star_formation/ColGMC_N1E4/'

    index = 905

    gas_particles = read_set_from_file(folder+'hydro_gas_particles_i{:05}.hdf5'.format(index),
        'hdf5')

    column_density(gas_particles)
    column_density(gas_particles, mode='CO')
    #column_density(gas_particles, mode='H+')
    column_density(gas_particles, mode='H2')

    plt.show()
