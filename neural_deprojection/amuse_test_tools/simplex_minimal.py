import numpy as np

import time

from amuse.units import units, nbody_system, constants
from amuse.ext.molecular_cloud import molecular_cloud
from amuse.ext.evrard_test import body_centered_grid_unit_cube
from amuse.community.simplex.interface import SimpleX


Mcloud = 1000. | units.MSun
Rcloud = 3. | units.pc
Tcloud = 100. | units.K

N = 1000

sigma = 0.01 | units.pc

box_size = 10. | units.pc



converter = nbody_system.nbody_to_si(Mcloud, Rcloud)
gas_particles = molecular_cloud(targetN=N, convert_nbody=converter,
    base_grid=body_centered_grid_unit_cube).result

gas_particles.x += np.random.normal(size=len(gas_particles)) * sigma
gas_particles.y += np.random.normal(size=len(gas_particles)) * sigma
gas_particles.z += np.random.normal(size=len(gas_particles)) * sigma

gas_particles.flux = 0. | units.s**-1
gas_particles.u = (Tcloud*constants.kB) / (1.008*constants.u)
gas_particles.rho = Mcloud / ( 4./3.*np.pi * Rcloud**3 )

gas_particles.xion = 0.


radiative = SimpleX(redirection='none', number_of_workers=8)


radiative.parameters.blackbody_spectrum_flag = True
radiative.parameters.thermal_evolution_flag = True
radiative.parameters.box_size = box_size
radiative.parameters.timestep = 1. | 1e3*units.yr

radiative.particles.add_particles(gas_particles)


radiative.commit_particles()



start = time.time()

radiative.evolve_model( 0.05 | units.Myr )

end = time.time()

print ("Rad 1 in {a} s".format(a=end-start), flush=True)


start = time.time()

radiative.evolve_model( 0.1 | units.Myr )

end = time.time()

print ("Rad 2 in {a} s".format(a=end-start), flush=True)

radiative.stop()
