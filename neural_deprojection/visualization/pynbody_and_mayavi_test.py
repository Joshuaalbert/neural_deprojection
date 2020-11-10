from mayavi import mlab
import pynbody as nb
import numpy as np
import h5py
import pysph.tools.interpolator as interp
from pysph.base.particle_array import ParticleArray
from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction
import pysph.base
from mayavi.tools.pipeline import scalar_field
import matplotlib.pyplot as plt

my_file1 = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Test/Clusterfind/Magneticum/Box2b_hr/snap_026/0/simcut/kurz0cv5y4ln1vmc/snap_026'
my_file2 = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Test/Clusterfind/Magneticum/Box2b_hr/snap_026/2/simcut/ruyswn41lioyz9gx/snap_026'
my_file3 = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Test/Clusterfind/Magneticum/Box2b_hr/snap_026/3/simcut/o149i3rbl4t01npk/snap_026'
jul_file = '/home/matthijs/PycharmProjects/TensorflowTest/hydro_gas_particles_i00905.hdf5'

f1 = nb.load(my_file1)
f2 = nb.load(my_file2)
f3 = nb.load(my_file3)

hdf = h5py.File(jul_file, 'r')

print(type(f3))
data = hdf['particles']['0000000001']['attributes']
print(data['x'].attrs.get('units'))
f4 = nb.new(gas=9577)
print(list(data.keys()))
print('='*20)
for name in list(data.keys()):
    # print(data[name].attrs.get('units'))
    f4.gas[name] = data[name]
f4.set_units_system(velocity='m s^-1', distance='m', mass='kg', temperature='K')
print(list(f4.gas.all_keys()))
# print(data['temp'][:10])
print(data['u'][:10])
print(f4.gas['smooth'][:10])

# print(type(f1))

f = f4
# f.physical_units()
sim = f.gas
print(list(sim.all_keys()))

av_z = True

qty1 = 'radius'
qty2 = 'u'
m = 1 #magnification
o = [0., 0., 0.] #offset
res = 500 #resolution
filename = 'radius.png'

# volume1 = nb.plot.sph.volume(f4.gas, resolution=200, qty = qty1, log = True, dynamic_range = 10)
# # volume2 = nb.plot.sph.volume(f.gas, resolution=200, qty = qty2)
#
max_x = np.max(sim['x'])
min_x = np.min(sim['x'])
print(max_x, min_x)

img1 = nb.plot.sph.image(sim, qty=qty1, av_z=av_z, width=max_x - min_x, resolution=res, magnification=m, offset=o, filename = filename)
# img2 = nb.plot.sph.velocity_image(sim, width=max_x - min_x, magnification=m, filename = filename, quiverkey = False)
# img3 = nb.plot.sph.contour(sim, qty1, width=max_x - min_x, resolution=res, av_z=av_z, levels = 20, filename = filename)

plt.show()

# mlab.show()