from mayavi import mlab
import pynbody as nb
import numpy as np
import h5py
import matplotlib.pyplot as plt

my_file1 = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Test/Clusterfind/Magneticum/Box2b_hr/snap_026/0/simcut/kurz0cv5y4ln1vmc/snap_026'
my_file2 = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Test/Clusterfind/Magneticum/Box2b_hr/snap_026/2/simcut/ruyswn41lioyz9gx/snap_026'
my_file3 = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Test/Clusterfind/Magneticum/Box2b_hr/snap_026/3/simcut/o149i3rbl4t01npk/snap_026'
jul_file = '/home/matthijs/PycharmProjects/TensorflowTest/hydro_gas_particles_i00905.hdf5'

f1 = nb.load(my_file1)
f2 = nb.load(my_file2)
f3 = nb.load(my_file3)

#Create a new 'SimSnap' object from an hdf5 file
hdf = h5py.File(jul_file, 'r')
data = hdf['particles']['0000000001']['attributes']
#print(list(data.keys()))
f4 = nb.new(gas=9577)
for name in list(data.keys()):
    f4.gas[name] = data[name]

# Not sure it's necessary to change the units, but here is code for it if needed
# print(data['density'].attrs.get('units'))
# f4.set_units_system(velocity='m s^-1', distance='m', mass='kg', temperature='K')

#This function changes the x, y and z coordinates so that they are centered around 0 and optionally offset from 0
def localize(sim, offset = [0.,0.,0.]):
    sim['x'] = (sim['x'] - (np.max(sim['x']) + np.min(sim['x'])) / 2) + offset[0]
    sim['y'] = (sim['y'] - (np.max(sim['y']) + np.min(sim['y'])) / 2) + offset[1]
    sim['z'] = sim['z'] - (np.max(sim['z']) + np.min(sim['z'])) / 2 + offset [2]

sim = f4.gas
print(list(sim.all_keys()))

localize(sim) #Magneticum snapshots are not centered around 0
av_z = True #average the quantity down line of sight, if False a slice at z=0 is taken unless z != 0 in the offset
qty1 = 'rho'
qty2 = 'u'
m = 1 #magnification
res = 500 #resolution
width = np.max(sim['x']) - np.min(sim['x']) #width of the image without magnification
filename = 'picture_file.png' #filename to store pictures, use 'filename' keyword in image function

# MAYAVI PLOTTING

#I can't get this to work for the hdf5 file yet, interpolating gives a grid of zeros

# volume1 = nb.plot.sph.volume(sim, resolution=200, qty = qty1, log = True, dynamic_range = 10)
# volume2 = nb.plot.sph.volume(sim, resolution=200, qty = qty2)

# mlab.show()

# PYNBODY PLOTTING

# img1 = nb.plot.sph.image(sim, qty=qty1, width=width/m, av_z=av_z, resolution=res)
# img2 = nb.plot.sph.velocity_image(sim, width=width/m, filename = filename, quiverkey = False)
img3 = nb.plot.sph.contour(sim, qty1, width=width/m, resolution=res, av_z=av_z, levels = 20)

plt.show()