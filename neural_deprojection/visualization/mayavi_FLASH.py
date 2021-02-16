from mayavi import mlab

import yt
import numpy as np
from timeit import default_timer
from itertools import product
from scipy import interpolate

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull

# import itertools

# quicker way of interpolating. found here:
# https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

# def interp_weights(xyz, uvw, d=3):
#     tri = qhull.Delaunay(xyz)
#     simplex = tri.find_simplex(uvw)
#     vertices = np.take(tri.simplices, simplex, axis=0)
#     temp = np.take(tri.transform, simplex, axis=0)
#     delta = uvw - temp[:, d]
#     bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
#     return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
#
#
# def interpolate(values, vtx, wts, fill_value=np.nan):
#     ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
#     ret[np.any(wts < 0, axis=1)] = fill_value
#     return ret
#
# folder_path = '/home/s1825216/data/'
folder_path = '/home/julius/Desktop/SCD/SeanData/'
snapshot = 3136
filename = 'turbsph_hdf5_plt_cnt_{}'.format(snapshot)  # only plt file, will automatically find part file

file_path = folder_path + filename
ds = yt.load(file_path)  # loads in data into data set class. This is what we will use to plot field values
ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary
#
t_0 = default_timer()
# Plot scatter with mayavi
_x = np.linspace(np.min(ad['x'].in_units('pc').to_value()), np.max(ad['x'].in_units('pc').to_value()), 100)
_y = np.linspace(np.min(ad['y'].in_units('pc').to_value()), np.max(ad['y'].in_units('pc').to_value()), 100)
_z = np.linspace(np.min(ad['z'].in_units('pc').to_value()), np.max(ad['z'].in_units('pc').to_value()), 100)
x, y, z = np.meshgrid(_x, _y, _z, indexing='ij')

idx_list = np.arange(len(ad['x']))
print(len(idx_list))
subset = np.random.choice(idx_list, 1000, replace=False)
print(len(subset))
print(len(list(set(subset))))
# print('make uvw...')
# axis = np.linspace(np.min(ad['x'].to_value()), np.max(ad['z'].to_value()), 100)
# lists = [axis]*3
# uvw = np.array([p for p in product(*lists)])
# print('make xyz...')
# xyz = np.column_stack((ad['x'].to_value(), ad['y'].to_value(), ad['z'].to_value()))
# # uvw = (x, y, z)
#
# print('interp_weights...')
# vtx, wts = interp_weights(xyz, uvw)
#
# print('interp...')
# interp = interpolate(ad['density'].to_value(), vtx, wts)

# print(ad['x'].in_units('pc').to_value())

interp = interpolate.griddata((ad['x'][subset].in_units('pc').to_value(),
                               ad['y'][subset].in_units('pc').to_value(),
                               ad['z'][subset].in_units('pc').to_value()),
                              ad['density'][subset].to_value(),
                              (x, y, z),
                              fill_value=0.0)
#
# print('save...')
# np.save('interpolated_snapshot.npy', interp)
#
# print('done, time: {}'.format(default_timer() - t_0))
# print(interp)
# print(interp.shape)
# interp = np.load('interpolated_snapshot.npy')
print(interp.shape)
print(np.max(interp))
print(np.min(interp))
figure = mlab.figure('DensityPlot')
source = mlab.pipeline.scalar_field(x,y,z,interp)

vol = mlab.pipeline.volume(source, vmin=0, vmax=0.04 * np.max(interp))

points = mlab.points3d(ad['x'][subset].in_units('pc').to_value(),
                       ad['y'][subset].in_units('pc').to_value(),
                       ad['z'][subset].in_units('pc').to_value()
                       , mode='point')  # , ad['density'][subset].to_value())
upper_lim = np.max(ad['x'].in_units('pc').to_value())
lower_lim = np.min(ad['x'].in_units('pc').to_value())

mlab.axes(ranges=[lower_lim, upper_lim, lower_lim, upper_lim, lower_lim, upper_lim])
mlab.show()
