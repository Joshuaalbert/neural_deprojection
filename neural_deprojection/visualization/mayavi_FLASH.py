from mayavi import mlab

import yt
import numpy as np
from timeit import default_timer
# from scipy import interpolate

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
# import itertools

# quicker way of interpolating. found here:
# https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

def interp_weights(xyz, uvw, d=3):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

folder_path = '/home/s1825216/data/'
# folder_path = '/home/julius/Desktop/SCD/SeanData/'
snapshot = 3136
filename = 'turbsph_hdf5_plt_cnt_{}'.format(snapshot)  # only plt file, will automatically find part file

file_path = folder_path + filename
ds = yt.load(file_path)  # loads in data into data set class. This is what we will use to plot field values
ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary

t_0 = default_timer()
# Plot scatter with mayavi
_x = np.linspace(np.min(ad['x'].to_value()), np.max(ad['x'].to_value()), 200)
_y = np.linspace(np.min(ad['y'].to_value()), np.max(ad['y'].to_value()), 200)
_z = np.linspace(np.min(ad['z'].to_value()), np.max(ad['z'].to_value()), 200)
x, y, z = np.meshgrid(_x, _y, _z, indexing='ij')


vtx, wts = interp_weights((ad['x'].to_value(), ad['y'].to_value(), ad['z'].to_value()),
                          (x, y, z))

interp = interpolate(ad['density'].to_value(), vtx, wts)

# interp = interpolate.griddata((ad['x'].to_value(), ad['y'].to_value(), ad['z'].to_value()), ad['density'].to_value(),
#                               (x, y, z))                  # this function is waaaay too slow

np.save('interpolated_snapshot.npy', interp)

print('done, time: {}'.format(default_timer() - t_0))

# interp = np.load('interpolated_snapshot.npy')
# print(interp.shape)
# print(np.max(interp))
# print(np.min(interp))
# figure = mlab.figure('DensityPlot')
# source = mlab.pipeline.scalar_field(interp)
#
# vol = mlab.pipeline.volume(source)
# mlab.show()
