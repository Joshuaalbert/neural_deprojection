import yt
import numpy as np
from tqdm import tqdm
from timeit import default_timer
from random import gauss

yt.funcs.mylog.setLevel(40)  # Surpresses YT status output.

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def make_rand_vector(dims):
    """
    Make unit vector in random direction
    Args:
        dims: dimension

    Returns: random unit vector

    """
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


folder_path = '~/Desktop/SCD/SeanData/'
# folder_path = '~/data/SeanData/M3f2/'
snapshot = 3136
filename = 'turbsph_hdf5_plt_cnt_{}'.format(snapshot)  # only plt file, will automatically find part file

file_path = folder_path + filename
ds = yt.load(file_path)  # loads in data into data set class. This is what we will use to plot field values
ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary
# containing all data available to be parsed through.
# e.g. print ad['mass'] will print the list of all cell masses.
# if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]

max_cell_ind = np.max(ad['grid_indices']).to_value()

print('find feature_array_length...')
for cell_ind in tqdm(range(0, 1+max_cell_ind)):
    cell_region = ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))
    if len(cell_region['grid_indices']) != 0:
        feature_array_length = round(len(cell_region['x'])**(1/3.))
        break
print('done')

num_of_projections = 30
resolution = 512
field ='density'
width = np.max(ad['x'].to_value()) - np.min(ad['x'].to_value())

property_values = []
property_transforms = [lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x,
                       np.log10, np.log10, np.log10, np.log10, lambda x: x]
property_names = ['x', 'y', 'z', 'velocity_x', 'velocity_y', 'velocity_z', 'density',
                  'temperature', 'cell_mass', 'cell_volume', 'gravitational_potential']

for cell_ind in tqdm(range(0, 1+max_cell_ind)):
    cell_region = ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))
    if len(cell_region['grid_indices']) != 0:
        _values = []
        for name, transform in zip(property_names, property_transforms):
            _values.append(transform(cell_region[name].to_value().reshape((feature_array_length, feature_array_length,
                                                                           feature_array_length))))
        property_values.append(np.stack(_values, axis=-1))      # list 16 16 16 f

property_values = np.stack(property_values, axis=0)     # n 16 16 16 f

projections = 0
while projections < num_of_projections:
    viewing_vec = make_rand_vector(3)
    rot_mat = rotation_matrix_from_vectors([1,0,0], viewing_vec)

    image = yt.off_axis_projection(ds, center=[0, 0, 0], normal_vector=viewing_vec, item=field, width=width,
                                   resolution=resolution)
    xyz = property_values[:, :, :, :, :3]
    velocity_xyz = property_values[:, :, :, :, 3:6]
    xyz = np.einsum('ap,ijklp->ijkla', rot_mat, xyz)
    velocity_xyz = np.einsum('ap,ijklp->ijkla', rot_mat, velocity_xyz)

    positions = np.mean(xyz, axis=(1, 2, 3))

    projections += 1