import yt
import numpy as np
from tqdm import tqdm
from timeit import default_timer
from random import gauss
import os
import glob
from multiprocessing import Pool

yt.funcs.mylog.setLevel(40)  # Suppresses YT status output.

# folder_path = '~/Desktop/SCD/SeanData/'
folder_path = '/disks/extern_collab_data/lewis/run3/M3f2/'
examples_dir = '/data2/hendrix/examples/'
# examples_dir = '/home/julius/Desktop/SCD/SeanData/examples/'
# snapshot = 3136
snapshot_list = np.arange(3100, 3137)[::-1]
# snapshot_list = [3136]
# folder_path = '/disks/extern_collab_data/lewis/run3/'
# examples_dir = '/home/hendrix/data/examples/'

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
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]


def process_snapshot_individual_nodes(snapshot):
    print('loading particle and plt file...')
    t_0 = default_timer()
    filename = 'turbsph_hdf5_plt_cnt_{}'.format(snapshot)  # only plt file, will automatically find part file

    file_path = folder_path + filename
    ds = yt.load(file_path)  # loads in data into data set class. This is what we will use to plot field values
    ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary
    # containing all data available to be parsed through.
    # e.g. print ad['mass'] will print the list of all cell masses.
    # if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]
    print('done, time: {}'.format(default_timer() - t_0))

    max_cell_ind = int(np.max(ad['grid_indices']).to_value())
    num_of_projections = 30
    resolution_in_pc = 0.1
    field = 'density'
    width = np.max(ad['x'].to_value()) - np.min(ad['x'].to_value())
    resolution = int(round(width * 3.24078e-19 / resolution_in_pc))

    print('making property_values...')
    t_0 = default_timer()

    property_values = []
    property_transforms = [lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x,
                           np.log10, np.log10, np.log10, np.log10, lambda x: x, lambda x: x, lambda x: x]
    property_names = ['x', 'y', 'z', 'velocity_x', 'velocity_y', 'velocity_z', 'density', 'temperature',
                      'cell_mass', 'cell_volume', 'gravitational_potential', 'grid_level', 'grid_indices']

    for cell_ind in tqdm(range(0, 1 + max_cell_ind)):
        cell_region = ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))
        if len(cell_region['grid_indices']) != 0:
            _values = []
            for name, transform in zip(property_names, property_transforms):
                _values.append(transform(cell_region[name].to_value()))
            property_values.extend(np.stack(_values, axis=-1))

    property_values = np.array(property_values)     # n f
    print('done, time: {}'.format(default_timer() - t_0))

    print('making projections and rotating coordinates')
    t_0 = default_timer()

    # positions = []
    # properties = []
    # proj_images = []
    # extra_info = []

    projections = 0
    while projections < num_of_projections:
        print(projections)
        viewing_vec = make_rand_vector(3)
        rot_mat = rotation_matrix_from_vectors([1, 0, 0], viewing_vec)

        _extra_info = [snapshot, viewing_vec, resolution, width, field]
        proj_image = yt.off_axis_projection(ds, center=[0, 0, 0], normal_vector=viewing_vec, item=field, width=width,
                                            resolution=resolution)
        xyz = property_values[:, :3]    # n 3
        velocity_xyz = property_values[:, 3:6]    # n 3
        xyz = np.einsum('ap,np->na', rot_mat, xyz)      # n 3
        velocity_xyz = np.einsum('ap,np->na', rot_mat, velocity_xyz)        # n 3

        _properties = property_values.copy()        # n f
        _properties[:, :3] = xyz        # n f
        _properties[:, 3:6] = velocity_xyz      # n f
        _positions = xyz        # n 3

        example_idx = len(glob.glob(os.path.join(examples_dir, 'example_*')))
        path_to_example_folder = os.path.join(examples_dir, "example_{:04d}".format(example_idx))
        os.makedirs(path_to_example_folder, exist_ok=True)
        np.savez(os.path.join(path_to_example_folder, "data.npz"), positions=_positions, properties=_properties,
                 proj_image=proj_image, extra_info=_extra_info)

        projections += 1

    print('done, time: {}'.format(default_timer() - t_0))

    # print('saving data...')
    # t_0 = default_timer()
    # # examples_dir is where all your examples will go.
    # for (_positions, _properties, proj_image, _extra_info) in zip(positions, properties, proj_images, extra_info):
    #     # _positions is (num_nodes, 3)
    #     # _properties in (num_nodes,) + per_node_property_shape
    #     # proj_image is (width, height, channels)
    #     # _extra_image is any array of extra information like viewing angle, etc that might be useful later.
    #     example_idx = len(glob.glob(os.path.join(examples_dir, 'example_*')))
    #     os.makedirs(os.path.join(examples_dir, "example_{:04d}".format(example_idx)), exist_ok=True)
    #     np.savez("data.npz", positions=_positions, properties=_properties, proj_image=proj_image, extra_info=_extra_info)
    #
    # print('done, time: {}'.format(default_timer() - t_0))

def process_snapshot_feature_array(snapshot):
    print('loading particle and plt file...')
    t_0 = default_timer()
    filename = 'turbsph_hdf5_plt_cnt_{}'.format(snapshot)  # only plt file, will automatically find part file

    file_path = folder_path + filename
    ds = yt.load(file_path)  # loads in data into data set class. This is what we will use to plot field values
    ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary
    # containing all data available to be parsed through.
    # e.g. print ad['mass'] will print the list of all cell masses.
    # if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]

    print('done, time: {}'.format(default_timer() - t_0))

    print('find feature_array_length...')
    t_0 = default_timer()

    max_cell_ind = int(np.max(ad['grid_indices']).to_value())

    for cell_ind in tqdm(range(0, 1 + max_cell_ind)):
        cell_region = ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))
        if len(cell_region['grid_indices']) != 0:
            feature_array_length = round(len(cell_region['x']) ** (1 / 3.))
            break
    print('done, time: {}'.format(default_timer() - t_0))

    num_of_projections = 30
    resolution_in_pc = 0.1
    field = 'density'
    width = np.max(ad['x'].to_value()) - np.min(ad['x'].to_value())
    resolution = int(round(width * 3.24078e-19 / resolution_in_pc))

    print('making property_values...')
    t_0 = default_timer()

    property_values = []
    property_transforms = [lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x,
                           np.log10, np.log10, np.log10, np.log10, lambda x: x]
    property_names = ['x', 'y', 'z', 'velocity_x', 'velocity_y', 'velocity_z', 'density',
                      'temperature', 'cell_mass', 'cell_volume', 'gravitational_potential']

    for cell_ind in tqdm(range(0, 1 + max_cell_ind)):
        cell_region = ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))
        if len(cell_region['grid_indices']) != 0:
            _values = []
            for name, transform in zip(property_names, property_transforms):
                _values.append(transform(cell_region[name].to_value().reshape((feature_array_length, feature_array_length,
                                                                               feature_array_length))))
            property_values.append(np.stack(_values, axis=-1))  # list 16 16 16 f

    property_values = np.stack(property_values, axis=0)  # n 16 16 16 f

    print('done, time: {}'.format(default_timer() - t_0))

    print('making projections and rotating coordinates')
    t_0 = default_timer()

    # positions = []
    # properties = []
    # proj_images = []
    # extra_info = []

    projections = 0
    while projections < num_of_projections:
        print(projections)
        viewing_vec = make_rand_vector(3)
        rot_mat = rotation_matrix_from_vectors([1, 0, 0], viewing_vec)

        _extra_info = [snapshot, viewing_vec, resolution, width, field]
        proj_image = yt.off_axis_projection(ds, center=[0, 0, 0], normal_vector=viewing_vec, item=field, width=width,
                                            resolution=resolution)
        xyz = property_values[:, :, :, :, :3]
        velocity_xyz = property_values[:, :, :, :, 3:6]
        xyz = np.einsum('ap,ijklp->ijkla', rot_mat, xyz)
        velocity_xyz = np.einsum('ap,ijklp->ijkla', rot_mat, velocity_xyz)

        _properties = property_values.copy()
        _properties[:, :, :, :, :3] = xyz
        _properties[:, :, :, :, 3:6] = velocity_xyz
        _positions = np.mean(xyz, axis=(1, 2, 3))

        example_idx = len(glob.glob(os.path.join(examples_dir, 'example_*')))
        path_to_example_folder = os.path.join(examples_dir, "example_{:04d}".format(example_idx))
        os.makedirs(path_to_example_folder, exist_ok=True)
        np.savez(os.path.join(path_to_example_folder, "data.npz"), positions=_positions, properties=_properties,
                 proj_image=proj_image, extra_info=_extra_info)

        projections += 1

    print('done, time: {}'.format(default_timer() - t_0))

    # print('saving data...')
    # t_0 = default_timer()
    # # examples_dir is where all your examples will go.
    # for (_positions, _properties, proj_image, _extra_info) in zip(positions, properties, proj_images, extra_info):
    #     # _positions is (num_nodes, 3)
    #     # _properties in (num_nodes,) + per_node_property_shape
    #     # proj_image is (width, height, channels)
    #     # _extra_image is any array of extra information like viewing angle, etc that might be useful later.
    #     example_idx = len(glob.glob(os.path.join(examples_dir, 'example_*')))
    #     os.makedirs(os.path.join(examples_dir, "example_{:04d}".format(example_idx)), exist_ok=True)
    #     np.savez("data.npz", positions=_positions, properties=_properties, proj_image=proj_image, extra_info=_extra_info)
    #
    # print('done, time: {}'.format(default_timer() - t_0))


if __name__ == '__main__':
    # pool = Pool(os.cpu_count() - 2)
    pool = Pool(12)
    pool.map(process_snapshot_individual_nodes, snapshot_list)
    # process_snapshot_individual_nodes(3136)