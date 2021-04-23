# TEST TEST

import yt
from random import gauss
import os
import glob
import tensorflow as tf

from timeit import default_timer
from itertools import product
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs, networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from networkx.drawing import draw
from tqdm import tqdm
from scipy.optimize import bisect
from scipy.spatial.ckdtree import cKDTree
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multiprocessing import get_context, Pool, Lock, set_start_method

# mp_lock = Lock()

yt.funcs.mylog.setLevel(40)  # Suppresses YT status output.


def _random_ortho_matrix(n):
    """
    Samples a random orthonormal num_parent,num_parent matrix from Stiefels manifold.
    From https://stackoverflow.com/a/38430739

    Args:
        n: Size of matrix, draws from O(n) group.

    Returns: random [n,n] matrix with determinant = +-1
    """
    H = np.random.normal(size=(n, n))
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def _random_special_ortho_matrix(n):
    """
    Samples a random orthonormal num_parent,num_parent matrix from Stiefels manifold.
    From https://stackoverflow.com/a/38430739

    Args:
        key: PRNG seed
        n: Size of matrix, draws from O(n) group.

    Returns: random [n,n] matrix with determinant = +-1
    """
    det = -1.
    while det < 0:
        Q = _random_ortho_matrix(n)
        det = np.linalg.det(Q)
    return Q


def graph_tuple_to_feature(graph: GraphsTuple, name=''):
    return {
        f'{name}_nodes': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.nodes, tf.float32)).numpy()])),
        f'{name}_edges': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.edges, tf.float32)).numpy()])),
        f'{name}_senders': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.senders, tf.int64)).numpy()])),
        f'{name}_receivers': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.receivers, tf.int64)).numpy()]))}


def save_examples(generator, snapshot, save_dir=None,
                  examples_per_file=26, num_examples=1, prefix='train'):
    """
    Saves a list of GraphTuples to tfrecords.

    Args:
        generator: generator (or list) of (GraphTuples, image).
            Generator is more efficient.
        save_dir: dir to save tfrecords in
        examples_per_file: int, max number examples per file

    Returns: list of tfrecord files.
    """
    print("Saving data in tfrecords.")
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    files = []
    data_iterable = iter(generator)
    data_left = True
    # pbar = tqdm(total=num_examples)
    while data_left:

        # mp_lock.acquire()  # make sure no duplicate files are made / replaced
        # file_idx = len(glob.glob(os.path.join(save_dir, 'train_*')))
        # mp_lock.release()

        file = os.path.join(save_dir, 'train_{:04d}.tfrecords'.format(snapshot))
        files.append(file)
        with tf.io.TFRecordWriter(file) as writer:
            for i in range(examples_per_file + 1):  # + 1 otherwise it makes a second tf rec file that is empty
                try:
                    print('try data...')
                    (graph, image, snapshot, projection) = next(data_iterable)
                except StopIteration:
                    data_left = False
                    break
                graph = get_graph(graph, 0)
                features = dict(
                    image=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(image, tf.float32)).numpy()])),
                    snapshot=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(snapshot, tf.int32)).numpy()])),
                    projection=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(projection, tf.int32)).numpy()])),
                    **graph_tuple_to_feature(graph, name='graph')
                )
                features = tf.train.Features(feature=features)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                # pbar.update(1)
    print("Saved in tfrecords: {}".format(files))
    return files


def generate_example_random_choice(positions, properties, number_of_virtual_nodes, plot, k):
    print('choice nn')
    idx_list = np.arange(len(positions))
    virtual_node_positions = positions[np.random.choice(idx_list, number_of_virtual_nodes, replace=False)]

    kdtree = cKDTree(virtual_node_positions)
    dist, indices = kdtree.query(positions)

    virtual_properties = np.zeros((len(np.bincount(indices)), len(properties[0])))

    mean_sum = [lambda x: np.bincount(indices, weights=x) / np.maximum(1., np.bincount(indices)),  # mean
                lambda x: np.bincount(indices, weights=x)]  # sum

    mean_sum_enc = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]

    for p, enc in zip(np.arange(len(properties[0])), mean_sum_enc):
        virtual_properties[:, p] = mean_sum[enc](properties[:, p])
        virtual_positions = virtual_properties[:, :3]

    graph = nx.OrderedMultiDiGraph()
    kdtree = cKDTree(virtual_positions)
    dist, idx = kdtree.query(virtual_positions, k=k + 1)
    receivers = idx[:, 1:]  # N,k
    senders = np.arange(virtual_positions.shape[0])  # N
    senders = np.tile(senders[:, None], [1, k])  # N,k

    receivers = receivers.flatten()
    senders = senders.flatten()

    n_nodes = virtual_positions.shape[0]

    pos = dict()  # for plotting node positions.
    edgelist = []
    for node, feature, position in zip(np.arange(n_nodes), virtual_properties, virtual_positions):
        graph.add_node(node, features=feature)
        pos[node] = position[:2]

    # edges = np.stack([senders, receivers], axis=-1) + sibling_node_offset
    for u, v in zip(senders, receivers):
        graph.add_edge(u, v, features=np.array([1., 0.]))
        graph.add_edge(v, u, features=np.array([1., 0.]))
        edgelist.append((u, v))
        edgelist.append((v, u))

    graph.graph["features"] = np.array([0.])
    # plotting

    print('len(pos) = {}\nlen(edgelist) = {}'.format(len(pos), len(edgelist)))
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        draw(graph, ax=ax, pos=pos, node_color='blue', edge_color='red', node_size=10, width=0.1)

        image_dir = '/data2/hendrix/images/'
        graph_image_idx = len(glob.glob(os.path.join(image_dir, 'graph_image_*')))
        plt.savefig(os.path.join(image_dir, 'graph_image_{}'.format(graph_image_idx)))

    return networkxs_to_graphs_tuple([graph],
                                     node_shape_hint=[virtual_positions.shape[1] + virtual_properties.shape[1]],
                                     edge_shape_hint=[2])


def generate_data(positions, properties, proj_images, extra_info, number_of_virtual_nodes, plotting,
                  number_of_neighbours, save_dir='/data2/hendrix/train_data/',
                  image_shape=(256, 256, 1)):
    """
    Routine for generating train data in tfrecords

    Args:
        data_dirs: where simulation data is.
        save_dir: where tfrecords will go.

    Returns: list of tfrecords.
    """

    snapshot = extra_info[0][0]

    def data_generator():
        print("Making graphs.")
        for idx in range(len(positions)):
            print("Generating data from projection {}".format(idx))
            _positions = positions[idx]
            _properties = properties[idx]
            proj_image = proj_images[idx].reshape(image_shape)
            snapshot = extra_info[idx][0]
            projection = extra_info[idx][1]
            graph = generate_example_random_choice(_positions, _properties, number_of_virtual_nodes, plotting,
                                                   number_of_neighbours)
            print('\ngenerator output:\n', len(graph.edges), len(graph.nodes),
                  '\n', proj_image.shape, '\n', snapshot, '\n', projection)
            yield (graph, proj_image, snapshot, projection)

    train_tfrecords = save_examples(data_generator(),
                                    snapshot,
                                    save_dir=save_dir,
                                    examples_per_file=len(positions),
                                    num_examples=len(positions),
                                    prefix='train')
    return train_tfrecords


def snapshot_to_tfrec(snapshot_file, save_dir, num_of_projections, number_of_virtual_nodes, number_of_neighbours,
                      plotting):
    """
    load snapshot plt file, rotate for different viewing angles, make projections and corresponding graph nets. Save to
    tfrec files.

    Args:
        snapshot: snaphot
        save_dir: location of tfrecs

    Returns:

    """
    print('loading particle and plt file {}...'.format(snapshot_file))
    t_0 = default_timer()
    # filename = 'turbsph_hdf5_plt_cnt_{:04d}'.format(snapshot)  # only plt file, will automatically find part file
    #
    # file_path = folder_path + filename
    # print(f'file: {file_path}')
    ds = yt.load(snapshot_file)  # loads in data into data set class. This is what we will use to plot field values
    ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary
    # containing all data available to be parsed through.
    # e.g. print ad['mass'] will print the list of all cell masses.
    # if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]
    print('done, time: {}'.format(default_timer() - t_0))

    snapshot = int(snapshot_file[-4:])
    field = 'density'
    width = np.max(ad['x'].to_value()) - np.min(ad['x'].to_value())
    resolution = 256

    print('making property_values...')
    t_0 = default_timer()

    property_values = []
    property_transforms = [lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x, lambda x: x,
                           np.log10, np.log10, np.log10, np.log10]

    property_names = ['x', 'y', 'z', 'velocity_x', 'velocity_y', 'velocity_z', 'gravitational_potential',
                      'density', 'temperature', 'cell_mass', 'cell_volume']

    unit_names = ['pc', 'pc', 'pc', '10*km/s', '10*km/s', '10*km/s', 'J/mg', 'Msun/pc**3', 'K', 'Msun', 'pc**3']

    _values = []
    for name, transform, unit in zip(property_names, property_transforms, unit_names):
        print('\nproperty name: ', name)
        print('in unit', ad[name].in_units(unit).to_value())
        print('transform', transform(ad[name].in_units(unit).to_value()))
        _values.append(transform(ad[name].in_units(unit).to_value()))

    return 1

    # property_values = np.array(property_values)  # n f
    property_values = np.array(_values).T  # n f
    print('done, time: {}'.format(default_timer() - t_0))

    print('making projections and rotating coordinates')
    t_0 = default_timer()

    positions = []
    properties = []
    proj_images = []
    extra_info = []

    projections = 0
    while projections < num_of_projections:
        # print(projections)

        V = np.eye(3)
        R = _random_special_ortho_matrix(3)
        Vprime = np.linalg.inv(R) @ V

        north_vector = Vprime[:, 1]
        viewing_vec = Vprime[:, 2]

        _extra_info = [snapshot, projections, viewing_vec, resolution, width, field]
        proj_image = yt.off_axis_projection(ds, center=[0, 0, 0], normal_vector=viewing_vec, north_vector=north_vector,
                                            item=field, width=width, resolution=resolution)

        proj_image = np.log10(np.where(proj_image < 1e-5, 1e-5, proj_image))

        xyz = property_values[:, :3]  # n 3
        velocity_xyz = property_values[:, 3:6]  # n 3

        rotated_xyz = (R @ xyz.T).T
        rotated_velocity_xyz = (R @ velocity_xyz.T).T

        _properties = property_values.copy()  # n f
        _properties[:, :3] = rotated_xyz  # n f
        _properties[:, 3:6] = rotated_velocity_xyz  # n f
        _properties[:, 6] = _properties[:, 6]  # n f
        _positions = xyz  # n 3

        positions.append(_positions)
        properties.append(_properties)
        proj_images.append(proj_image)
        extra_info.append(_extra_info)

        projections += 1

    generate_data(positions=positions, properties=properties, proj_images=proj_images, extra_info=extra_info,
                  save_dir=save_dir, number_of_virtual_nodes=number_of_virtual_nodes, plotting=plotting,
                  number_of_neighbours=number_of_neighbours)

    print('done, time: {}'.format(default_timer() - t_0))


def main():
    # claude_name_list = ['M4r5b',
    #                     'M4r5b-3',
    #                     'M4r5b-5',
    #                     'M4r5s-2',
    #                     'M4r5s-4',
    #                     'M4r6b',
    #                     'M4r6b-3',
    #                     'M4r6s',
    #                     'M4r5b-2',
    #                     'M4r5b-4',
    #                     'M4r5s',
    #                     'M4r5s-3',
    #                     'M4r5s-5',
    #                     'M4r6b-2',
    #                     'M4r6b-4']
    #
    # for n in claude_name_list:
    #     folder_path = '/disks/extern_collab_data/cournoyer/{}/'.format(n)
    #     save_dir = '/data2/hendrix/ClaudeData/{}/'.format(n)

    folder_path = '/disks/extern_collab_data/lewis/run1/'       # run1=M3, run2=M3f, run3=M3f2, run4=M4
    save_dir = '/net/para33/data2/hendrix/SeanData/M3/'

    # snapshot_list = []
    # for snap in all_snapshots:
    #     file = os.path.join(folder_path,'turbsph_hdf5_plt_cnt_{:04d}'.format(snap))
    #     snapshot_list.append(file)

    snapshot_list = glob.glob(os.path.join(folder_path, 'turbsph_hdf5_plt_cnt_*'))
    print('len(snapshot_list): ', len(snapshot_list))

    # snapshot_list = np.array(snapshot_list)
    # snapshot_list = snapshot_list[[int(s[-4:]) % 3 == 0 for s in snapshot_list]]       # only keep every third snapshot

    print('len(snapshot_list): ', len(snapshot_list))
    print(snapshot_list)

    number_of_projections = 26
    number_of_virtual_nodes = 50000
    number_of_neighbours = 6
    plotting = False

    params = [(snapsh,
               save_dir,
               number_of_projections,
               number_of_virtual_nodes,
               number_of_neighbours,
               plotting) for snapsh in snapshot_list]

    # file = os.path.join(folder_path, 'turbsph_hdf5_plt_cnt_{:04d}'.format(265))
    #
    # snapshot_to_tfrec(file,
    #                   save_dir,
    #                   number_of_projections,
    #                   number_of_virtual_nodes,
    #                   number_of_neighbours,
    #                   plotting
    #                   )

    with get_context("spawn").Pool(processes=15) as pool:
        # pool = Pool(15)
        pool.starmap(snapshot_to_tfrec, params)
        pool.close()

    # for param in params:
    #     file, save_dir, number_of_projections, number_of_virtual_nodes, number_of_neighbours, plotting = param
    #     snapshot_to_tfrec(file, save_dir, number_of_projections, number_of_virtual_nodes, number_of_neighbours, plotting)


if __name__ == '__main__':
    set_start_method("spawn")

    main()
