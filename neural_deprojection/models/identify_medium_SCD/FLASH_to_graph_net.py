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

from multiprocessing import Pool, Lock

mp_lock = Lock()

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


def save_examples(generator, save_dir=None,
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
    pbar = tqdm(total=num_examples)
    while data_left:

        mp_lock.acquire()  # make sure no duplicate files are made / replaced
        tf_files = glob.glob(os.path.join(save_dir, 'train_*'))
        file_idx = len(tf_files)
        mp_lock.release()

        file = os.path.join(save_dir, 'train_{:04d}.tfrecords'.format(file_idx))
        files.append(file)
        with tf.io.TFRecordWriter(file) as writer:
            for i in range(examples_per_file):
                try:
                    (graph, image, example_idx) = next(data_iterable)
                except StopIteration:
                    data_left = False
                    break
                graph = get_graph(graph, 0)
                features = dict(
                    image=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(image, tf.float32)).numpy()])),
                    example_idx=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(example_idx, tf.int32)).numpy()])),
                    **graph_tuple_to_feature(graph, name='graph')
                )
                features = tf.train.Features(feature=features)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                pbar.update(1)
    print("Saved in tfrecords: {}".format(files))
    return files


def generate_example_random_choice(positions, properties, k=26, plot=False):
    print('choice nn')
    idx_list = np.arange(len(positions))
    virtual_node_positions = positions[np.random.choice(idx_list, 1000, replace=False)]

    kdtree = cKDTree(virtual_node_positions)
    dist, indices = kdtree.query(positions)

    virtual_properties = np.zeros((len(np.bincount(indices)), len(properties[0])))

    mean_sum = [lambda x: np.bincount(indices, weights=x) / np.maximum(1., np.bincount(indices)),  # mean
                lambda x: np.bincount(indices, weights=x)]  # sum

    mean_sum_enc = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]

    for p, enc in zip(np.arange(len(properties[0])), mean_sum_enc):
        virtual_properties[:, p] = mean_sum[enc](properties[:, p])
        virtual_positions = virtual_properties[:, :3]

    graph = nx.DiGraph()
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


def generate_data(positions, properties, proj_images, save_dir='/data2/hendrix/train_data_2/',
                  image_shape=(256, 256, 1)):
    """
    Routine for generating train data in tfrecords

    Args:
        data_dirs: where simulation data is.
        save_dir: where tfrecords will go.

    Returns: list of tfrecords.
    """

    def data_generator():
        print("Making graphs.")
        for idx in tqdm(range(len(positions))):
            print("Generating data from projection {}".format(idx))
            _positions = positions[idx]
            _properties = properties[idx]
            proj_image = proj_images[idx].reshape(image_shape)
            graph = generate_example_random_choice(_positions, _properties)
            yield (graph, proj_image, idx)

    train_tfrecords = save_examples(data_generator(),
                                    save_dir=save_dir,
                                    examples_per_file=len(positions),
                                    num_examples=len(positions),
                                    prefix='train')
    return train_tfrecords


def snaphot_to_tfrec(snapshot, save_dir='/data2/hendrix/train_data_2/'):
    """
    load snapshot plt file, rotate for different viewing angles, make projections and corresponding graph nets. Save to
    tfrec files.

    Args:
        snapshot: snaphot
        save_dir: location of tfrecs

    Returns:

    """
    print('loading particle and plt file...')
    t_0 = default_timer()
    filename = 'turbsph_hdf5_plt_cnt_{:04d}'.format(snapshot)  # only plt file, will automatically find part file

    file_path = folder_path + filename
    # print(f'file: {file_path}')
    ds = yt.load(file_path)  # loads in data into data set class. This is what we will use to plot field values
    ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary
    # containing all data available to be parsed through.
    # e.g. print ad['mass'] will print the list of all cell masses.
    # if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]
    print('done, time: {}'.format(default_timer() - t_0))

    max_cell_ind = int(np.max(ad['grid_indices']).to_value())
    num_of_projections = 5
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

    for cell_ind in tqdm(range(0, 1 + max_cell_ind)):
        cell_region = ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))
        if len(cell_region['grid_indices']) != 0:
            _values = []
            for name, transform, unit in zip(property_names, property_transforms, unit_names):
                _values.append(transform(cell_region[name].in_units(unit).to_value()))
            property_values.extend(np.stack(_values, axis=-1))

    property_values = np.array(property_values)  # n f
    print('done, time: {}'.format(default_timer() - t_0))

    print('making projections and rotating coordinates')
    t_0 = default_timer()

    positions = []
    properties = []
    proj_images = []
    extra_info = []

    projections = 0
    while projections < num_of_projections:
        print(projections)

        V = np.eye(3)
        rot_mat = _random_special_ortho_matrix(3)
        Vprime = rot_mat @ V

        north_vector = Vprime[:, 1]
        viewing_vec = Vprime[:, 2]

        _extra_info = [snapshot, viewing_vec, resolution, width, field]
        proj_image = yt.off_axis_projection(ds, center=[0, 0, 0], normal_vector=viewing_vec, north_vector=north_vector,
                                            item=field, width=width, resolution=resolution)

        proj_image = np.log10(np.where(proj_image < 1e-5, 1e-5, proj_image))

        xyz = property_values[:, :3]  # n 3
        velocity_xyz = property_values[:, 3:6]  # n 3
        xyz = np.einsum('ap,np->na', rot_mat, xyz)  # n 3
        velocity_xyz = np.einsum('ap,np->na', rot_mat, velocity_xyz)  # n 3

        _properties = property_values.copy()  # n f
        _properties[:, :3] = xyz  # n f
        _properties[:, 3:6] = velocity_xyz  # n f
        _properties[:, 6] = _properties[:, 6]  # n f
        _positions = xyz  # n 3

        positions.append(_positions)
        properties.append(_properties)
        proj_images.append(proj_image)
        extra_info.append(_extra_info)

        projections += 1

    generate_data(positions, properties, proj_images, save_dir)

    print('done, time: {}'.format(default_timer() - t_0))


if __name__ == '__main__':
    # folder_path = '~/Desktop/SCD/SeanData/'
    folder_path = '/disks/extern_collab_data/lewis/run3/'
    examples_dir = '/data2/hendrix/examples/'
    # examples_dir = '/home/julius/Desktop/SCD/SeanData/examples/'
    all_snapshots = np.arange(3137)
    # snapshot_list = np.random.choice(all_snapshots, 5, replace=False)
    snapshot_list = [3136, 3000, 2500, 2000, 1500]

    save_dir = '/data2/hendrix/train_data_2/'

    pool = Pool(5)
    pool.map(snaphot_to_tfrec, snapshot_list)
