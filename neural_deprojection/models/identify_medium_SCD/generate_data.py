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


def std(tensor, axis):
    return tf.math.sqrt(tf.reduce_mean(tensor ** 2, axis=axis))


def find_screen_length(distance_matrix, k_mean):
    """
    Get optimal screening length.

    Args:
        distance_matrix: [num_points, num_points]
        k_mean: float

    Returns: float the optimal screen length
    """
    dist_max = distance_matrix.max()

    distance_matrix_no_loops = np.where(distance_matrix == 0., np.inf, distance_matrix)

    def get_k_mean(length):
        paired = distance_matrix_no_loops < length
        degree = np.sum(paired, axis=-1)
        return degree.mean()

    def loss(length):
        return get_k_mean(length) - k_mean

    if loss(0.) * loss(dist_max) >= 0.:
        # When there are fewer than k_mean+1 nodes in the list,
        # it's impossible for the average degree to be equal to k_mean.
        # So choose max screening length. Happens when f(low) and f(high) have same sign.
        return dist_max
    return bisect(loss, 0., dist_max, xtol=0.001)


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


def generate_example_nn(positions, properties, k=26, resolution=2, plot=False):
    print('example nn')

    resolution = 3.086e18 * resolution  # pc to cm

    node_features = []
    node_positions = []

    box_size = (np.max(positions), np.min(positions))  # box that encompasses all of the nodes
    axis = np.arange(box_size[1] + resolution, box_size[0], resolution)
    lists = [axis] * 3
    virtual_node_pos = [p for p in product(*lists)]
    virtual_kdtree = cKDTree(virtual_node_pos)
    particle_kdtree = cKDTree(positions)
    indices = virtual_kdtree.query_ball_tree(particle_kdtree, np.sqrt(3) / 2. * resolution)

    for i, p in enumerate(indices):
        if len(p) == 0:
            continue
        virt_pos, virt_prop = make_virtual_node(properties[p])
        node_positions.append(virt_pos)
        node_features.append(virt_prop)

    node_features = np.array(node_features)
    node_positions = np.array(node_positions)

    graph = nx.DiGraph()

    kdtree = cKDTree(node_positions)
    dist, idx = kdtree.query(node_positions, k=k + 1)
    receivers = idx[:, 1:]  # N,k
    senders = np.arange(node_positions.shape[0])  # N
    senders = np.tile(senders[:, None], [1, k])  # N,k
    receivers = receivers.flatten()
    senders = senders.flatten()

    n_nodes = node_positions.shape[0]

    pos = dict()  # for plotting node positions.
    edgelist = []

    for node, feature, position in zip(np.arange(n_nodes), node_features, node_positions):
        graph.add_node(node, features=feature)
        pos[node] = (position[:2] - box_size[1]) / (box_size[0] - box_size[1])

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
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        draw(graph, ax=ax, pos=pos, node_color='green', edge_color='red')

        image_dir = '/data2/hendrix/images/'
        graph_image_idx = len(glob.glob(os.path.join(image_dir, 'graph_image_*')))
        plt.savefig(os.path.join(image_dir, 'graph_image_{}'.format(graph_image_idx)))

    return networkxs_to_graphs_tuple([graph],
                                     node_shape_hint=[node_positions.shape[1] + node_features.shape[1]],
                                     edge_shape_hint=[2])


def generate_example(positions, properties, k_mean=26, plot=False):
    """
    Generate a geometric graph from positions.

    Args:
        positions: [num_points, 3] positions used for graph construction.
        properties: [num_points, F0,...,Fd] each node will have these properties of shape [F0,...,Fd]
        k_mean: float
        plot: whether to plot graph.

    Returns: GraphTuple
    """
    graph = nx.DiGraph()
    sibling_edgelist = []
    parent_edgelist = []
    pos = dict()  # for plotting node positions.
    real_nodes = list(np.arange(positions.shape[0]))
    while positions.shape[0] > 1:
        # n_nodes, n_nodes
        dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        opt_screen_length = find_screen_length(dist, k_mean)
        print("Found optimal screening length {}".format(opt_screen_length))

        distance_matrix_no_loops = np.where(dist == 0., np.inf, dist)
        A = distance_matrix_no_loops < opt_screen_length

        senders, receivers = np.where(A)
        n_edge = senders.size

        # num_points, F0,...Fd
        # if positions is to be part of features then this should already be set in properties.
        # We don't concatentate here. Mainly because properties could be an image, etc.
        sibling_nodes = properties
        n_nodes = sibling_nodes.shape[0]

        sibling_node_offset = len(graph.nodes)
        for node, feature, position in zip(np.arange(sibling_node_offset, sibling_node_offset + n_nodes), sibling_nodes,
                                           positions):
            graph.add_node(node, features=feature)
            pos[node] = position[:2]

        # edges = np.stack([senders, receivers], axis=-1) + sibling_node_offset
        for u, v in zip(senders + sibling_node_offset, receivers + sibling_node_offset):
            graph.add_edge(u, v, features=np.array([1., 0.]))
            graph.add_edge(v, u, features=np.array([1., 0.]))
            sibling_edgelist.append((u, v))
            sibling_edgelist.append((v, u))

        # for virtual nodes
        sibling_graph = GraphsTuple(nodes=None,  # sibling_nodes,
                                    edges=None,
                                    senders=senders,
                                    receivers=receivers,
                                    globals=None,
                                    n_node=np.array([n_nodes]),
                                    n_edge=np.array([n_edge]))

        sibling_graph = graphs_tuple_to_networkxs(sibling_graph)[0]
        # completely connect
        connected_components = sorted(nx.connected_components(nx.Graph(sibling_graph)), key=len)
        _positions = []
        _properties = []
        for connected_component in connected_components:
            print("Found connected component {}".format(connected_component))
            indices = list(sorted(list(connected_component)))
            virtual_position, virtual_property = make_virtual_node(positions[indices, :], properties[indices, ...])
            _positions.append(virtual_position)
            _properties.append(virtual_property)

        virtual_positions = np.stack(_positions, axis=0)
        virtual_properties = np.stack(_properties, axis=0)

        ###
        # add virutal nodes
        # num_parents, 3+F
        parent_nodes = virtual_properties
        n_nodes = parent_nodes.shape[0]
        parent_node_offset = len(graph.nodes)
        parent_indices = np.arange(parent_node_offset, parent_node_offset + n_nodes)
        # adding the nodes to global graph
        for node, feature, virtual_position in zip(parent_indices, parent_nodes, virtual_positions):
            graph.add_node(node, features=feature)
            print("new virtual {}".format(node))
            pos[node] = virtual_position[:2]

        for parent_idx, connected_component in zip(parent_indices, connected_components):

            child_node_indices = [idx + sibling_node_offset for idx in list(sorted(list(connected_component)))]
            for child_node_idx in child_node_indices:
                graph.add_edge(parent_idx, child_node_idx, features=np.array([0., 1.]))
                graph.add_edge(child_node_idx, parent_idx, features=np.array([0., 1.]))
                parent_edgelist.append((parent_idx, child_node_idx))
                parent_edgelist.append((child_node_idx, parent_idx))
                print("connecting {}<->{}".format(parent_idx, child_node_idx))

        positions = virtual_positions
        properties = virtual_properties

    # plotting

    virutal_nodes = list(set(graph.nodes) - set(real_nodes))
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        draw(graph, ax=ax, pos=pos, node_color='green', edgelist=[], nodelist=real_nodes)
        draw(graph, ax=ax, pos=pos, node_color='purple', edgelist=[], nodelist=virutal_nodes)
        draw(graph, ax=ax, pos=pos, edge_color='blue', edgelist=sibling_edgelist, nodelist=[])
        draw(graph, ax=ax, pos=pos, edge_color='red', edgelist=parent_edgelist, nodelist=[])
        plt.show()

    return networkxs_to_graphs_tuple([graph],
                                     node_shape_hint=[positions.shape[1] + properties.shape[1]],
                                     edge_shape_hint=[2])


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


def feature_to_graph_tuple(name=''):
    schema = {}
    schema[f'{name}_nodes'] = tf.io.FixedLenFeature([], dtype=tf.string)
    schema[f'{name}_senders'] = tf.io.FixedLenFeature([], dtype=tf.string)
    schema[f'{name}_receivers'] = tf.io.FixedLenFeature([], dtype=tf.string)
    return schema


def decode_examples(record_bytes, node_shape=None, image_shape=None, k=None):
    """
    Decodes raw bytes as returned from tf.data.TFRecordDataset([example_path]) into a GraphTuple and image
    Args:
        record_bytes: raw bytes
        node_shape: shape of nodes if known.
        edge_shape: shape of edges if known.
        image_shape: shape of image if known.

    Returns: (GraphTuple, image)
    """
    parsed_example = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        dict(
            idx=tf.io.FixedLenFeature([], dtype=tf.string),
            image=tf.io.FixedLenFeature([], dtype=tf.string),
            virtual_properties=tf.io.FixedLenFeature([], dtype=tf.string),
            snapshot=tf.io.FixedLenFeature([], dtype=tf.string),
            projection=tf.io.FixedLenFeature([], dtype=tf.string),
            extra_info=tf.io.FixedLenFeature([], dtype=tf.string)
            # **feature_to_graph_tuple('graph')
        )
    )
    idx =  tf.io.parse_tensor(parsed_example['idx'], tf.int32)
    idx.set_shape([None] + [k + 1])
    graph_nodes = tf.io.parse_tensor(parsed_example['virtual_properties'], tf.float32)
    graph_nodes.set_shape([None] + list(node_shape))
    image = tf.io.parse_tensor(parsed_example['image'], tf.float32)
    image.set_shape(image_shape)
    snapshot = tf.io.parse_tensor(parsed_example['snapshot'], tf.int32)
    snapshot.set_shape(())
    projection = tf.io.parse_tensor(parsed_example['projection'], tf.int32)
    projection.set_shape(())
    extra_info = tf.io.parse_tensor(parsed_example['extra_info'], tf.float32)
    extra_info.set_shape([None])

    receivers = idx[:, 1:]  # N,k
    senders = tf.cast(tf.range(tf.shape(graph_nodes)[0:1][0]), idx.dtype) # N
    senders = tf.tile(senders[:, None], tf.constant([1, k], tf.int32))  # N, k

    receivers = tf.reshape(receivers, shape=[-1])
    senders = tf.reshape(senders, shape=[-1])


    receivers_both_directions = tf.concat([receivers, senders], axis=0)
    senders_both_directions = tf.concat([senders, receivers], axis=0)

    n_node = tf.shape(graph_nodes)[0:1]
    n_edge = tf.shape(senders_both_directions)[0:1]

    graph_data_dict = dict(nodes=graph_nodes,
                           edges=tf.zeros((n_edge[0], 1)),
                           globals=tf.zeros([1]),
                           receivers=receivers_both_directions,
                           senders=senders_both_directions,
                           n_node=n_node,
                           n_edge=n_edge)

    return (graph_data_dict, image, snapshot, projection, extra_info)


def get_data_info(data_dirs):
    """
    Get information of saved data.

    Args:
        data_dirs: data directories

    Returns:

    """

    def data_generator():
        for idx, dir in tqdm(enumerate(data_dirs)):
            print("Generating data from {}".format(dir))
            positions, properties, image = _get_data(dir)
            yield (properties, image, dir)

    data_iterable = iter(data_generator())

    open('data_info.txt', 'w').close()

    while True:
        try:
            (properties, image, dir) = next(data_iterable)
        except StopIteration:
            break

        with open("data_info.txt", "a") as text_file:
            print(f"dir: {dir}\n"
                  f"    image_min: {np.min(image)}\n"
                  f"    image_max: {np.max(image)}\n"
                  f"    properties_min: {np.around(np.min(properties, axis=0), 2)}\n"
                  f"    properties_max: {np.around(np.max(properties, axis=0), 2)}\n", file=text_file)


def get_data_image(data_dirs):
    """
    Get information of saved data.

    Args:
        data_dirs: data directories

    Returns:

    """

    image_dir = '/data2/hendrix/projection_images/'

    def data_generator():
        for idx, dir in tqdm(enumerate(data_dirs)):
            print("Generating data from {}".format(dir))
            positions, properties, image = _get_data(dir)
            yield (properties, image, dir)

    data_iterable = iter(data_generator())

    while True:
        try:
            (properties, image, dir) = next(data_iterable)
        except StopIteration:
            break
        print('save image...')
        proj_image_idx = len(glob.glob(os.path.join(image_dir, 'proj_image_*')))
        plt.imsave(os.path.join(image_dir, 'proj_image_{}.png'.format(proj_image_idx)),
                   image[:, :, 0])
        print('saved.')


def generate_data(data_dir, save_dir='/data2/hendrix/train_data_2/'):
    """
    Routine for generating train data in tfrecords

    Args:
        data_dirs: where simulation data is.
        save_dir: where tfrecords will go.

    Returns: list of tfrecords.
    """
    npz_files = glob.glob(os.path.join(data_dir, '*'))

    def data_generator():
        print("Making graphs.")

        for idx, dir in tqdm(enumerate(npz_files)):
            print("Generating data from {}/{}".format(data_dir, dir))
            positions, properties, image = _get_data(dir)
            graph = generate_example_random_choice(positions, properties)
            yield (graph, image, idx)

    train_tfrecords = save_examples(data_generator(),
                                    save_dir=save_dir,
                                    examples_per_file=len(npz_files),
                                    num_examples=len(example_dirs),
                                    prefix='train')
    return train_tfrecords


###
# specific to project


def make_virtual_node(properties):
    """
    Aggregate positions and properties of nodes into one virtual node.

    Args:
        positions: [N, 3]
        properties: [N, F0,...Fd]

    Returns: [3], [F0,...,Fd]
    """

    virtual_properties = np.zeros(11)
    virtual_properties[:6] = np.mean(properties[:, 6], axis=0)
    virtual_properties[6] = np.sum(properties[:, 6])
    virtual_properties[7:9] = np.mean(properties[:, 7:9], axis=0)
    virtual_properties[9:11] = np.sum(properties[:, 9:11], axis=0)

    return np.mean(properties[:, 3], axis=0), virtual_properties


def aggregate_lowest_level_cells(positions, properties):
    '''
    aggregate the lowest level particles.

    Args:
        positions: node positions  [n, 3]
        properties: node properties   [n, f]

    Returns:
        agg_positions: aggregated node positions [m, 3]
        agg_properties: aggregated node properties  [m, f]
    '''

    lowest_level = np.max(properties[:, 11])
    lowest_level_positions = positions[properties[:, 11] == lowest_level]  # [j, 3]
    lowest_level_properties = properties[properties[:, 11] == lowest_level]  # [j, f]
    cell_inds = list(set(lowest_level_properties[:, 12]))  # [m-(n-j)]
    grouped_ll_positions = [lowest_level_positions[lowest_level_properties[:, 12] == ind] for ind in
                            cell_inds]  # [m-(n-j), 4096, 3]
    grouped_ll_properties = [lowest_level_properties[lowest_level_properties[:, 12] == ind] for ind in
                             cell_inds]  # [m-(n-j), 4096, f]

    agg_positions = positions[properties[:, 11] < lowest_level]  # [n-j, 3]
    agg_properties = properties[properties[:, 11] < lowest_level]  # [n-j, f]

    agg_positions = np.concatenate((agg_positions, np.mean(grouped_ll_positions, axis=0)))  # [m, 3]
    agg_properties = np.concatenate((agg_properties, np.mean(grouped_ll_properties, axis=0)))  # [m, f]

    return agg_positions, agg_properties


def _get_data(dir):
    """
    Should return the information for a single simulation.

    Args:
        dir: directory with sim data.

    Returns:
        positions for building graph
        properties for putting in nodes and aggregating upwards
        image corresponding to the graph
        extra info corresponding to the example

    """

    f = np.load(dir)
    positions = f['positions']
    properties = f['properties']
    image = f['proj_image']
    image = image.reshape((256, 256, 1))

    # properties = properties / np.std(properties, axis=0)        # normalize values

    # extra_info = f['extra_info']

    return positions, properties, image  # , extra_info


def make_tutorial_data(examples_dir):
    for i in range(10):
        example_idx = len(glob.glob(os.path.join(examples_dir, 'example_*')))
        data_dir = os.path.join(examples_dir, 'example_{:04d}'.format(example_idx))
        os.makedirs(data_dir, exist_ok=True)
        positions = np.random.uniform(0., 1., size=(50, 3))
        properties = np.random.uniform(0., 1., size=(50, 5))
        image = np.random.uniform(size=(24, 24, 1))
        np.savez(os.path.join(data_dir, 'data.npz'), positions=positions, properties=properties, image=image)


if __name__ == '__main__':
    examples_dir = '/data2/hendrix/examples/'
    train_data_dir = '/data2/hendrix/train_data_2/'

    example_dirs = glob.glob(os.path.join(examples_dir, 'example_*'))

    print(example_dirs)

    # get_data_info(example_dirs)
    # get_data_image(example_dirs)

    # list_of_example_dirs = []
    # temp_lst = []
    # for example_dir in example_dirs:
    #     if len(temp_lst) == 32:
    #         list_of_example_dirs.append(temp_lst)
    #         temp_lst = []
    #     else:
    #         temp_lst.append(example_dir)
    # list_of_example_dirs.append(temp_lst)

    # print(f'number of tfrecfiles: {len(list_of_example_dirs)}')

    pool = Pool(1)
    pool.map(generate_data, example_dirs)
