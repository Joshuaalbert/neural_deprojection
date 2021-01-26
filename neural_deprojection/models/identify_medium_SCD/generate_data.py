import os
import glob
import tensorflow as tf

from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs, networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from networkx.drawing import draw
from tqdm import tqdm
from scipy.optimize import bisect
from scipy.spatial.ckdtree import cKDTree
import pylab as plt


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

def generate_example_nn(positions, properties, k=1, plot=False):
    """
    Generate a k-nn graph from positions.

    Args:
        positions: [num_points, 3] positions used for graph constrution.
        properties: [num_points, F0,...,Fd] each node will have these properties of shape [F0,...,Fd]
        k: int, k nearest neighbours are connected.
        plot: whether to plot graph.

    Returns: GraphTuple
    """
    graph = nx.DiGraph()

    kdtree = cKDTree(positions)
    dist, idx = kdtree.query(positions, k=k+1)
    receivers = idx[:, 1:]#N,k
    senders = np.arange(positions.shape[0])#N
    senders = np.tile(senders[:, None], [1,k])#N,k
    receivers = receivers.flatten()
    senders = senders.flatten()



    n_nodes = positions.shape[0]

    pos = dict()  # for plotting node positions.
    edgelist = []

    for node, feature, position in zip(np.arange(n_nodes), properties, positions):
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

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        draw(graph, ax=ax, pos=pos, node_color='green', edge_color='red')
        plt.show()

    return networkxs_to_graphs_tuple([graph],
                                     node_shape_hint=[positions.shape[1] + properties.shape[1]],
                                     edge_shape_hint=[2])


def generate_example(positions, properties, k_mean=26, plot=False):
    """
    Generate a geometric graph from positions.

    Args:
        positions: [num_points, 3] positions used for graph constrution.
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
        # [1,0] for siblings, [0,1] for parent-child
        sibling_edges = np.tile([[1., 0.]], [n_edge, 1])

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
                  examples_per_file=32, num_examples=1, prefix='train'):
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
        file_idx = len(files)
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
    return {f'{name}_nodes': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_edges': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_senders': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_receivers': tf.io.FixedLenFeature([], dtype=tf.string)}


def decode_examples(record_bytes, node_shape=None, edge_shape=None, image_shape=None):
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
            image=tf.io.FixedLenFeature([], dtype=tf.string),
            example_idx=tf.io.FixedLenFeature([], dtype=tf.string),
            **feature_to_graph_tuple('graph')
        )
    )
    image = tf.io.parse_tensor(parsed_example['image'], tf.float32)
    image.set_shape(image_shape)
    example_idx = tf.io.parse_tensor(parsed_example['example_idx'], tf.int32)
    example_idx.set_shape(())
    graph_nodes = tf.io.parse_tensor(parsed_example['graph_nodes'], tf.float32)
    if node_shape is not None:
        graph_nodes.set_shape([None] + list(node_shape))
    graph_edges = tf.io.parse_tensor(parsed_example['graph_edges'], tf.float32)
    if edge_shape is not None:
        graph_edges.set_shape([None] + list(edge_shape))
    receivers = tf.io.parse_tensor(parsed_example['graph_receivers'], tf.int64)
    receivers.set_shape([None])
    senders = tf.io.parse_tensor(parsed_example['graph_senders'], tf.int64)
    senders.set_shape([None])
    graph = GraphsTuple(nodes=graph_nodes,
                        edges=graph_edges,
                        globals=tf.zeros([1]),
                        receivers=receivers,
                        senders=senders,
                        n_node=tf.shape(graph_nodes)[0:1],
                        n_edge=tf.shape(graph_edges)[0:1])
    return (graph, image, example_idx)


def generate_data(data_dirs, save_dir):

    """
    Routine for generating train data in tfrecords

    Args:
        data_dirs: where simulation data is.
        save_dir: where tfrecords will go.

    Returns: list of tfrecords.
    """
    def data_generator():
        print("Making graphs.")
        for idx, dir in tqdm(enumerate(data_dirs)):
            print("Generating data from {}".format(dir))
            positions, properties, image = _get_data(dir)
            # graph = generate_example(positions, properties, k_mean=26)
            graph = generate_example_nn(positions, properties, k=3, plot=False)
            yield (graph, image, idx)

    train_tfrecords = save_examples(data_generator(),
                                    save_dir,
                                    examples_per_file=32,
                                    num_examples=len(data_dirs),
                                    prefix='train')
    return train_tfrecords

###
# specific to project

def make_virtual_node(positions, properties):
    """
    Aggregate positions and properties of nodes into one virtual node.

    Args:
        positions: [N, 3]
        properties: [N, F0,...Fd]

    Returns: [3], [F0,...,Fd]
    """
    return np.mean(positions, axis=0), np.mean(properties, axis=0)


def _get_data(dir):
    """
    Should return the information for a single simulation.

    Args:
        dir: directory with sim data.

    Returns:
        positions for building graph
        properties for putting in nodes and aggregating upwards
        image corresponding to the graph

    """
    print("Generating fake data.")
    f = np.load(os.path.join(dir, 'data.npz'))
    positions = f['positions']
    properties = f['properties']
    image = f['image']
    # positions = np.random.uniform(0., 1., size=(50, 3))
    # properties = np.random.uniform(0., 1., size=(50, 5))
    # image = np.random.uniform(size=(24, 24, 1))
    return positions, properties, image

def make_tutorial_data(examples_dir):
    for i in range(10):
        example_idx = len(glob.glob(os.path.join(examples_dir,'example_*')))
        data_dir = os.path.join(examples_dir,'example_{:04d}'.format(example_idx))
        os.makedirs(data_dir, exist_ok=True)
        positions = np.random.uniform(0., 1., size=(50, 3))
        properties = np.random.uniform(0., 1., size=(50, 5))
        image = np.random.uniform(size=(24, 24, 1))
        np.savez(os.path.join(data_dir, 'data.npz'), positions=positions, properties=properties, image=image)

if __name__ == '__main__':
    make_tutorial_data('tutorial_data')
    tfrecords = generate_data(glob.glob(os.path.join('tutorial_data','example_*')), 'test_train_data')
    dataset = tf.data.TFRecordDataset(tfrecords).map(
        lambda record_bytes: decode_examples(record_bytes, edge_shape=[2], node_shape=[5]))
    loaded_graph = iter(dataset)
    for (graph, image, example_idx) in iter(dataset):
        print(graph.nodes.shape, image.shape, example_idx)
