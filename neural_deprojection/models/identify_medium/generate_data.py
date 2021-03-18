import os
import glob
import tensorflow as tf

from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from networkx.drawing import draw
from tqdm import tqdm
from scipy.spatial.ckdtree import cKDTree
import pylab as plt


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
    graph = nx.OrderedMultiDiGraph()

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
        graph.add_edge(u, v, features=None)
        graph.add_edge(v, u, features=None)
        edgelist.append((u, v))
        edgelist.append((v, u))

    graph.graph['features'] = None

    # plotting

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        draw(graph, ax=ax, pos=pos, node_color='green', edge_color='red')
        plt.show()

    return networkxs_to_graphs_tuple([graph], node_shape_hint=[positions.shape[1] + properties.shape[1]])


def graph_tuple_to_feature(graph: GraphsTuple, name=''):
    schema = {}
    schema[f'{name}_nodes'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.nodes, tf.float32)).numpy()]))
    schema[f'{name}_senders'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.senders, tf.int64)).numpy()]))
    schema[f'{name}_receivers'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.receivers, tf.int64)).numpy()]))
    return schema


def save_examples(generator,
                  save_dir=None,
                  num_examples=1):
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
    data_iterable = iter(generator)
    pbar = tqdm(total=num_examples)

    file = os.path.join(save_dir, 'examples.tfrecords')
    with tf.io.TFRecordWriter(file) as writer:
        for (graph, image, example_idx) in data_iterable:
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
    files = [file]
    print("Saved in tfrecords: {}".format(files))
    return files


def feature_to_graph_tuple(name=''):
    schema = {}
    schema[f'{name}_nodes'] = tf.io.FixedLenFeature([], dtype=tf.string)
    schema[f'{name}_senders'] = tf.io.FixedLenFeature([], dtype=tf.string)
    schema[f'{name}_receivers'] = tf.io.FixedLenFeature([], dtype=tf.string)
    return schema


def decode_examples(record_bytes, node_shape=None, image_shape=None):
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
    graph_nodes.set_shape([None] + list(node_shape))
    receivers = tf.io.parse_tensor(parsed_example['graph_receivers'], tf.int64)
    receivers.set_shape([None])
    senders = tf.io.parse_tensor(parsed_example['graph_senders'], tf.int64)
    senders.set_shape([None])
    n_node = tf.shape(graph_nodes)[0:1]
    n_edge = tf.shape(senders)[0:1]

    # graph = GraphsTuple(nodes=graph_nodes,
    #                     edges=None,
    #                     globals=None,
    #                     receivers=receivers,
    #                     senders=senders,
    #                     n_node=n_node,
    #                     n_edge=n_edge)

    graph_data_dict = dict(nodes=graph_nodes,
                        receivers=receivers,
                        senders=senders,
                        n_node=n_node,
                        n_edge=n_edge)
    return (graph_data_dict, image, example_idx)


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
            graph = generate_example_nn(positions, properties, k=6, plot=False)
            yield (graph, image, idx)

    train_tfrecords = save_examples(data_generator(),
                                    save_dir,
                                    num_examples=len(data_dirs))
    return train_tfrecords


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
    f = np.load(os.path.join(dir, 'data.npz'))
    positions = f['positions']
    properties = f['properties']
    image = f['image']
    return positions, properties, image

def make_tutorial_data(examples_dir, N=100, num_nodes=3000):
    print("Generating fake data.")
    for i in range(N):
        example_idx = len(glob.glob(os.path.join(examples_dir,'example_*')))
        data_dir = os.path.join(examples_dir,'example_{:04d}'.format(example_idx))
        os.makedirs(data_dir, exist_ok=True)
        positions = np.random.uniform(0., 1., size=(num_nodes, 3))
        density = np.exp(-np.linalg.norm(positions - np.random.uniform(size=3), axis=-1)/0.1)
        density += np.exp(-np.linalg.norm(positions - np.random.uniform(size=3), axis=-1)/0.1)
        density += np.exp(-np.linalg.norm(positions - np.random.uniform(size=3), axis=-1)/0.1)
        properties = np.concatenate([positions,density[:,None]], axis=-1)
        image = np.histogram2d(positions[:,0], positions[:,1], weights=density,
                               bins=(np.linspace(0.,1., 19), np.linspace(0.,1., 19)))[0][:,:,None]
        image += np.percentile(image.flatten(), 5) * np.random.normal(size=image.shape)
        np.savez(os.path.join(data_dir, 'data.npz'), positions=positions, properties=properties, image=image)



if __name__ == '__main__':
    make_tutorial_data('tutorial_train_data',N=100)
    make_tutorial_data('tutorial_test_data',N=100)
    tfrecords = generate_data(glob.glob(os.path.join('tutorial_train_data','example_*')), os.path.join('data','train'))
    tfrecords = generate_data(glob.glob(os.path.join('tutorial_test_data','example_*')), os.path.join('data','test'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(
        lambda record_bytes: decode_examples(record_bytes, node_shape=[4]))
    loaded_graph = iter(dataset)
    for (graph, image, example_idx) in iter(dataset):
        print(graph, image.shape, example_idx)
