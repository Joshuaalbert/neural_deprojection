import os
from typing import List

import tensorflow as tf

from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs, networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from networkx.drawing import draw
from tqdm import tqdm
from scipy.optimize import bisect
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

    return bisect(loss, 0., dist_max, xtol=0.001)


def make_virtual_node(positions, properties):
    """
    Aggregate positions and properties of nodes into one virtual node.

    Args:
        positions: [N, 3]
        properties: [N, F]

    Returns: [3], [F]
    """
    return np.mean(positions, axis=0), np.mean(properties, axis=0)

def generate_example(positions, properties, k_mean=26):
    """
    Generate a geometric graph from positions.

    Args:
        positions: [num_points, 3]
        properties: [num_points, F]
        k_mean: float

    Returns: GraphTuple
    """
    graph = nx.DiGraph()
    edge_colours = []
    while positions.shape[0] > 1:
        #n_nodes, n_nodes
        dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        opt_screen_length = find_screen_length(dist, k_mean)
        print("Found optimal screening length {}".format(opt_screen_length))

        distance_matrix_no_loops = np.where(dist == 0., np.inf, dist)
        A = distance_matrix_no_loops < opt_screen_length

        senders, receivers = np.where(A)
        n_edge = senders.size
        # [1,0] for siblings, [0,1] for parent-child
        sibling_edges = np.tile([[1.,0.]], [n_edge, 1])

        # num_points, 3+F
        sibling_nodes = np.concatenate([positions, properties], axis=-1)
        n_nodes = sibling_nodes.shape[0]

        sibling_node_offset = len(graph.nodes)
        for node, feature in zip(np.arange(sibling_node_offset, sibling_node_offset+n_nodes), sibling_nodes):
            graph.add_node(node, features=feature)

        # edges = np.stack([senders, receivers], axis=-1) + sibling_node_offset
        for u,v in zip(senders+sibling_node_offset, receivers+sibling_node_offset):
            graph.add_edge(u,v,features=np.array([1., 0.]))
            graph.add_edge(v,u,features=np.array([1., 0.]))
            edge_colours.append('blue')
            edge_colours.append('blue')

        # for virtual nodes
        sibling_graph = GraphsTuple(nodes=sibling_nodes,
                            edges=sibling_edges,
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
        print(list(connected_components))
        for connected_component in connected_components:
            indices = list(sorted(list(connected_component)))
            virtual_position, virtual_property = make_virtual_node(positions[indices, :], properties[indices, :])
            _positions.append(virtual_position)
            _properties.append(virtual_property)

        virtual_positions = np.stack(_positions, axis=0)
        virtual_properties = np.stack(_properties, axis=0)

        ###
        # add virutal nodes
        # num_parents, 3+F
        parent_nodes = np.concatenate([virtual_positions, virtual_properties], axis=-1)
        n_nodes = parent_nodes.shape[0]
        parent_node_offset = len(graph.nodes)
        parent_indices = np.arange(parent_node_offset, parent_node_offset + n_nodes)
        # adding the nodes to global graph
        for node, feature in zip(parent_indices, parent_nodes):
            graph.add_node(node, features=feature)
            print("new virtual", node)

        for parent_idx, connected_component in zip(parent_indices, connected_components):

            child_node_indices = [idx + sibling_node_offset for idx in list(sorted(list(connected_component)))]
            print(child_node_indices)
            for child_node_idx in child_node_indices:
                graph.add_edge(parent_idx, child_node_idx, features=np.array([0., 1.]))
                graph.add_edge(child_node_idx, parent_idx, features=np.array([0., 1.]))
                edge_colours.append('red')
                edge_colours.append('red')
                print(parent_idx, child_node_idx)

        positions = virtual_positions
        properties = virtual_properties

    [print(graph.nodes[n]['features']) for n in graph.nodes]

    draw(graph, pos={n:graph.nodes[n]['features'][:2] for n in graph.nodes}, edge_color=edge_colours)
    plt.show()
    return graph


def generate_data(data_dir, num_examples):
    target_graphs, graphs, rank = [], [], []
    for i in range(num_examples):
        n_nodes = np.random.randint(100, 120)
        k_mean = np.log(n_nodes)  # np.random.randint(-n_nodes//2, n_nodes//2))
        _target_graph, _graphs, _rank = generate_example(n_nodes, k_mean, dim=2)
        target_graphs = target_graphs + [_target_graph] * len(_rank)
        graphs = graphs + _graphs
        rank = rank + _rank
    graphs = networkxs_to_graphs_tuple(graphs)
    target_graphs = networkxs_to_graphs_tuple(target_graphs)
    train_tfrecords = save_examples(target_graphs, graphs, rank, data_dir, examples_per_file=32, prefix='train')

    return train_tfrecords


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


def save_examples(target_graphs: GraphsTuple, graphs: GraphsTuple, rank: List[tf.Tensor], save_dir=None,
                  examples_per_file=32, prefix='train'):
    """
    Saves a list of GraphTuples to tfrecords.

    Args:
        graphs: list of GraphTuples
        images: list of images
        save_dir: dir to save in
        examples_per_file: int, max number examples per file

    Returns: list of tfrecord files.
    """
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    file_idx = 0
    files = set()
    # file = os.path.join(save_dir, 'train_{:03d}.tfrecords'.format(file_idx))
    file = os.path.join(save_dir, f'{prefix}_all.tfrecords')

    with tf.io.TFRecordWriter(file) as writer:
        for i in tqdm(range(target_graphs.n_node.shape[0])):
            target_graph = get_graph(target_graphs, i)
            graph = get_graph(graphs, i)
            r = rank[i]
            if count == examples_per_file:
                count = 0
                file_idx += 1
            features = dict(
                rank=tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(r, tf.float32)).numpy()])),
                **graph_tuple_to_feature(target_graph, name='target_graph'),
                **graph_tuple_to_feature(graph, name='graph')
            )
            features = tf.train.Features(feature=features)
            example = tf.train.Example(features=features)
            files.add(file)
            writer.write(example.SerializeToString())
            count += 1
    files = list(files)
    print("Saved in tfrecords: {}".format(files))
    return files


def feature_to_graph_tuple(name=''):
    return {f'{name}_nodes': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_edges': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_senders': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_receivers': tf.io.FixedLenFeature([], dtype=tf.string)}


def decode_examples(record_bytes):
    """
    Decodes raw bytes as returned from tf.data.TFRecordDataset([example_path]) into a GraphTuple and image
    Args:
        record_bytes: raw bytes

    Returns: (GraphTuple, image)
    """
    parsed_example = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        dict(
            rank=tf.io.FixedLenFeature([], dtype=tf.string),
            **feature_to_graph_tuple('target_graph'),
            **feature_to_graph_tuple('graph')
        )
    )
    rank = tf.io.parse_tensor(parsed_example['rank'], tf.float32)
    rank.set_shape([])
    rank = rank[None]
    target_graph_nodes = tf.io.parse_tensor(parsed_example['target_graph_nodes'], tf.float32)
    target_graph_nodes.set_shape([None,2])
    target_graph_edges = tf.io.parse_tensor(parsed_example['target_graph_edges'], tf.float32)
    receivers = tf.io.parse_tensor(parsed_example['target_graph_receivers'], tf.int64)
    receivers.set_shape([None])
    senders = tf.io.parse_tensor(parsed_example['target_graph_senders'], tf.int64)
    senders.set_shape([None])
    target_graph = GraphsTuple(nodes=target_graph_nodes,
                               edges=target_graph_edges,
                               globals=None,
                               receivers=receivers,
                               senders=senders,
                               n_node=tf.shape(target_graph_nodes)[0:1],
                               n_edge=tf.shape(target_graph_edges)[0:1])
    graph_nodes = tf.io.parse_tensor(parsed_example['graph_nodes'], tf.float32)
    graph_nodes.set_shape([None,2])
    graph_edges = tf.io.parse_tensor(parsed_example['graph_edges'], tf.float32)
    receivers = tf.io.parse_tensor(parsed_example['graph_receivers'], tf.int64)
    receivers.set_shape([None])
    senders = tf.io.parse_tensor(parsed_example['graph_senders'], tf.int64)
    senders.set_shape([None])
    graph = GraphsTuple(nodes=graph_nodes,
                        edges=graph_edges,
                        globals=None,
                        receivers=receivers,
                        senders=senders,
                        n_node=tf.shape(graph_nodes)[0:1],
                        n_edge=tf.shape(graph_edges)[0:1])
    return (target_graph, graph, rank)


if __name__ == '__main__':
    positions = np.random.uniform(0.,1.,size=(50, 3))
    properties = positions
    generate_example(positions, properties, k_mean=3)
#     generate_example(500, 5, dim=2)
#     # tfrecords = generate_data('./test_data', 100)
#     # dataset = tf.data.TFRecordDataset(tfrecords).map(decode_examples)
#     # loaded_graph = iter(dataset)
#     # for (target_graph, graph, rank) in loaded_graph:
#     #     print(target_graph, graph, rank)
