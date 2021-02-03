import os
import glob
import tensorflow as tf
from itertools import product
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs, networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from networkx.drawing import draw
from tqdm import tqdm
from scipy.optimize import bisect
from scipy.spatial.ckdtree import cKDTree
import pylab as plt
import yt
from astropy.io import fits
import h5py


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

def generate_example_nn(positions, properties, k=26, resolution=0.5, plot=False):
    """
    Generate a k-nn graph from positions.

    Args:
        positions: [num_points, 3] positions used for graph constrution.
        properties: [num_points, F0,...,Fd] each node will have these properties of shape [F0,...,Fd]
        k: int, k nearest neighbours are connected.
        plot: whether to plot graph.

    Returns: GraphTuple
    """
    print('example nn')

    resolution = 3.086e18 * 1000000 * resolution  # pc to cm

    node_features = []
    node_positions = []

    box_size = (np.max(positions), np.min(positions))  # box that encompasses all of the nodes
    print(box_size)
    axis = np.arange(box_size[1], box_size[0], resolution)
    print(f'Axis length = {len(axis)}')
    lists = [axis, axis, axis]
    virtual_node_pos = [p for p in product(*lists)]
    # virtual_node_pos = list(combinations_with_replacement(np.arange(box_size[1], box_size[0], resolution), 3))
    print(f'Lenght virtual nodes pos = {len(virtual_node_pos)}')
    virtual_kdtree = cKDTree(virtual_node_pos)
    print(f'Virtual tree n = {virtual_kdtree.n}')
    print(f'positions = {positions.shape}')
    particle_kdtree = cKDTree(positions)
    print(f'Particle tree n = {particle_kdtree.n}')
    indices = virtual_kdtree.query_ball_tree(particle_kdtree, np.sqrt(3) / 2. * resolution)
    print(len(list(indices)))

    for i, p in enumerate(indices):
        if len(p) == 0:
            continue
        virt_pos, virt_prop = make_virtual_node(positions[p], properties[p])
        node_positions.append(virt_pos)
        node_features.append(virt_prop)

    node_features = np.array(node_features)
    node_positions = np.array(node_positions)
    print(node_features.shape)
    print(node_positions.shape)

    # Directed graph
    graph = nx.DiGraph()

    # Create cKDTree class to find the nearest neighbours of the positions
    kdtree = cKDTree(node_positions)
    # idx has shape (positions, k+1) and contains for every position the indices of the nearest neighbours
    dist, idx = kdtree.query(node_positions, k=k+1)
    #downscale resolution

    # The index of the first nearest neighbour is the position itself, so we discard that one
    receivers = idx[:, 1:]#N,k
    senders = np.arange(node_positions.shape[0])# Just a range from 0 to the number of positions
    senders = np.tile(senders[:, None], [1,k])#N,k
    # senders looks like (for 4 positions and 3 nn's)
    # [[0  0  0]
    #  [1  1  1]
    #  [2  2  2]
    #  [3  3  3]]

    # Every position has k connections and every connection has a sender and a receiver
    # The indices of receivers and senders correspond to each other (so receiver[32] belongs to sender[32])
    # The value of indices in senders and receivers correspond to the index they have in the positions array.
    # (so if sender[32] = 6, then that sender has coordinates positions[6])
    receivers = receivers.flatten() # shape is (len(positions) * k,)
    senders = senders.flatten() # shape is (len(positions) * k,)

    n_nodes = node_positions.shape[0] #number of nodes is the number of positions

    if plot:
        pos = dict()  # for plotting node positions.
        edgelist = []

        # Now put the data in the directed graph: first the nodes with their positions and properties
        # pos just takes the x and y coordinates of the position so a 2D plot can be made
        for node, feature, position in zip(np.arange(n_nodes), node_features, node_positions):
            graph.add_node(node, features=feature)
            pos[node] = (position[:2] - box_size[1])/(box_size[0] - box_size[1])

        # Next add the edges using the receivers and senders arrays we just created
        # Note that an edge is added for both directions
        # The features of the edge are dummy arrays at the moment
        # The edgelist is for the plotting
        # edges = np.stack([senders, receivers], axis=-1) + sibling_node_offset
        for u, v in zip(senders, receivers):
            graph.add_edge(u, v, features=np.array([1., 0.]))
            graph.add_edge(v, u, features=np.array([1., 0.]))
            edgelist.append((u, v))
            edgelist.append((v, u))

        print(graph.number_of_nodes())
        print(graph.number_of_edges())
        print(pos)

        #Plotting
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        print('check 1')
        draw(graph, ax=ax[0], pos=pos, node_color='green', edge_color='red')
        ax[1].scatter(positions[:,0], positions[:,1],s=2.)
        print('check 2')
        plt.show()
        print('check 3')


    else:
        for node, feature, position in zip(np.arange(n_nodes), node_features, node_positions):
                graph.add_node(node, features=feature)

        for u, v in zip(senders, receivers):
                graph.add_edge(u, v, features=np.array([1., 0.]))
                graph.add_edge(v, u, features=np.array([1., 0.]))

    # Global dummy variable, this needs to be defined in order to turn the graph into a graph tuple,
    # see networkxs_to_graphs_tuple documentation
    graph.graph["features"] = np.array([0.])

    #Important step: return the graph, which is a networkx class, as a graphs tuple!
    # positions.shape[1] = 3, properties.shape[1] = the number of features,
    # so node_shape_hint tells the function the number of length of the attribute for every node
    # edge_shape_hint: the edges, at the moment, have a dummy attribute of size two
    return networkxs_to_graphs_tuple([graph],
                                     node_shape_hint=[node_features.shape[1]],
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

    # If the directory where to save the tfrecords is not specified, save them in the current working directory
    if save_dir is None:
        save_dir = os.getcwd()
    # If the save directory does not yet exist, create it
    os.makedirs(save_dir, exist_ok=True)
    # Files will be returned
    files = []
    # next(data_iterable) gives the next dataset in the iterable
    data_iterable = iter(generator)
    data_left = True
    # Status bar
    pbar = tqdm(total=num_examples)
    while data_left:
        # For every 'examples_per_file' (=32) example directories, create a tf_records file
        file_idx = len(files)
        file = os.path.join(save_dir, 'train_{:04d}.tfrecords'.format(file_idx))
        files.append(file)
        # 'writer' can write to 'file'
        with tf.io.TFRecordWriter(file) as writer:
            for i in range(examples_per_file):
                # Yield a dataset extracted by the generator
                try:
                    (graph, image, example_idx) = next(data_iterable)
                except StopIteration:
                    data_left = False
                    break
                # Write the graph, image and example_idx to the tfrecord file
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
                # Set the features up so they can be written to the tfrecord file
                features = tf.train.Features(feature=features)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                # Status bar update
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

        # Take all the data directories in data_dirs and yield a graph, image and index for each directory
        # (directories in 'tutorial_data' in this case which contain data.npz files)
        for idx, dir in tqdm(enumerate(data_dirs)):
            print("Generating data from {}".format(dir))

            # For every dir (contains a data.npz file) extract the numpy arrays
            # So the data in numpy arrays, which were first turned into data.npz files
            # are now extracted again as numpy arrays and converted (at least the positions and properties) to a graph.
            print(dir)
            positions, properties, image = _get_data(dir)

            # Create a graph with the positions and properties
            # Use generate_example for Julius' case and generate_example_nn for Matthijs' case.

            # graph = generate_example(positions, properties, k_mean=26)
            graph = generate_example_nn(positions, properties, plot=True)

            # This function is a generator, so it doesn't keep used and upcoming data in memory.
            yield (graph, image, idx)

    # Save the data as tfrecords and return the filenames of the tfrecords
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

    # Load the data.npz file and extract and return the data (numpy arrays)
    f = np.load(os.path.join(dir, 'data.npz'))
    positions = f['positions']
    properties = f['properties']
    image = f['image']
    # positions = np.random.uniform(0., 1., size=(50, 3))
    # properties = np.random.uniform(0., 1., size=(50, 5))
    # image = np.random.uniform(size=(24, 24, 1))
    return positions, properties, image

def make_tutorial_data(examples_dir):
    # Directory that contains magneticum cluster directories
    # which contain a snapshot (particle data) of the cluster and an xray image
    magneticum_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/graph_test/Magneticum/'

    # Directory that contains bahamas cluster directories
    # which contain a snapshot (particle data) of the cluster and an xray image
    bahamas_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/graph_test/Bahamas_small/'

    # Create a list of directories which contain cluster data
    magneticum_clusters = glob.glob(magneticum_dir + '*')
    bahamas_clusters = glob.glob(bahamas_dir + '*')

    # Field structure for magnecticum snapshots (which is not default)
    my_field_def = (
        "Coordinates",
        "Velocities",
        "Mass",
        "ParticleIDs",
        ("InternalEnergy", "Gas"),
        ("Density", "Gas"),
        ("SmoothingLength", "Gas"),
    )

    for cluster_dir in bahamas_clusters:
        # Define the snapshot filename
        # The magneticum snapshots have no file extension and the bahamas snapshots have a .hdf5 extension
        snap_file = list(set(glob.glob(cluster_dir + "/*")) - set(glob.glob(cluster_dir + "/*.*")))
        if len(snap_file) == 0:
            snap_file = glob.glob(cluster_dir + '/*.hdf5')[0]
            magneticum = False
        else:
            snap_file = snap_file[0]
            magneticum = True

        # Define the xray image filename (which has a .fits extension)
        xray_file = glob.glob(cluster_dir + '/*.fits')[0]

        print(snap_file)
        print(xray_file)

        if magneticum:
            # Load the data for a magneticum cluster
            # Set long_ids = True because the IDs are 64-bit ints
            ds_yt = yt.load(snap_file, long_ids=True, field_spec=my_field_def)
            ad = ds_yt.all_data()

            positions = ad['Gas', 'Coordinates'].d
            velocities = ad['Gas', 'Velocities'].d
            rho = ad['Gas', 'Density'].d
            u = ad['Gas', 'InternalEnergy'].d
            extra_info = 'Magneticum'
        else:
            # Load the data for a bahamas cluster
            with h5py.File(snap_file, 'r') as ds_h5:

                positions = np.array(ds_h5['PartType0']['Coordinates'])
                positions = (positions - np.mean(positions, axis=0)) / 1e6
                velocities = np.log(np.array(ds_h5['PartType0']['Velocity']) / 1e6)
                rho = np.log(np.array(ds_h5['PartType0']['Density']) / 1e-29)
                u = np.log(np.array(ds_h5['PartType0']['InternalEnergy']) / 1e15)
                ds_h5.close()
            extra_info = 'Bahamas'

        # Combine the properties in a single array
        p_T = positions.T
        v_T = velocities.T
        properties = np.stack((p_T[0], p_T[1], p_T[2], v_T[0], v_T[1], v_T[2], rho, u), axis=1)
        # properties = np.stack((rho, u), axis=1)


        # Load the xray image
        with fits.open(xray_file) as hdul:
            image = np.array(hdul[0].data, dtype='float64').reshape(4880,4880,1)

        # Print the number of particles
        print('The number of gas particles in {} is {}'.format(snap_file.split('/')[-1], positions.shape[0]))

        # Create an example directory in the 'examples_dir'
        # and put a data.npz file in there that contains al the info of the snapshot/xray img pair
        example_idx = len(glob.glob(os.path.join(examples_dir, 'example_*')))
        new_dir = os.path.join(examples_dir, "example_{:04d}".format(example_idx))
        os.makedirs(new_dir, exist_ok=True)
        print(new_dir)
        np.savez(new_dir + "/data.npz", positions=positions, properties=properties, image=image,
                 extra_info=extra_info)

    """
    #Create 10 examples
    for i in range(10):
        # Create a new data directory for a new example
        example_idx = len(glob.glob(os.path.join(examples_dir,'example_*')))
        data_dir = os.path.join(examples_dir,'example_{:04d}'.format(example_idx))
        os.makedirs(data_dir, exist_ok=True)
        # Define positions, properties and images here
        # The example uses random numbers
        # This needs to be replaced by actual data
        positions = np.random.uniform(0., 1., size=(50, 3))
        properties = np.random.uniform(0., 1., size=(50, 5))
        image = np.random.uniform(size=(24, 24, 1))
        # Save the data as an data.npz file
        np.savez(os.path.join(data_dir, 'data.npz'), positions=positions, properties=properties, image=image)
    """

if __name__ == '__main__':
    # Save data in 'tutorial_data'
    # In this example it's random data, but eventually we want to use actual data
    data_dir = 'cluster_data_small'
    test_train_dir = 'test_train_data_small'

    if not os.path.isdir(data_dir):
        make_tutorial_data(data_dir)
    else:
        pass

    # Save the data from 'tutorial data' as tfrecords
    # tfrecords are saved in test_train_data
    if not os.path.isdir(test_train_dir):
        tfrecords = generate_data(glob.glob(os.path.join(data_dir,'example_*')), test_train_dir)
    else:
        tfrecords = glob.glob(test_train_dir + '/*')

    print(tfrecords)


    # Decode the tfrecord files again into the (graph tuple, image, index) dataset
    # This is also used in main.py to retrieve the datasets for a neural network
    dataset = tf.data.TFRecordDataset(tfrecords).map(
        lambda record_bytes: decode_examples(record_bytes, edge_shape=[2], node_shape=[8]))

    # make_tutorial_data() + numpy arrays (pos, props, img) --> data.npz

    # generate_data(
    # _get_data() + data.npz  --> numpy arrays (pos, props, img)

    # generate_example_nn() + numpy arrays (pos, props, img) --> graph tuple, image and index (graph, img, idx)

    # save_examples() + graph tuple, image and index (graph, img, idx) --> tfrecord
    # )

    # decode_examples() + tfrecord --> graph tuple, image and index (graph, img, idx)

    loaded_graph = iter(dataset)
    # example index is an integer as a tensorflow tensor
    for (graph, image, example_idx) in iter(dataset):
        print(graph.nodes.shape, image.shape, example_idx)
