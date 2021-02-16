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
import csv
from pyxsim import ThermalSourceModel, PhotonList
import soxs
import gadget as g
import pyxsim


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


def generate_example_nn(positions, properties, k=26, resolution=0.2, plot=False):
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

    # resolution = 3.08618 * 1000000 * resolution  # pc to cm

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
    print(f'Node feature shape : {node_features.shape}')
    print(f'Node positions shape : {node_positions.shape}')

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

        print(f'Particle graph nodes : {graph.number_of_nodes()}')
        print(f'Particle graph edges : {graph.number_of_edges()}')

        #Plotting
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        print('Plotting...')
        draw(graph, ax=ax[0], pos=pos, node_color='green', edge_color='red')
        ax[1].scatter(positions[:,0], positions[:,1],s=2.)
        print('Plot done, showing...')
        plt.show()


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

    image = tf.where(image <= 0, tf.exp(-5), image)

    image = tf.math.log(image / 43.) + tf.math.log(0.5)
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


def generate_data(data_dirs, save_dir, offsets, scales):

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
        for idx, data_dir in tqdm(enumerate(data_dirs)):
            print("Generating data from {}".format(data_dir))

            # For every dir (contains a data.npz file) extract the numpy arrays
            # So the data in numpy arrays, which were first turned into data.npz files
            # are now extracted again as numpy arrays and converted (at least the positions and properties) to a graph.
            print(data_dir)
            positions, properties, xray_images = _get_data(data_dir)

            if len(offsets) == len(properties) and len(scales) == len(properties):
                properties = (properties - offsets.reshape((len(properties), 1))) / scales.reshape((len(properties), 1))
            else:
                raise ValueError("Offsets and/or scales do not have the same length as properties.")

            xray_images[image == 0] = 1e-5
            xray_images = np.log(xray_images)

            # Create a graph with the positions and properties
            graph = generate_example_nn(positions, properties, plot=True)

            # This function is a generator, which has the advantage of not keeping used and upcoming data in memory.
            yield (graph, xray_images, idx)

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


def _get_data(data_dir):
    """
    Should return the information for a single simulation.

    Args:
        data_dir: directory with sim data.

    Returns:
        positions for building graph
        properties for putting in nodes and aggregating upwards
        image corresponding to the graph

    """
    print("Generating data.")

    # Load the data.npz file and extract and return the data (numpy arrays)
    f = np.load(os.path.join(data_dir, 'data.npz'))
    positions = f['positions']
    properties = f['properties']
    xray_images = f['images']

    # # Dummy data
    # print("Generating fake data.")
    #
    # positions = np.random.uniform(0., 1., size=(50, 3))
    # properties = np.random.uniform(0., 1., size=(50, 5))
    # image = np.random.uniform(size=(24, 24, 1))

    return positions, properties, xray_images


def line_of_sight_to_string(normal_vector, decimals: int = 2):
    """

    Converts a line_of_sight vector to a string: shows the first 2 decimals of every dimension
    This is added to the xray fits filename

    Args:
        normal_vector: (array) Line of sight vector to transform to string
        decimals: How many decimals to use for the angles

    Returns: example for two decimals: '_67_-07_13'

    """
    normal_string = ''
    for i in normal_vector:
        angle_string = str(i).replace('.', '')
        if angle_string[0] == '-':
            angle_string = '-' + angle_string[2:2+decimals]
        else:
            angle_string = angle_string[1:1+decimals]
        normal_string += '_' + angle_string
    return normal_string


def make_bahamas_clusters(data_dir,
                          save_dir,
                          number_of_clusters=200,
                          starting_cluster=0):
    """

    Args:
        data_dir: Directory containing the bahamas data (e.g. 'AGN_TUNED_nu0_L100N256_WMAP9')
        save_dir: Directory to save the cluster hdf5 files in
        number_of_clusters: Number of clusters for which to make hdf5 files
        starting_cluster: The *number_of_clusters* largest clusters are selected, with index 0 the largest cluster.
        xray: Whether to create xray images for the clusters
        numbers_of_xray_images: How many xrays images to make for every cluster,
        this keyword is not used when xray is False

    Returns:
        Creates in the save_dir hdf5 files of clusters containing particle data
        and possibly a number of xray images of the clusters.

    """

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    #The original file contains all the particles in the simulation
    #Eventually we want to make a new hdf5 file that contains only the particles from a certain cluster
    path = os.path.join(data_dir, 'data/particledata_032/eagle_subfind_particles_032.0.hdf5')
    original_file = h5py.File(path, 'r')
    groups = list(original_file.keys())

    #Determine which groups contain particles and which don't.
    particle_groups = []
    non_particle_groups = []
    for group in groups:
        if 'PartType' in group:
            particle_groups.append(group)
        else:
            non_particle_groups.append(group)

    #Determine particle sub- and subsubgroups.
    #For every group that contains particles, a selection needs to be taken that only contains the particles belonging to a particular cluster
    particle_subgroups = {group: list(original_file[group].keys()) for group in particle_groups}
    particle_subsubgroups = {group: {subgroup: list(original_file[group][subgroup].keys()) for subgroup in ['ElementAbundance', 'SmoothedElementAbundance']} for group in ['PartType0', 'PartType4']}

    #Load the particle data
    snapnum = 32
    pdata = g.Gadget(data_dir, "particles", snapnum, sim="BAHAMAS")

    #Find the FoF group number for every particle for every particle type
    fof_group_number_per_particle = {ptype: pdata.read_var(ptype + '/GroupNumber', verbose=False) for ptype in particle_groups}

    #Find the different FoF groups for every particle type by taking the set of the previous variable,
    #taking the length of the set gives an estimate of the number of 'clusters' per particle type
    fof_sets = {ptype: list(set(fof_group_number_per_particle[ptype])) for ptype in particle_groups}

    for cluster in range(starting_cluster, starting_cluster + number_of_clusters):

        cluster_dir = os.path.join(save_dir, f'cluster_{cluster:03}')
        if not os.path.isdir(cluster_dir):
            os.makedirs(cluster_dir, exist_ok=True)
        #Create a new file in which the cluster particles will be saved
        with h5py.File(os.path.join(cluster_dir, f'cluster_{cluster:03}.hdf5'), 'w') as f2:

            #Non-particle groups can just be copied
            for group in non_particle_groups:
                original_file.copy(group, f2)

            #Take a subset of all particles which are present in the cluster and add their properties to the new hdf5 file
            for group in particle_groups:
                #Indices of the particles in the cluster subset for a certain particle type
                inds = np.where(fof_group_number_per_particle[group] == fof_sets[group][cluster])[0]
                #Make sure there are particles of the current type in the cluster subset
                if len(inds) != 0:
                    for subgroup in particle_subgroups[group]:
                        #These subgroups are actual groups instead of datasets, so we need to go one layer deeper
                        if subgroup in ['ElementAbundance', 'SmoothedElementAbundance']:
                            for subsubgroup in particle_subsubgroups[group][subgroup]:
                                field = group + '/' + subgroup + '/' + subsubgroup
                                #Create a new dataset with the subset of the particles
                                f2.create_dataset(field, data=pdata.read_var(field, verbose=False)[inds])
                                #Also add the attributes
                                for attr in list(original_file[field].attrs.items()):
                                    f2[field].attrs.create(attr[0], attr[1])
                        else:
                            #These 'subgroups' are datasets and can be added directly
                            field = group + '/' + subgroup
                            f2.create_dataset(field, data=pdata.read_var(field, verbose=False)[inds])
                            #Again also add the attributes
                            for attr in list(original_file[field].attrs.items()):
                                f2[field].attrs.create(attr[0], attr[1])
        print(f'cluster {cluster} particles are done')


def make_magneticum_fits(data_dir, snap_file, number_of_xray_images: int = 1):
    """

    Args:
        data_dir: Directory containing the Magneticum data
        snap_file: The snapshot filename (e.g. 'snap_128' or 'snap_132')
        number_of_xray_images: Xray images will be added to the cluster folder
        until this amount of xray images is reached.

    Returns:
        Creates a number of xray images for all clusters with less xray images
        than specified by 'number_of_xray_images'.

    """

    clusters = glob.glob(os.path.join(data_dir, snap_file + '/*/simcut/*/' + snap_file))

    # Parameters for making the xray images
    exp_time = (200., "ks")  # exposure time
    area = (2000.0, "cm**2")  # collecting area
    redshift = 0.10114
    min_E = 0.05  # Minimum energy of photons in keV
    max_E = 11.0  # Maximum energy of photons in keV
    Z = 0.3  # Metallicity in units of solar metallicity
    kT_min = 0.05  # Minimum temperature to solve emission for
    n_chan = 1000  # Number of channels in the spectrum
    nH = 0.04  # The foreground column density in units of 10^22 cm^{-2}. Only used if absorption is applied.
    radius = (2.0, "Mpc")  # Radius of the sphere which captures photons
    sky_center = [45., 30.]  # Ra and dec coordinates of the cluster (which are currently dummy values)

    # The line of sight is oriented such that the north vector
    # would be projected on the xray image, as a line from the center to the top of the image
    north_vector = np.array([0., 1., 0.])

    # Finds the center of the gas particles in the snapshot by taking the average of the position extrema
    def find_center(position, offset=(0., 0., 0.)):
        x, y, z = position.T

        x_max, x_min = max(x), min(x)
        y_max, y_min = max(y), min(y)
        z_max, z_min = max(z), min(z)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        return [x_center + offset[0], y_center + offset[1], z_center + offset[2]], [x_max - x_min, y_max - y_min, z_max - z_min]

    # SimCut files (until November 24, 2020) have fields in this order, so we need to specify them
    # my_field_def = (
    #     "Coordinates",
    #     "Velocities",
    #     "Mass",
    #     "ParticleIDs",
    #     ("InternalEnergy", "Gas"),
    #     ("Density", "Gas"),
    #     ("SmoothingLength", "Gas"),
    # )

    # For every selected snapshot, create an xray image for one or more lines of sight
    for cluster in clusters:
        # Take a number of random lines of sight to create the xray images
        existing_xray_images = len(glob.glob(os.path.dirname(cluster) + '/xray_image*'))

        if existing_xray_images >= number_of_xray_images:
            continue

        print(f"Making {number_of_xray_images - existing_xray_images} new xray images in cluster {cluster.split('/')[-4]}")

        # Set long_ids = True because the IDs are 64-bit ints
        ds = yt.load(cluster, long_ids=True)
        print(ds.field_list)

        # Use the gas particle positions to find the center of the snapshot
        ad = ds.all_data()
        pos = ad['Gas', 'Coordinates'].d
        c = find_center(pos, [0., 0., 0.])[0]

        # Create a sphere around the center of the snapshot, which captures the photons
        sp = ds.sphere(c, radius)
        yt.ProjectionPlot(ds, "z", 'Density', center=c, width=(1000000.0, "kpc")).save()

        # Set a minimum temperature to leave out that shouldn't be X-ray emitting,
        # set metallicity to 0.3 Zsolar (should maybe fix later)
        # The source model determines the distribution of photons that are emitted
        source_model = pyxsim.ThermalSourceModel("apec", min_E, max_E, n_chan, Zmet=Z, kT_min=kT_min)

        # Create the photonlist
        photons = pyxsim.PhotonList.from_data_source(sp, redshift, area, exp_time, source_model)

        lines_of_sight = 2.0 * (np.random.random((number_of_xray_images - existing_xray_images, 3)) - 0.5)

        # Make an xray image for a set of lines of sight
        for line_of_sight in lines_of_sight:
            # Finds the events along a certain line of sight
            events_z = photons.project_photons(line_of_sight, sky_center, absorb_model="tbabs", nH=nH, north_vector=north_vector)

            events_z.write_simput_file("magneticum", overwrite=True)

            # Determine which events get detected by the AcisI intstrument of Chandra
            soxs.instrument_simulator("magneticum_simput.fits", "magneticum_evt.fits", exp_time, "chandra_acisi_cy0",
                                      sky_center, overwrite=True, ptsrc_bkgnd=False, foreground=False, instr_bkgnd=False)

            soxs.write_image("magneticum_evt.fits",
                             os.path.join(os.path.dirname(cluster),
                                          "xray_image" + line_of_sight_to_string(line_of_sight) + ".fits"),
                             emin=min_E,
                             emax=max_E,
                             overwrite=True)


def make_bahamas_fits(data_dir, number_of_xray_images=1):

    cluster_dirs = glob.glob(os.path.join(data_dir, 'cluster_*')).sort()
    clusters = [int(os.path.basename(cluster_dir).split('_')[1]) for cluster_dir in cluster_dirs]

    # Define the centers of clusters as the center of potential of friends-of-friends groups
    # 'subh' stands for subhalos
    snapnum = 32
    gdata = g.Gadget(data_dir, 'subh', snapnum, sim='BAHAMAS')
    centers = gdata.read_var('FOF/GroupCentreOfPotential', verbose=False)
    # Convert to codelength by going from cm to Mpc and from Mpc to codelength
    centers /= gdata.cm_per_mpc * 1.42855 #maybe a factor 4 for the big simulation

    # Parameters for making the xray images
    exp_time = (200., "ks")  # exposure time
    area = (2000.0, "cm**2")  # collecting area
    redshift = 0.05
    min_E = 0.05  # Minimum energy of photons in keV
    max_E = 11.0  # Maximum energy of photons in keV
    Z = 0.3  # Metallicity in units of solar metallicity
    kT_min = 0.05  # Minimum temperature to solve emission for
    n_chan = 1000  # Number of channels in the spectrum
    nH = 0.04  # The foreground column density in units of 10^22 cm^{-2}. Only used if absorption is applied.
    radius = 5.0  # Radius of the sphere which captures photons
    sky_center = [45., 30.]  # Ra and dec coordinates of the cluster (which are currently dummy values)

    # The line of sight is oriented such that the north vector
    # would be projected on the xray image as a line from the center to the top of the image.
    north_vector = np.array([0., 1., 0.])

    # Set a minimum temperature to leave out that shouldn't be X-ray emitting, set metallicity to 0.3 Zsolar (should maybe fix later)
    # The source model determines the energy distribution of photons that are emitted
    source_model = ThermalSourceModel("apec", min_E, max_E, n_chan, Zmet=Z)

    # For some reason the Bahamas snapshots are structured so that when you load one snapshot, you load the entire simulation box
    # so there is not a specific reason to choose the first element of filenames
    filenames = glob.glob(os.path.join(data_dir, 'data/snapshot_032/*'))
    snap_file = filenames[0]
    ds = yt.load(snap_file)

    for cluster in clusters:
        # Create a sphere around the center of the snapshot, which captures the photons
        sphere_center = centers[cluster]
        sp = ds.sphere(sphere_center, radius)

        # Create the photonlist
        photons = PhotonList.from_data_source(sp, redshift, area, exp_time, source_model)

        # Take a number of random lines of sight to create the xray images
        lines_of_sight = 2.0 * (np.random.random((number_of_xray_images, 3)) - 0.5)

        for line_of_sight in lines_of_sight:
            # Finds the events along a certain line of sight
            events_z = photons.project_photons(line_of_sight, sky_center, absorb_model="tbabs", nH=nH,
                                               north_vector=north_vector)

            events_z.write_simput_file("bahamas", overwrite=True)

            # Determine which events get detected by the AcisI intstrument of Chandra
            soxs.instrument_simulator("bahamas_simput.fits", "bahamas_evt.fits", exp_time, "chandra_acisi_cy0",
                                      sky_center, overwrite=True, ptsrc_bkgnd=False, foreground=False, instr_bkgnd=False)

            # Write the detections to a fits file
            soxs.write_image("bahamas_evt.fits",
                             os.path.join(cluster_dirs[cluster],
                                          f"img_{cluster:03}" + line_of_sight_to_string(line_of_sight) + ".fits"),
                             emin=min_E,
                             emax=max_E,
                             overwrite=True)
        print(f'cluster {cluster} image is done')


def snapshots_and_names(magneticum_data_dir, bahamas_data_dir, magneticum_snapshots, bahamas_snapshots):
    """

    Returns filenames containing cluster particle data.

    Args:
        magneticum_data_dir: (string) Directory containing the Magneticum data
        bahamas_data_dir: (string) Directory containing the BAHAMAS data
        magneticum_snapshots: (list of strings) Magneticum snapshots for which to return cluster filenames
        bahamas_snapshots: (list of strings) BAHAMAS snapshots for which to return cluster filenames

    Returns: List of snapshot lists containing cluster filenames. Also a list of snapshot names.

    """
    snapshots = []
    snapshot_names = []

    for snapshot in magneticum_snapshots:
        try:
            clusters = glob.glob(os.path.join(magneticum_data_dir, snapshot + '/*/simcut/*/' + snapshot))

            snapshots.append(clusters)
            snapshot_names.append(snapshot)
        except:
            print(f'No data directory for snapshot {snapshot}')
            continue

    for snapshot in bahamas_snapshots:
        try:
            clusters = glob.glob(os.path.join(bahamas_data_dir, snapshot + '/*/cluster_*'))
            snapshots.append(clusters)
            snapshot_names.append(snapshot)
        except:
            print(f'No data directory for snapshot {snapshot}')
            continue

    return snapshots, snapshot_names


def extract_data(cluster, snapshot_name):
    """

    Extracts the cluster position, velocity, density and internal energy data and returns the data in numpy arrays.

    Args:
        cluster: (string) Filename containing the cluster data
        snapshot_name: (string) Name of the snapshot

    Returns:

    """
    my_field_def = (
        "Coordinates",
        "Velocities",
        "Mass",
        "ParticleIDs",
        ("InternalEnergy", "Gas"),
        ("Density", "Gas"),
        ("SmoothingLength", "Gas"),
    )

    if snapshot_name[0:3] == 'AGN':
        with h5py.File(cluster, 'r') as ds_h5:
            positions = np.array(ds_h5['PartType0']['Coordinates'])
            velocities = np.array(ds_h5['PartType0']['Velocity'])
            rho = np.array(ds_h5['PartType0']['Density'])
            u = np.array(ds_h5['PartType0']['InternalEnergy'])
    else:
        ds_yt = yt.load(cluster, long_ids=True, field_spec=my_field_def)
        ad = ds_yt.all_data()

        positions = ad['Gas', 'Coordinates'].d
        velocities = ad['Gas', 'Velocities'].d
        rho = ad['Gas', 'Density'].d
        u = ad['Gas', 'InternalEnergy'].d
    return positions, velocities, rho, u


def combine_data(magneticum_data_dir, bahamas_data_dir, magneticum_snapshots, bahamas_snapshots):
    """
    This function extracts the particle and image data from the binary (Magneticum), hdf5 (BAHAMAS) and fits (xrays)
    files. The data from each cluster is combined in a data.npz file and saved in a separate example directory.

    Args:
        magneticum_data_dir: (string) Directory containing the Magneticum data
        bahamas_data_dir: (string) Directory containing the BAHAMAS data
        magneticum_snapshots: (list of strings) Magneticum snapshots for which to combine and save the data
        bahamas_snapshots: (list of strings) BAHAMAS snapshots for which to combine and save the data

    """
    snapshots, snapshot_names = snapshots_and_names(magneticum_data_dir,
                                                    bahamas_data_dir,
                                                    magneticum_snapshots,
                                                    bahamas_snapshots)

    for idx, snapshot in enumerate(snapshots):

        for cluster in snapshot:
            positions, velocities, rho, u = extract_data(cluster, snapshot_names[idx])

            # Combine the properties in a single array
            p_t = positions.T
            v_t = velocities.T
            properties = np.stack((p_t[0], p_t[1], p_t[2], v_t[0], v_t[1], v_t[2], rho, u), axis=1)

            xray_images = glob.glob(os.path.join(os.path.dirname(cluster) + 'xray*'))
            images = []

            for xray_image in xray_images:
                # Load the xray image
                with fits.open(xray_image) as hdu:
                    image = np.array(hdu[0].data, dtype='float32').reshape(4880, 4880, 1)

                images.append(image)

            images = np.array(images)

            # Print the number of particles
            print('The number of gas particles in {} is {}'.format(cluster.split('/')[-1], positions.shape[0]))

            # Create an example directory in the 'examples_dir'
            # and put a data.npz file in there that contains al the info of the snapshot/xray img pair
            example_idx = len(glob.glob(os.path.join(snapshot_names[idx] + '_examples', 'example_*')))
            new_dir = os.path.join(snapshot_names[idx] + '_examples', "example_{:04d}".format(example_idx))
            os.makedirs(new_dir, exist_ok=True)
            print(new_dir)
            np.savez(new_dir + "/data.npz", positions=positions, properties=properties, images=images)


def make_data_csv(magneticum_data_dir, bahamas_data_dir, magneticum_snapshots, bahamas_snapshots):
    """
    This function creates csv files containing the mean and standard deviation of the features of the clusters. The
    hard-coded features currently consist of positions, velocities, density, and internal energy. The means of the means
    and standard deviations are also calculated and can be used as offsets and scales to normalize the data.

    Args:
        magneticum_data_dir: (string) Directory containing the Magneticum data
        bahamas_data_dir: (string) Directory containing the BAHAMAS data
        magneticum_snapshots: (list of strings) Magneticum snapshots for which to calculate the means and standard deviations
        bahamas_snapshots: (list of strings) BAHAMAS snapshots for which to calculate the means and standard deviations

    """
    snapshots, snapshot_names = snapshots_and_names(magneticum_data_dir,
                                                    bahamas_data_dir,
                                                    magneticum_snapshots,
                                                    bahamas_snapshots)

    columns = ['particles',
               'mean x', 'std x',
               'mean y', 'std y',
               'mean z', 'std z',
               'mean vx', 'std vx',
               'mean vy', 'std vy',
               'mean vz', 'std vz',
               'mean rho', 'std rho',
               'mean U', 'std U']
    mean_columns = [' ',
                    'offset x', 'scale x',
                    'offset y', 'scale y',
                    'offset z', 'scale z',
                    'offset vx', 'scale vx',
                    'offset vy', 'scale vy',
                    'offset vz', 'scale vz',
                    'offset rho', 'scale rho',
                    'offset U', 'scale U']

    for idx, snapshot in enumerate(snapshots):
        if len(snapshot) == 0:
            continue
        else:
            print(f'Making csv file for snapshot : {snapshot_names[idx]}')
            print(f'Number of clusters : {len(snapshot)}')
            mean_and_std_arr = []

            for cluster in snapshot:
                positions, velocities, rho, u = extract_data(cluster, snapshot_names[idx])

                mean_and_std = [len(positions),
                                np.mean(positions[:, 0]),
                                np.std(positions[:, 0]),
                                np.mean(positions[:, 1]),
                                np.std(positions[:, 1]),
                                np.mean(positions[:, 2]),
                                np.std(positions[:, 2]),
                                np.mean(velocities[:, 0]),
                                np.std(velocities[:, 0]),
                                np.mean(velocities[:, 1]),
                                np.std(velocities[:, 1]),
                                np.mean(velocities[:, 2]),
                                np.std(velocities[:, 2]),
                                np.mean(np.log(rho)),
                                np.std(np.log(rho)),
                                np.mean(np.log(u)),
                                np.std(np.log(u))]
                mean_and_std_arr.append(mean_and_std)

            mean_and_std_arr = np.array(sorted(mean_and_std_arr, key=lambda row: row[0])[::-1])

            offsets_and_scales = np.array([np.mean(arr) for arr in np.array(mean_and_std_arr).T[1:]])
            offsets_and_scales = np.concatenate((np.array([0]), offsets_and_scales))

            with open(snapshot_names[idx] + '.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                writer.writerow(mean_columns)
                writer.writerow(offsets_and_scales)
                writer.writerow(columns)
                writer.writerows(mean_and_std_arr)


def read_data_csv():
    """

    Print the offsets and scales saved in snapshot csv files.

    """
    csv_files = glob.glob(os.getcwd() + '/*.csv')
    for file in csv_files:
        if os.path.basename(file)[0:3] == 'AGN':
            sim = 'Bahamas'
        else:
            sim = 'Magneticum'

        with open(file, 'r') as f:
            reader = csv.reader(f)
            columns = next(reader)[1:]
            offsets_and_scales = next(reader)[1:]
            print(f'Snapshot : {file.split("/")[-1].split(".")[0]}')
            print(f'Simulation : {sim}')
            print(f'Clusters : {len(list(reader)) - 1}')
            print(' ')
            for column, value in zip(columns, offsets_and_scales):
                if 'rho' in column or 'U' in column:
                    log = '(log)'
                else:
                    log = ''
                print(f'{column} : {value} {log}')
            print(' ')


if __name__ == '__main__':

    # Whether or not this code is running on ALICE
    alice = True

    # Whether to make xray images of the clusters in magneticum snapshots
    create_magneticum_xrays = False
    number_of_magneticum_xray_images = 1

    magneticum_snap_dirs = ['snap_128', 'snap_132', 'snap_136']
    # magneticum_snap_dirs = ['snap_128', 'snap_132', 'snap_136']

    # Whether to make hdf5 files that contain the particle data of individual FoF groups (clusters) in the BAHAMAS
    # snapshots. Specify the number of clusters for which to make hdf5 files. It will find the largest clusters
    # starting from the 'starting cluster' rank.
    create_bahamas_clusters = False
    number_of_clusters = 105
    starting_cluster = 95

    # Whether to make xrays images of the clusters in the BAHAMAS snapshots
    create_bahamas_xrays = False
    number_of_bahamas_xray_images = 1

    bahamas_snap_dirs = []
    # bahamas_snap_dirs = ['AGN_TUNED_nu0_L100N256_WMAP9', 'AGN_TUNED_nu0_L400N1024_WMAP9']

    # Whether to find and/or print the offsets and scales of the particle data
    calculate_offsets_and_scales = True
    print_offsets_and_scales = True

    # Whether to combine the particle data and image data in example directories as data.npz files
    combine = False

    # Whether to encode the combined data in tfrecord files and whether to normalize first.
    tfrec = False

    if alice:
        my_magneticum_data_dir = '/home/s2675544/data/Magneticum/Box2_hr'
        my_bahamas_data_dir = '/home/s2675544/data/Bahamas'
    else:
        my_magneticum_data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Magneticum/Box2_hr'
        my_bahamas_data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Bahamas'

    if create_magneticum_xrays:
        for snap_file in magneticum_snap_dirs:
            make_magneticum_fits(data_dir=my_magneticum_data_dir,
                                 snap_file=snap_file,
                                 number_of_xray_images=number_of_magneticum_xray_images)

    if create_bahamas_clusters or create_bahamas_xrays:
        # Determine which snapshot to use
        for snap_dir in bahamas_snap_dirs:
            data_sim_dir = os.path.join(os.path.dirname(my_bahamas_data_dir), snap_dir)
            save_sim_dir = os.path.join(my_bahamas_data_dir, snap_dir)

            if create_bahamas_clusters:
                make_bahamas_clusters(data_sim_dir,
                                      save_sim_dir,
                                      number_of_clusters=number_of_clusters,
                                      starting_cluster=starting_cluster)

            if create_bahamas_xrays:
                make_bahamas_fits(data_sim_dir,
                                  number_of_xray_images=number_of_bahamas_xray_images)

    if calculate_offsets_and_scales:
        make_data_csv(my_magneticum_data_dir,
                      my_bahamas_data_dir,
                      magneticum_snap_dirs,
                      bahamas_snap_dirs)

    if print_offsets_and_scales:
        read_data_csv()

    # Makes data.npz files in example dirs
    if combine:
        combine_data(magneticum_snap_dirs, bahamas_snap_dirs)

    if tfrec:
        for example_dir in magneticum_snap_dirs + bahamas_snap_dirs:
            with open(example_dir + '.csv', 'r') as csv_f:
                csv_reader = csv.reader(csv_f)
                for i in range(2):
                    means_and_std_values = next(csv_reader)[1:]
                _offsets = np.array(means_and_std_values[0::2])
                _scales = np.array(means_and_std_values[1::2])
            tfrecord_dir = example_dir + '_tfrecords'
            if not os.path.isdir(tfrecord_dir):
                tfrecords = generate_data(glob.glob(os.path.join(example_dir, 'example_*')),
                                          tfrecord_dir, offsets=_offsets, scales=_scales)
            else:
                tfrecords = glob.glob(tfrecord_dir + '/*')

            # Decode the tfrecord files again into the (graph tuple, image, index) dataset
            # This is also used in main.py to retrieve the datasets for a neural network
            dataset = tf.data.TFRecordDataset(tfrecords).map(
                lambda record_bytes: decode_examples(record_bytes, edge_shape=[2], node_shape=[8]))

            for (graph, image, example_idx) in iter(dataset):
                print(graph.nodes.shape, image.shape, example_idx)

            # SUMMARY
            # combine() + numpy arrays (pos, props, img) --> data.npz

            # generate_data(
            # _get_data() + data.npz  --> numpy arrays (pos, props, img)

            # generate_example_nn() + numpy arrays (pos, props, img) --> graph tuple, image and index (graph, img, idx)

            # save_examples() + graph tuple, image and index (graph, img, idx) --> tfrecord
            # )

            # decode_examples() + tfrecord --> graph tuple, image and index (graph, img, idx)

            # example index is an integer as a tensorflow tensor
