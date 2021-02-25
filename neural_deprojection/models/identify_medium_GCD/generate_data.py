import os
import glob
import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs, networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from networkx.drawing import draw
from tqdm import tqdm
from scipy.spatial.ckdtree import cKDTree
import pylab as plt
import yt
from astropy.io import fits
import h5py
import soxs
import gadget as g
import pyxsim
import mayavi
from mayavi import mlab

from multiprocessing import Pool, Lock

mp_lock = Lock()


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


# Finds the center of the gas particles in the snapshot by taking the average of the position extrema
def find_center(dataset, position, offset=(0., 0., 0.)):
    max_pos = np.max(position.T, axis=1)
    min_pos = np.min(position.T, axis=1)

    center = 0.5 * (max_pos + min_pos)
    box_size = max_pos - min_pos

    split_cluster = False

    for coord, side in enumerate(box_size):
        # print(side)
        # print(dataset.domain_width[coord])
        if side > 0.5 * dataset.domain_width[coord]:
            split_cluster = True

    return center + np.array(offset), box_size, split_cluster


@tf.function
def downsample(image):
    filter = tf.ones((2, 2, 1, 1)) * 0.25
    return tf.nn.conv2d(image[None, :, :, None],
                        filters=filter, strides=2,
                        padding='SAME')[0, :, :, 0]


def generate_example_random_choice(positions, properties, xray_image, hot_gas_positions, number_of_virtual_nodes=1000, k=26, plot=True):
    idx_list = np.arange(len(positions))
    virtual_node_positions = positions[np.random.choice(idx_list, number_of_virtual_nodes, replace=False)]

    kdtree = cKDTree(virtual_node_positions)
    dist, indices = kdtree.query(positions)

    virtual_properties = np.zeros((len(np.bincount(indices)), len(properties[0])))

    mean_sum = [lambda x: np.bincount(indices, weights=x) / np.maximum(1., np.bincount(indices)),       # mean
                lambda x: np.bincount(indices, weights=x)]              # sum

    mean_sum_enc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for p, enc in zip(np.arange(properties.shape[1]), mean_sum_enc):
        virtual_properties[:, p] = mean_sum[enc](properties[:, p])
        virtual_positions = virtual_properties[:, :3]

    # Directed graph
    graph = nx.DiGraph()

    # Create cKDTree class to find the nearest neighbours of the positions
    kdtree = cKDTree(virtual_properties)
    # idx has shape (positions, k+1) and contains for every position the indices of the nearest neighbours
    dist, idx = kdtree.query(virtual_properties, k=k+1)
    # downscale resolution

    # The index of the first nearest neighbour is the position itself, so we discard that one
    receivers = idx[:, 1:]  # N,k
    senders = np.arange(virtual_properties.shape[0])  # Just a range from 0 to the number of positions
    senders = np.tile(senders[:, None], [1, k])  # N,k
    # senders looks like (for 4 positions and 3 nn's)
    # [[0  0  0]
    #  [1  1  1]
    #  [2  2  2]
    #  [3  3  3]]

    # Every position has k connections and every connection has a sender and a receiver
    # The indices of receivers and senders correspond to each other (so receiver[32] belongs to sender[32])
    # The value of indices in senders and receivers correspond to the index they have in the positions array.
    # (so if sender[32] = 6, then that sender has coordinates positions[6])
    receivers = receivers.flatten()  # shape is (len(positions) * k,)
    senders = senders.flatten()  # shape is (len(positions) * k,)

    n_nodes = virtual_properties.shape[0] # number of nodes is the number of positions

    box_size = (np.min(virtual_positions), np.max(virtual_positions))

    if plot:
        pos = dict()  # for plotting node positions.
        edgelist = []

        # Now put the data in the directed graph: first the nodes with their positions and properties
        # pos just takes the x and y coordinates of the position so a 2D plot can be made
        for node, feature, position in zip(np.arange(n_nodes), virtual_properties, virtual_positions):
            graph.add_node(node, features=feature)
            pos[node] = (position[:2] - box_size[0])/(box_size[1] - box_size[0])

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

        dens_list = []

        for n in list(graph.nodes.data('features')):
            dens_list.append(n[1][6])

        print('Plotting graph...')
        fig, ax = plt.subplots(figsize=(8, 8))
        draw(graph, ax=ax, pos=pos,
             edge_color='red', node_size=10, width=0.1, arrowstyle='-',
             node_color=dens_list, cmap='viridis')
        plt.savefig('/home/s2675544/data/images/graph.png')
        plt.show()

        print('Plotting image...')
        fig, ax = plt.subplots(figsize=(8, 8))
        plot = ax.imshow(xray_image)
        fig.colorbar(plot, ax=ax)
        plt.savefig('/home/s2675544/data/images/xray.png')
        plt.show()

        # print('Plotting 3D image...')
        # mayavi_virtual_positions = virtual_positions / 1e27
        # mlab.points3d(mayavi_virtual_positions[:, 0],
        #               mayavi_virtual_positions[:, 1],
        #               mayavi_virtual_positions[:, 2],
        #               dens_list,
        #               resolution=8,
        #               scale_factor=0.0005,
        #               scale_mode='none',
        #               colormap='viridis')
        # for u, v in zip(senders, receivers):
        #     mlab.plot3d([mayavi_virtual_positions[u][0], mayavi_virtual_positions[v][0]],
        #                 [mayavi_virtual_positions[u][1], mayavi_virtual_positions[v][1]],
        #                 [mayavi_virtual_positions[u][2], mayavi_virtual_positions[v][2]],
        #                 tube_radius=None,
        #                 tube_sides=3,
        #                 opacity=0.1)
        # mlab.show()

        # Plotting
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        print('Plotting multiplot...')
        draw(graph, ax=ax[0, 0], pos=pos,
             edge_color='red', node_size=10, width=0.1, arrowstyle='-',
             node_color=dens_list, cmap='viridis')
        # draw(graph, ax=ax[0, 0], pos=pos, node_color='blue', edge_color='red', node_size=10, width=0.1)
        ax[0, 1].scatter(positions[:, 0], positions[:, 1], s=0.0005)
        ax[1, 0].scatter(hot_gas_positions[:, 0], hot_gas_positions[:, 1], s=0.0005)
        ax[0, 1].set_title('Gas particles')
        ax[1, 0].set_title('Hot gas particles (T > 10^5.3)')
        xray_plot = ax[1, 1].imshow(xray_image)
        fig.colorbar(xray_plot, ax=ax[1, 1])
        print('Multiplot done, showing...')
        plt.savefig('/home/s2675544/data/images/multiplot.png')
        plt.show()


    else:
        for node, feature, position in zip(np.arange(n_nodes), virtual_properties, virtual_positions):
                graph.add_node(node, features=feature)

        for u, v in zip(senders, receivers):
                graph.add_edge(u, v, features=np.array([1., 0.]))
                graph.add_edge(v, u, features=np.array([1., 0.]))

    # Global dummy variable, this needs to be defined in order to turn the graph into a graph tuple,
    # see networkxs_to_graphs_tuple documentation
    graph.graph["features"] = np.array([0.])

    # Important step: return the graph, which is a networkx class, as a graphs tuple!
    # positions.shape[1] = 3, properties.shape[1] = the number of features,
    # so node_shape_hint tells the function the number of length of the attribute for every node
    # edge_shape_hint: the edges, at the moment, have a dummy attribute of size two
    return networkxs_to_graphs_tuple([graph],
                                     node_shape_hint=[virtual_properties.shape[1]],
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
                  examples_per_file=32, num_examples=1, exp_time = None, prefix='train'):
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
        if exp_time is not None:
            exp_time_str = f'{exp_time}ks_'
        else:
            exp_time_str = ''
        file = os.path.join(save_dir, 'train_' + exp_time_str + '{:04d}.tfrecords'.format(file_idx))
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

    image = tf.where(image <= 0., tf.exp(-5.), image)

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

###
# specific to project

def generate_data(cluster_dirs, save_dir, units, number_of_projections=26 ,exp_time=1000., number_of_virtual_nodes=1000):
    train_tfrecords = []

    # Parameters for making the xray images
    exp_time = (exp_time, "ks")  # exposure time
    area = (2000.0, "cm**2")  # collecting area
    min_E = 0.05  # Minimum energy of photons in keV
    max_E = 11.0  # Maximum energy of photons in keV
    Z = 0.3  # Metallicity in units of solar metallicity
    kT_min = 0.05  # Minimum temperature to solve emission for
    n_chan = 1000  # Number of channels in the spectrum
    nH = 0.04  # The foreground column density in units of 10^22 cm^{-2}. Only used if absorption is applied.
    radius = (2.0, "Mpc")  # Radius of the sphere which captures photons
    sky_center = [45., 30.]  # Ra and dec coordinates of the cluster (which are currently dummy values)

    def data_generator():
        for cluster_idx, cluster in enumerate(cluster_dirs):

            if os.path.basename(snap_dir)[0:3] == 'AGN':
                with h5py.File(os.path.join(cluster, 'cluster_*'), 'r') as ds:
                    positions = np.array(ds['PartType0']['Coordinates'])
                    velocities = np.array(ds['PartType0']['Velocity'])
                    rho = np.array(ds['PartType0']['Density'])
                    u = np.array(ds['PartType0']['InternalEnergy'])
                    mass = np.array(ds['PartType0']['Mass'])
                    smooth = np.array(ds['PartType0']['SmoothingLength'])

                c = centers(int(os.path.basename(cluster)[-3:]))
                # Create a sphere around the center of the snapshot, which captures the photons
                sp = ds.sphere(c, radius)
                redshift = 0.05
                temp_fits_dir = 'bah_temp_fits_files'
            else:
                if os.path.basename(snap_dir) == 'snap_026':
                    # Clusters downloaded before november 2020 have a different field order.
                    my_field_def = (
                        "Coordinates",
                        "Velocities",
                        "Mass",
                        "ParticleIDs",
                        ("InternalEnergy", "Gas"),
                        ("Density", "Gas"),
                        ("SmoothingLength", "Gas"),
                    )
                    ds = yt.load(os.path.join(cluster, os.path.basename(snap_dir)),
                                 long_ids=True,
                                 field_spec=my_field_def)
                else:
                    ds = yt.load(os.path.join(cluster, os.path.basename(snap_dir)), long_ids=True)
                ad = ds.all_data()

                positions = ad['Gas', 'Coordinates'].in_cgs().d
                velocities = ad['Gas', 'Velocities'].in_cgs().d
                rho = ad['Gas', 'Density'].in_cgs().d
                u = ad['Gas', 'InternalEnergy'].in_cgs().d
                mass = ad['Gas', 'Mass'].in_cgs().d
                smooth = ad['Gas', 'SmoothingLength'].in_cgs().d

                # This is a filter which creates a new particle type (in memory), which
                # makes a cut on gas temperature to only consider gas that will really be
                # X-ray emitting
                def hot_gas(pfilter, data):
                    temp = data[pfilter.filtered_type, "temperature"]
                    return temp > 10 ** 5.3

                yt.add_particle_filter("hot_gas", function=hot_gas,
                                       filtered_type='gas', requires=["temperature"])

                ds.add_particle_filter("hot_gas")

                c, box_size, split = find_center(ds, ad['Gas', 'Coordinates'].d, [0., 0., 0.])
                print(box_size)
                print(f'Split cluster : {split}')

                if split:
                    continue
                # Create a sphere around the center of the snapshot, which captures the photons
                sp = ds.sphere(c, radius)
                redshift = ds.current_redshift
                # temp_fits_dir = 'mag_temp_fits_files'
                temp_fits_dir = ''

            p_t = positions.T
            v_t = velocities.T
            properties = np.stack((p_t[0], p_t[1], p_t[2], v_t[0], v_t[1], v_t[2], rho, u, mass, smooth), axis=1)

            print(' ')
            print(f'File : {cluster}')
            print(f'Cluster center : {c}')
            # print(f'Cluster box size : {box_size}')
            print(' ')

            # Set a minimum temperature to leave out that shouldn't be X-ray emitting,
            # set metallicity to 0.3 Zsolar (should maybe fix later)
            # The source model determines the distribution of photons that are emitted
            source_model = pyxsim.ThermalSourceModel("apec", min_E, max_E, n_chan, Zmet=Z, kT_min=kT_min)

            # Create the photonlist
            photons = pyxsim.PhotonList.from_data_source(sp, redshift, area, exp_time,
                                                         source_model)

            for projection_idx in tqdm(np.arange(number_of_projections)):

                print('\n\n\n')
                print(f'Cluster file: {cluster}')
                print(f'Cluster : {cluster_idx} / {len(cluster_dirs)}')
                print(f'Projection : {projection_idx} / {number_of_projections}')
                print('\n\n')

                V = np.eye(3)
                rot_mat = _random_special_ortho_matrix(3)
                # Vprime = rot_mat @ V
                Vprime = rot_mat.T @ V

                east_vector = Vprime[:, 0]
                north_vector = Vprime[:, 1]
                viewing_vec = Vprime[:, 2]

                xyz = properties[:, :3]  # n 3
                velocity_xyz = properties[:, 3:6]  # n 3
                xyz = (rot_mat @ xyz.T).T
                velocity_xyz = (rot_mat @ velocity_xyz.T).T
                # xyz = np.einsum('ap,np->na', rot_mat, xyz)  # n 3
                # velocity_xyz = np.einsum('ap,np->na', rot_mat, velocity_xyz)  # n 3
                hot_gas_pos = ad['hot_gas', 'position'].in_cgs().d
                hot_gas_pos = (rot_mat @ hot_gas_pos.T).T
                # hot_gas_pos = np.einsum('ap,np->na', rot_mat, hot_gas_pos)  # n 3

                _properties = properties.copy()  # n f
                _properties[:, :3] = xyz  # n f
                _properties[:, 3:6] = velocity_xyz  # n f
                _properties[:, 6] = _properties[:, 6]  # n f
                _positions = xyz  # n 3

                _properties[:, 0:3] = _properties[:, 0:3] - np.mean(_properties[:, 0:3], axis=0) / units[0:3]
                _properties[:, 3:6] = _properties[:, 3:6] / units[3:6]
                _properties[:, 6:] = np.log10(_properties[:, 6:] / units[6:])

                # Finds the events along a certain line of sight
                events_z = photons.project_photons(viewing_vec, sky_center, absorb_model="tbabs", nH=nH,
                                                   north_vector=north_vector)

                events_z.write_simput_file(os.path.join(temp_fits_dir, 'magneticum'), overwrite=True)

                # Determine which events get detected by the AcisI intstrument of Chandra
                soxs.instrument_simulator(os.path.join(temp_fits_dir, 'magneticum_simput.fits'),
                                          os.path.join(temp_fits_dir, 'magneticum_evt.fits'),
                                          exp_time,
                                          "chandra_acisi_cy0",
                                          sky_center,
                                          overwrite=True,
                                          ptsrc_bkgnd=False,
                                          foreground=False,
                                          instr_bkgnd=False)

                soxs.write_image(os.path.join(temp_fits_dir, 'magneticum_evt.fits'),
                                 os.path.join(temp_fits_dir, 'magneticum_img.fits'),
                                 emin=min_E,
                                 emax=max_E,
                                 overwrite=True)

                with fits.open(os.path.join(temp_fits_dir, 'magneticum_img.fits')) as hdu:
                    xray_image = np.array(hdu[0].data, dtype='float32')[1440:3440, 1440:3440]
                xray_image = downsample(xray_image).numpy()[:, :, None]
                xray_image = np.log10(np.where(xray_image < 1e-5, 1e-5, xray_image))

                off_axis_image = yt.off_axis_projection(data_source=ds,
                                                        center=c,
                                                        normal_vector=viewing_vec,
                                                        width=0.85 * box_size,
                                                        item='Density',
                                                        resolution=[800, 800],
                                                        north_vector=east_vector)
                # off_axis_image = np.log10(np.where(off_axis_image < 1e-5, 1e-5, off_axis_image))
                yt.write_image(np.log10(off_axis_image), '/home/s2675544/data/images/off_axis_proj.png')

                # For imshow the image is flipped
                plt_xray_image = xray_image[:, :, 0][::-1, :]

                # Create a graph with the positions and properties
                graph = generate_example_random_choice(_positions, _properties, plt_xray_image,
                                                       hot_gas_positions=hot_gas_pos,
                                                       number_of_virtual_nodes=number_of_virtual_nodes,
                                                       plot=True)

                # This function is a generator, which has the advantage of not keeping used and upcoming data in memory.
                yield (graph, xray_image, number_of_projections * cluster_idx + projection_idx)
    # Save the data as tfrecords and return the filenames of the tfrecords
    train_tfrecords.append(save_examples(data_generator(),
                                         save_dir,
                                         examples_per_file=number_of_projections,
                                         num_examples=number_of_projections * len(cluster_dirs),
                                         exp_time=int(exp_time[0]),
                                         prefix='train'))
    return train_tfrecords

def save_examples_multiprocessing(generator, save_dir=None,
                       examples_per_file=32, num_examples=1, exp_time=None, prefix='train'):
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
        if exp_time is not None:
            exp_time_str = f'{int(exp_time)}ks_'
        else:
            exp_time_str = ''

        mp_lock.acquire()  # make sure no duplicate files are made / replaced
        tf_files = glob.glob(os.path.join(save_dir, 'train_*'))
        file_idx = len(tf_files)
        mp_lock.release()

        file = os.path.join(save_dir, 'train_' + exp_time_str + '{:04d}.tfrecords'.format(file_idx))
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

def generate_data_multiprocessing(cluster_and_idx):
    cluster_idx = cluster_and_idx[0]
    cluster = cluster_and_idx[1]

    # Parameters for making the xray images
    exp_time = (exposure_time, "ks")  # exposure time
    area = (2000.0, "cm**2")  # collecting area
    min_E = 0.05  # Minimum energy of photons in keV
    max_E = 11.0  # Maximum energy of photons in keV
    Z = 0.3  # Metallicity in units of solar metallicity
    kT_min = 0.05  # Minimum temperature to solve emission for
    n_chan = 1000  # Number of channels in the spectrum
    nH = 0.04  # The foreground column density in units of 10^22 cm^{-2}. Only used if absorption is applied.
    radius = (2.0, "Mpc")  # Radius of the sphere which captures photons
    sky_center = [45., 30.]  # Ra and dec coordinates of the cluster (which are currently dummy values)

    if os.path.basename(snap_dir)[0:3] == 'AGN':
        with h5py.File(os.path.join(cluster, 'cluster_*'), 'r') as ds:
            positions = np.array(ds['PartType0']['Coordinates'])
            velocities = np.array(ds['PartType0']['Velocity'])
            rho = np.array(ds['PartType0']['Density'])
            u = np.array(ds['PartType0']['InternalEnergy'])
            mass = np.array(ds['PartType0']['Mass'])
            smooth = np.array(ds['PartType0']['SmoothingLength'])

        c = centers(int(os.path.basename(cluster)[-3:]))
        # Create a sphere around the center of the snapshot, which captures the photons
        sp = ds.sphere(c, radius)
        redshift = 0.05
        temp_fits_dir = 'bah_temp_fits_files'
    else:
        # This is a filter which creates a new particle type (in memory), which
        # makes a cut on gas temperature to only consider gas that will really be
        # X-ray emitting
        def hot_gas(pfilter, data):
            temp = data[pfilter.filtered_type, "temperature"]
            return temp > 10 ** 5.3

        yt.add_particle_filter("hot_gas", function=hot_gas,
                               filtered_type='gas', requires=["temperature"])

        ds = yt.load(os.path.join(cluster, os.path.basename(snap_dir)), long_ids=True)
        ad = ds.all_data()

        ds.add_particle_filter("hot_gas")

        positions = ad['Gas', 'Coordinates'].in_cgs().d
        velocities = ad['Gas', 'Velocities'].in_cgs().d
        rho = ad['Gas', 'Density'].in_cgs().d
        u = ad['Gas', 'InternalEnergy'].in_cgs().d
        mass = ad['Gas', 'Mass'].in_cgs().d
        smooth = ad['Gas', 'SmoothingLength'].in_cgs().d

        c, box_size, split = find_center(ds, ad['Gas', 'Coordinates'].d, [0., 0., 0.])
        print(box_size)
        print(f'Split cluster : {split}')

        if split:
            return
        # Create a sphere around the center of the snapshot, which captures the photons
        sp = ds.sphere(c, radius)
        redshift = ds.current_redshift
        # temp_fits_dir = 'mag_temp_fits_files'
        temp_fits_dir = ''

    p_t = positions.T
    v_t = velocities.T
    properties = np.stack((p_t[0], p_t[1], p_t[2], v_t[0], v_t[1], v_t[2], rho, u, mass, smooth), axis=1)

    print(' ')
    print(f'File : {cluster}')
    print(f'Cluster center : {c}')
    # print(f'Cluster box size : {box_size}')
    print(' ')

    # Set a minimum temperature to leave out that shouldn't be X-ray emitting,
    # set metallicity to 0.3 Zsolar (should maybe fix later)
    # The source model determines the distribution of photons that are emitted
    source_model = pyxsim.ThermalSourceModel("apec", min_E, max_E, n_chan, Zmet=Z, kT_min=kT_min)

    # Create the photonlist
    photons = pyxsim.PhotonList.from_data_source(sp, redshift, area, exp_time,
                                                 source_model)

    def data_generator():
        for projection_idx in tqdm(np.arange(number_of_projections)):
            print('\n\n\n')
            print(f'Cluster file: {cluster}')
            print(f'Cluster : {cluster_idx} / {len(cluster_dirs)}')
            print(f'Projection : {projection_idx} / {number_of_projections}')
            print('\n\n')

            V = np.eye(3)
            rot_mat = _random_special_ortho_matrix(3)
            Vprime = rot_mat.T @ V

            north_vector = Vprime[:, 1]
            viewing_vec = Vprime[:, 2]

            xyz = properties[:, :3]  # n 3
            velocity_xyz = properties[:, 3:6]  # n 3
            xyz = (rot_mat @ xyz.T).T
            velocity_xyz = (rot_mat @ velocity_xyz.T).T
            hot_gas_pos = ad['hot_gas', 'position'].in_cgs().d
            hot_gas_pos = (rot_mat @ hot_gas_pos.T).T

            _properties = properties.copy()  # n f
            _properties[:, :3] = xyz  # n f
            _properties[:, 3:6] = velocity_xyz  # n f
            _properties[:, 6] = _properties[:, 6]  # n f
            _positions = xyz  # n 3

            _properties[:, 0:3] = _properties[:, 0:3] - np.mean(_properties[:, 0:3], axis=0) / units[0:3]
            _properties[:, 3:6] = _properties[:, 3:6] / units[3:6]
            _properties[:, 6:] = np.log10(_properties[:, 6:] / units[6:])

            # Finds the events along a certain line of sight
            events_z = photons.project_photons(viewing_vec, sky_center, absorb_model="tbabs", nH=nH,
                                               north_vector=north_vector)

            events_z.write_simput_file(os.path.join(temp_fits_dir, 'magneticum'), overwrite=True)

            # Determine which events get detected by the AcisI intstrument of Chandra
            soxs.instrument_simulator(os.path.join(temp_fits_dir, 'magneticum_simput.fits'),
                                      os.path.join(temp_fits_dir, 'magneticum_evt.fits'),
                                      exp_time,
                                      "chandra_acisi_cy0",
                                      sky_center,
                                      overwrite=True,
                                      ptsrc_bkgnd=False,
                                      foreground=False,
                                      instr_bkgnd=False)

            soxs.write_image(os.path.join(temp_fits_dir, 'magneticum_evt.fits'),
                             os.path.join(temp_fits_dir, 'magneticum_img.fits'),
                             emin=min_E,
                             emax=max_E,
                             overwrite=True)

            with fits.open(os.path.join(temp_fits_dir, 'magneticum_img.fits')) as hdu:
                xray_image = np.array(hdu[0].data, dtype='float32')[1440:3440, 1440:3440]
            xray_image = downsample(xray_image).numpy()[:, :, None]
            xray_image = np.log10(np.where(xray_image < 1e-5, 1e-5, xray_image))

            # yt.off_axis_projection()

            # For imshow the image is flipped
            plt_xray_image = xray_image[:, :, 0][::-1, :]

            # Create a graph with the positions and properties
            graph = generate_example_random_choice(_positions, _properties, plt_xray_image,
                                                   hot_gas_positions=hot_gas_pos,
                                                   number_of_virtual_nodes=my_number_of_virtual_nodes,
                                                   plot=True)

            # This function is a generator, which has the advantage of not keeping used and upcoming data in memory.
            yield (graph, xray_image, number_of_projections * cluster_idx + projection_idx)

    # Save the data as tfrecords and return the filenames of the tfrecords
    save_examples_multiprocessing(data_generator(),
                                  save_dir=tfrecord_dir,
                                  examples_per_file=number_of_projections,
                                  exp_time=exp_time[0],
                                  prefix='train')


if __name__ == '__main__':

    yt.funcs.mylog.setLevel(40)  # Suppresses yt status output.
    soxs.utils.soxsLogger.setLevel(40)  # Suppresses soxs status output.
    pyxsim.utils.pyxsimLogger.setLevel(40)  # Suppresses pyxsim status output.

    # Define the directories containing the data
    if os.getcwd().split('/')[2] == 's2675544':
        my_magneticum_data_dir = '/home/s2675544/data/Magneticum/Box2_hr'
        my_bahamas_data_dir = '/home/s2675544/data/Bahamas'
        my_examples_dir = '/home/s2675544/data/examples'
        my_tf_records_dir = '/home/s2675544/data/tf_records'
        print('Running on ALICE')
    else:
        my_magneticum_data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Magneticum/Box2_hr'
        my_bahamas_data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Bahamas'
        my_examples_dir = '/home/matthijs/data/examples'
        my_tf_records_dir = '/home/matthijs/data/tf_records'
        print('Running at home')

    # Determine which snapshots to use
    magneticum_snap_dirs = ['snap_132']
    # Possible Magneticum dirs ['snap_128', 'snap_132', 'snap_136']

    bahamas_snap_dirs = []
    # Possible Bahamas dirs : ['AGN_TUNED_nu0_L100N256_WMAP9', 'AGN_TUNED_nu0_L400N1024_WMAP9']

    magneticum_snap_dirs = [os.path.join(my_magneticum_data_dir, snap_dir) for snap_dir in magneticum_snap_dirs]
    bahamas_snap_dirs = [os.path.join(my_bahamas_data_dir, snap_dir) for snap_dir in bahamas_snap_dirs]

    yr = 3.15576e7  # in seconds
    pc = 3.085678e18  # in cm
    M_sun = 1.989e33  # in gram

    _units = np.array([1e6 * pc,
                       1e6 * pc,
                       1e6 * pc,
                       1e-4 * pc / yr,
                       1e-4 * pc / yr,
                       1e-4 * pc / yr,
                       1e-7 * M_sun / pc ** 3,
                       1e-7 * (pc / yr) ** 2,
                       1e8 * M_sun,
                       1e5 * pc])

    # Whether to use multiprocessing and to rewrite tfrecord directories (if safe is True, doesn't rewrite)
    _multiprocessing = False
    safe = False

    for snap_dir in magneticum_snap_dirs + bahamas_snap_dirs:
        print(f'Snapshot directory : {snap_dir}')
        if os.path.basename(snap_dir)[0:3] == 'AGN':
            cluster_dirs = glob.glob(os.path.join(snap_dir, '*'))

            # Define the centers of clusters as the center of potential of friends-of-friends groups
            # 'subh' stands for subhalos
            snapnum = 32
            gdata = g.Gadget(os.path.join(os.path.dirname(my_bahamas_data_dir), snap_dir), 'subh', snapnum,
                             sim='BAHAMAS')
            centers = gdata.read_var('FOF/GroupCentreOfPotential', verbose=False)
            # Convert to codelength by going from cm to Mpc and from Mpc to codelength
            centers /= gdata.cm_per_mpc * 1.42855  # maybe a factor 4 for the big simulation
        else:
            cluster_dirs = glob.glob(os.path.join(snap_dir, '*/*/*'))

        tfrecord_dir = os.path.join(my_tf_records_dir, os.path.basename(snap_dir) + '_tf_records')
        print(f'Tensorflow records will be saved in : {tfrecord_dir}')
        print(f'Number of clusters : {len(cluster_dirs)}')

        if _multiprocessing:
            my_number_of_virtual_nodes = 100
            number_of_projections = 26
            exposure_time = 200.
            units = _units

            if os.path.isdir(tfrecord_dir) and safe:
                tfrecords = glob.glob(tfrecord_dir + '/*')
            else:
                pool = Pool(8)
                pool.map(generate_data_multiprocessing, enumerate(cluster_dirs))
        else:
            if os.path.isdir(tfrecord_dir) and safe:
                tfrecords = glob.glob(tfrecord_dir + '/*')
            else:
                tfrecords = generate_data(cluster_dirs=cluster_dirs,
                                          save_dir=tfrecord_dir,
                                          units=_units,
                                          number_of_projections=2,
                                          exp_time=1000.,
                                          number_of_virtual_nodes=1000)

