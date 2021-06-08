import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

import os
import glob
import yt
import h5py
import soxs
import pyxsim
import tensorflow as tf
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import neural_deprojection.models.identify_medium_GCD.gadget as g
from graph_nets.utils_np import networkxs_to_graphs_tuple, get_graph
from graph_nets.graphs import GraphsTuple
from neural_deprojection.models.identify_medium_GCD.model_utils import graph_tuple_to_feature, feature_to_graph_tuple, \
    decode_examples
from networkx.drawing import draw
from tqdm import tqdm
from scipy.spatial.ckdtree import cKDTree
from astropy.io import fits
from scipy import interpolate
from multiprocessing import Pool, Lock

if os.getcwd().split('/')[2] == 'matthijs':
    from mayavi import mlab
else:
    import matplotlib
    matplotlib.use('agg')

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
def check_split(dataset, position):
    max_pos = np.max(position.T, axis=1)
    min_pos = np.min(position.T, axis=1)

    box_size = max_pos - min_pos

    split_cluster = False

    for coord, side in enumerate(box_size):
        # print(side)
        # print(dataset.domain_width[coord].in_cgs())
        if side > 0.5 * dataset.domain_width[coord].in_cgs():
            split_cluster = True

    return box_size, split_cluster


def get_index(cluster_dirname):
    if 'AGN' in cluster_dirname:
        index = int(cluster_dirname.split('/')[-1][-3:])
    else:
        index = int(cluster_dirname.split('/')[-3])
    return index


def get_simulation_name(cluster):
    if cluster.split('/')[-3] == 'Bahamas':
        return 'Bahamas'
    else:
        return 'Magneticum'


def existing_clusters(record_bytes):
    """
    Determines which clusters are already made into tfrecords
    Args:
        record_bytes: raw bytes

    Returns: (cluster_idx, projection_idx)
    """
    parsed_example = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        dict(
            cluster_idx=tf.io.FixedLenFeature([], dtype=tf.string),
            projection_idx=tf.io.FixedLenFeature([], dtype=tf.string),
            image=tf.io.FixedLenFeature([], dtype=tf.string)
        )
    )
    cluster_idx = tf.io.parse_tensor(parsed_example['cluster_idx'], tf.int32)
    projection_idx = tf.io.parse_tensor(parsed_example['projection_idx'], tf.int32)
    image = tf.io.parse_tensor(parsed_example['image'], tf.float32)
    return (cluster_idx, projection_idx, image)


@tf.function
def downsample(image):
    filter = tf.ones((2, 2, 1, 1)) * 0.25
    return tf.nn.conv2d(image[None, :, :, None],
                        filters=filter, strides=2,
                        padding='SAME')[0, :, :, 0]


def generate_example_random_choice(properties,
                                   xray_image,
                                   hot_gas_positions,
                                   base_data_dir,
                                   center,
                                   vprime,
                                   dataset,
                                   sphere,
                                   number_of_virtual_nodes=1000,
                                   k=26,
                                   plot=0):
    idx_list = np.arange(len(properties))
    if len(properties) > number_of_virtual_nodes:
        virtual_node_positions = properties[:, :3][np.random.choice(idx_list, number_of_virtual_nodes, replace=False)]

        kdtree = cKDTree(virtual_node_positions)
        dist, indices = kdtree.query(properties[:, :3])

        # print('indices shape: ', len(indices), max(indices))

        virtual_properties = np.zeros((len(np.bincount(indices)), len(properties[0])))

        mean_sum = [lambda x: np.bincount(indices, weights=x) / np.maximum(1., np.bincount(indices)),  # mean
                    lambda x: np.bincount(indices, weights=x)]  # sum

        mean_sum_enc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for p, enc in zip(np.arange(properties.shape[1]), mean_sum_enc):
            # if plot > 0:
            #     print('p: ', p)
            #     print('properties shape: ', len(properties[:, p]), max(properties[:, p]))
            #     print('Maximum: ', np.maximum(1., np.bincount(indices)))
            #     print('properties shape: ', len(np.bincount(indices, weights=properties[:, p])), max(np.bincount(indices, weights=properties[:, p])))
            virtual_properties[:, p] = mean_sum[enc](properties[:, p])
            virtual_positions = virtual_properties[:, :3]
    else:
        virtual_positions = properties[:, :3]
        virtual_properties = properties

    # Create cKDTree class to find the nearest neighbours of the positions
    kdtree = cKDTree(virtual_properties)
    # idx has shape (positions, k+1) and contains for every position the indices of the nearest neighbours
    dist, idx = kdtree.query(virtual_properties, k=k + 1)
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

    receivers_bi_directional = np.concatenate((receivers, senders))
    senders_bi_directional = np.concatenate((senders, receivers))

    graph_nodes = tf.convert_to_tensor(virtual_properties, tf.float32)
    graph_nodes.set_shape([None, len(virtual_properties[0])])
    receivers = tf.convert_to_tensor(receivers_bi_directional, tf.int32)
    receivers.set_shape([None])
    senders = tf.convert_to_tensor(senders_bi_directional, tf.int32)
    senders.set_shape([None])
    n_node = tf.shape(graph_nodes)[0:1]
    n_edge = tf.shape(senders)[0:1]

    graph_data_dict = dict(nodes=graph_nodes,
                           edges=tf.zeros((n_edge[0], 1)),
                           globals=tf.zeros([1]),
                           receivers=receivers,
                           senders=senders,
                           n_node=n_node,
                           n_edge=n_edge)

    if plot > 0:
        graph_img_plotting(plotting=plot,
                           virtual_properties=virtual_properties,
                           senders=senders_bi_directional,
                           receivers=receivers_bi_directional,
                           xray_image=xray_image,
                           positions=properties[:, :3],
                           hot_gas_positions=hot_gas_positions,
                           base_data_dir=base_data_dir,
                           center=center,
                           vprime=vprime,
                           dataset=dataset,
                           sphere=sphere)

    return GraphsTuple(**graph_data_dict)


def graph_img_plotting(plotting,
                       virtual_properties,
                       senders,
                       receivers,
                       xray_image,
                       positions,
                       hot_gas_positions,
                       base_data_dir,
                       center,
                       vprime,
                       dataset,
                       sphere):
    max_pos = np.max(positions.T, axis=1)
    min_pos = np.min(positions.T, axis=1)
    box_size = max_pos - min_pos

    if plotting == 1:
        print('Plotting xray...')
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].scatter(center[0] + hot_gas_positions[:, 0], center[1] + hot_gas_positions[:, 1],
                      s=10 ** (2.5 - np.log10(positions.shape[0])))
        ax[0].set_xlim(center[0] - 0.5 * box_size[0], center[0] + 0.5 * box_size[0])
        ax[0].set_ylim(center[1] - 0.5 * box_size[1], center[1] + 0.5 * box_size[1])
        ax[0].set_title('Hot gas particles')
        ax[1].imshow(xray_image)
        # fig.colorbar(xray_plot, ax=ax[1])
        ax[1].set_title('Xray')
        ax[2].scatter(center[0] + positions.T[0], center[1] + positions.T[1],
                      s=10 ** (2.5 - np.log10(positions.shape[0])))
        ax[2].set_xlim(center[0] - 0.5 * box_size[0], center[0] + 0.5 * box_size[0])
        ax[2].set_ylim(center[1] - 0.5 * box_size[1], center[1] + 0.5 * box_size[1])
        ax[2].set_title('Gas particles')
        plt.savefig(os.path.join(base_data_dir, 'images/xray.png'))
        plt.show()

    if plotting > 1:
        graph = nx.OrderedMultiDiGraph()

        n_nodes = virtual_properties.shape[0]  # number of nodes is the number of positions

        virtual_box_size = (np.min(virtual_properties[:, :3]), np.max(virtual_properties[:, :3]))

        pos = dict()  # for plotting node positions.
        edgelist = []

        # Now put the data in the directed graph: first the nodes with their positions and properties
        # pos just takes the x and y coordinates of the position so a 2D plot can be made
        for node, feature, position in zip(np.arange(n_nodes), virtual_properties, virtual_properties[:, :3]):
            graph.add_node(node, features=feature)
            pos[node] = (position[:2] - virtual_box_size[0]) / (virtual_box_size[1] - virtual_box_size[0])

        # Next add the edges using the receivers and senders arrays we just created
        # Note that an edge is added for both directions
        # The features of the edge are dummy arrays at the moment
        # The edgelist is for the plotting
        # edges = np.stack([senders, receivers], axis=-1) + sibling_node_offset
        for u, v in zip(senders, receivers):
            graph.add_edge(u, v, features=np.array([1., 0.]))
            edgelist.append((u, v))

        print(f'Particle graph nodes : {graph.number_of_nodes()}')
        print(f'Particle graph edges : {graph.number_of_edges()}')

        dens_list = []

        for n in list(graph.nodes.data('features')):
            dens_list.append(n[1][6])

        if plotting == 2:
            # Plotting
            fig, ax = plt.subplots(2, 2, figsize=(12, 12))
            print('Plotting multiplot...')
            draw(graph, ax=ax[0, 0], pos=pos,
                 edge_color='red', node_size=10 ** (4 - np.log10(len(dens_list))), width=0.1, arrowstyle='-',
                 node_color=dens_list, cmap='viridis')
            # draw(graph, ax=ax[0, 0], pos=pos, node_color='blue', edge_color='red', node_size=10, width=0.1)
            ax[0, 1].scatter(center[0] + hot_gas_positions[:, 0], center[1] + hot_gas_positions[:, 1],
                             s=10 ** (2.5 - np.log10(positions.shape[0])))
            ax[0, 1].set_xlim(center[0] - 0.5 * box_size[0], center[0] + 0.5 * box_size[0])
            ax[0, 1].set_ylim(center[1] - 0.5 * box_size[1], center[1] + 0.5 * box_size[1])
            ax[0, 1].set_title('Hot gas particles')
            ax[1, 0].imshow(xray_image)
            # fig.colorbar(xray_plot, ax=ax[1])
            # ax[1].set_title('Xray')
            ax[1, 1].scatter(center[0] + positions.T[0], center[1] + positions.T[1],
                             s=10 ** (2.5 - np.log10(positions.shape[0])))
            ax[1, 1].set_xlim(center[0] - 0.5 * box_size[0], center[0] + 0.5 * box_size[0])
            ax[1, 1].set_ylim(center[1] - 0.5 * box_size[1], center[1] + 0.5 * box_size[1])
            ax[1, 1].set_title('Gas particles')
            print('Multiplot done, showing...')
            plt.savefig(os.path.join(base_data_dir, 'images/multiplot.png'))
            plt.show()

        if plotting == 3:
            interp_virtual_positions = virtual_properties[:, :3] / 1e4
            fig, ax = plt.subplots(figsize=(8, 8))
            _x = np.linspace(np.min(interp_virtual_positions[:, 0]), np.max(interp_virtual_positions[:, 0]), 300)
            _y = np.linspace(np.min(interp_virtual_positions[:, 1]), np.max(interp_virtual_positions[:, 1]), 300)
            _z = np.linspace(np.min(interp_virtual_positions[:, 2]), np.max(interp_virtual_positions[:, 2]), 300)
            x, y, z = np.meshgrid(_x, _y, _z, indexing='ij')

            interp = interpolate.griddata((interp_virtual_positions[:, 0],
                                           interp_virtual_positions[:, 1],
                                           interp_virtual_positions[:, 2]),
                                          dens_list,
                                          xi=(x, y, z), fill_value=0.0)

            im = np.mean(interp, axis=2).T[::-1, ]
            im = np.log10(np.where(im / np.max(im) < 1e-3, 1e-3, im / np.max(im)))
            ax.imshow(im)
            plt.savefig(os.path.join(base_data_dir, 'images/interp.png'))
            plt.show()

        if plotting == 4:
            print('Plotting off-axis projection...')
            east_vector = vprime[:, 0]
            north_vector = vprime[:, 1]
            viewing_vec = vprime[:, 2]
            fig, ax = plt.subplots(figsize=(8, 8))
            off_axis_image = yt.off_axis_projection(data_source=dataset,
                                                    center=sphere.center,
                                                    normal_vector=viewing_vec,
                                                    width=0.01 * dataset.domain_width,
                                                    item='Density',
                                                    resolution=[400, 400],
                                                    north_vector=east_vector)
            off_axis_image = np.log10(np.where(off_axis_image < 1e17, 1e17, off_axis_image))
            # off_axis_image = off_axis_image.to_ndarray() / np.max(off_axis_image.to_ndarray())
            # off_axis_image = np.log10(np.where(off_axis_image < 1e-5, 1e-5, off_axis_image))
            # print(f'Maximum : {np.max(off_axis_image)}')
            off_axis_image = ax.imshow(off_axis_image)
            fig.colorbar(off_axis_image, ax=ax)
            plt.savefig(os.path.join(base_data_dir, 'images/off_axis_proj.png'))
            plt.show()
            # yt.write_image(np.log10(off_axis_image), os.path.join(base_data_dir, 'images/off_axis_proj.png'))

        if plotting == 5:
            print('Plotting 3D image...')
            mlab.points3d(virtual_properties.T[0],
                          virtual_properties.T[1],
                          virtual_properties.T[2],
                          dens_list,
                          resolution=8,
                          scale_factor=0.15,
                          scale_mode='none',
                          colormap='viridis')
            for u, v in zip(senders, receivers):
                mlab.plot3d([virtual_properties[u][0], virtual_properties[v][0]],
                            [virtual_properties[u][1], virtual_properties[v][1]],
                            [virtual_properties[u][2], virtual_properties[v][2]],
                            tube_radius=None,
                            tube_sides=3,
                            opacity=0.1)
            mlab.show()


def save_examples(generator,
                  save_dir=None,
                  examples_per_file=32,
                  num_examples=1,
                  exp_time=None,
                  prefix='train'):
    """
    Saves a list of GraphTuples to tfrecords.

    Args:
        generator: generator (or list) of (GraphTuples, image).
            Generator is more efficient.
        save_dir: dir to save tfrecords in
        examples_per_file: int, max number examples per file
        num_examples: number of examples
        exp_time: exposure time (used in filename)

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
            exp_time_str = f'_{int(exp_time)}ks_'
        else:
            exp_time_str = ''

        mp_lock.acquire()  # make sure no duplicate files are made / replaced
        tf_files = glob.glob(os.path.join(save_dir, 'train_*'))
        file_idx = len(tf_files)
        indices = sorted([int(tf_file.split('.')[0][-4:]) for tf_file in tf_files])
        for idx, ind in enumerate(indices):
            if idx != ind:
                file_idx = idx
                break
        file = os.path.join(save_dir, prefix + exp_time_str + '{:04d}.tfrecords'.format(file_idx))
        files.append(file)
        mp_lock.release()

        # 'writer' can write to 'file'
        with tf.io.TFRecordWriter(file) as writer:
            for i in range(examples_per_file + 1):
                # Yield a dataset extracted by the generator
                try:
                    (graph, image, cluster_idx, projection_idx, vprime) = next(data_iterable)
                except StopIteration:
                    data_left = False
                    break
                # Write the graph, image and example_idx to the tfrecord file
                graph = get_graph(graph, 0)
                features = dict(
                    image=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(image, tf.float32)).numpy()])),
                    vprime=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(vprime, tf.float32)).numpy()])),
                    cluster_idx=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(cluster_idx, tf.int32)).numpy()])),
                    projection_idx=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(projection_idx, tf.int32)).numpy()])),
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


def load_data_bahamas(cluster, centers, base_data_dir):
    with h5py.File(os.path.join(cluster, os.path.basename(cluster) + '.hdf5'), 'r') as ds:
        positions = np.array(ds['PartType0']['Coordinates'])
        velocities = np.array(ds['PartType0']['Velocity'])
        rho = np.array(ds['PartType0']['Density'])
        u = np.array(ds['PartType0']['InternalEnergy'])
        mass = np.array(ds['PartType0']['Mass'])
        smooth = np.array(ds['PartType0']['SmoothingLength'])

    # For some reason the Bahamas snapshots are structured so that when you load one part of the snapshot,
    # you load the entire simulation box, so there is not a specific reason to choose the first element of filenames
    filenames = glob.glob(os.path.join(base_data_dir, cluster.split('/')[-2], 'data/snapshot_032/*.hdf5'))
    snap_file = filenames[0]
    ds = yt.load(snap_file)

    # Create a sphere around the center of the snapshot, which captures the photons
    c = centers[get_index(cluster)]

    # c[0] -= 1.0
    # c[1] += 1.0

    properties = np.stack((positions.T[0],
                           positions.T[1],
                           positions.T[2],
                           velocities.T[0],
                           velocities.T[1],
                           velocities.T[2],
                           rho,
                           u,
                           mass,
                           smooth), axis=1)

    return properties, c, ds


def load_data_magneticum(cluster, snap_dir):
    ds = yt.load(os.path.join(cluster, snap_dir), long_ids=True)
    ad = ds.all_data()

    positions = ad['Gas', 'Coordinates'].in_cgs().d
    velocities = ad['Gas', 'Velocities'].in_cgs().d
    rho = ad['Gas', 'Density'].in_cgs().d
    u = ad['Gas', 'InternalEnergy'].in_cgs().d
    mass = ad['Gas', 'Mass'].in_cgs().d
    smooth = ad['Gas', 'SmoothingLength'].in_cgs().d

    # Create a sphere around the center of the snapshot, which captures the photons
    c = np.mean(ds.all_data()['Gas', 'Coordinates'].d.T, axis=1)
    # c[0] += 500
    # c[1] -= 0

    properties = np.stack((positions.T[0],
                           positions.T[1],
                           positions.T[2],
                           velocities.T[0],
                           velocities.T[1],
                           velocities.T[2],
                           rho,
                           u,
                           mass,
                           smooth), axis=1)

    return properties, c, ds


def generate_data(cluster,
                  tfrecord_dir,
                  base_data_dir,
                  cluster_dirs,
                  snap_dir,
                  centers,
                  number_of_projections=26,
                  exp_time=1000.,
                  redshift=0.20,
                  number_of_virtual_nodes=1000,
                  number_of_neighbours=26,
                  plotting=0):
    cluster_idx = get_index(cluster)
    good_cluster = True
    print(f'\nStarting new cluster : {cluster_idx}')

    yr = 3.15576e7  # in seconds
    pc = 3.085678e18  # in cm
    Mpc = 1e6 * pc
    M_sun = 1.989e33  # in gram

    # Parameters for making the xray images
    exp_t = (exp_time, "ks")  # exposure time
    area = (1000.0, "cm**2")  # collecting area
    emin = 0.05  # Minimum energy of photons in keV
    emax = 11.0  # Maximum energy of photons in keV
    metallicty = 0.3  # Metallicity in units of solar metallicity
    kt_min = 0.05  # Minimum temperature to solve emission for
    n_chan = 1000  # Number of channels in the spectrum
    hydrogen_dens = 0.04  # The foreground column density in units of 10^22 cm^{-2}. Only used if absorption is applied.
    radius = (4.0, "Mpc")  # Radius of the sphere which captures photons
    sky_center = [0., 0.]  # Ra and dec coordinates of the cluster (which are currently dummy values)
    hot_gas_temp = 10 ** 5.4

    units = np.array([Mpc,
                      Mpc,
                      Mpc,
                      1e-4 * pc / yr,
                      1e-4 * pc / yr,
                      1e-4 * pc / yr,
                      1e-7 * M_sun / pc ** 3,
                      1e-7 * (pc / yr) ** 2,
                      1e8 * M_sun,
                      1e5 * pc])

    # Load in particle data and prepare for making an xray image.
    if get_simulation_name(cluster) == 'Bahamas':
        properties, c, ds = load_data_bahamas(cluster=cluster,
                                              centers=centers,
                                              base_data_dir=base_data_dir)
    else:
        properties, c, ds = load_data_magneticum(cluster=cluster, snap_dir=snap_dir)

    sp = ds.sphere(c, radius)

    _box_size, split = check_split(ds, properties[:, :3])
    print(f'\nBox size : {_box_size}')
    print(f'Cluster center : {sp.center}')
    print(f'Split cluster : {split}')

    if split:
        print(f'\nThe positions of the particles in cluster {cluster_idx} are '
              f'split by a periodic boundary and the easiest solution for this '
              f'is to leave the cluster out of the dataset.')
        good_cluster = False

    if properties.shape[0] < number_of_virtual_nodes:
        print(f'\nThe cluster contains {properties.shape[0]} particles '
              f'which is not enough to make {number_of_virtual_nodes} virtual nodes.')
        good_cluster = False

    # Set a minimum temperature to leave out that shouldn't be X-ray emitting,
    # set metallicity to 0.3 Zsolar (should maybe fix later)
    # The source model determines the distribution of photons that are emitted
    source_model = pyxsim.ThermalSourceModel(spectral_model="apec",
                                             emin=emin,
                                             emax=emax,
                                             nchan=n_chan,
                                             Zmet=metallicty,
                                             kT_min=kt_min)

    # Create the photonlist
    photons = pyxsim.PhotonList.from_data_source(data_source=sp,
                                                 redshift=redshift,
                                                 area=area,
                                                 exp_time=exp_t,
                                                 source_model=source_model)

    # Calculate the physical diameter of the image with : distance * fov = diameter
    chandra_acis_fov = 0.0049160  # in radians
    cutout_box_size = photons.parameters["fid_d_a"].d * chandra_acis_fov * Mpc

    number_of_photons = int(np.sum(photons["num_photons"]))
    if number_of_photons > 5e8:
        print(f'\nThe number of photons {number_of_photons} is too large and will take too long to process '
              f'so cluster {cluster_idx} is skipped.')
        good_cluster = False

    if plotting > 0:
        # This is a filter which creates a new particle type (in memory), which
        # makes a cut on gas temperature to only consider gas that will really be
        # X-ray emitting
        def hot_gas(pfilter, data):
            temp = data[pfilter.filtered_type, "temperature"]
            return temp > hot_gas_temp

        yt.add_particle_filter("hot_gas", function=hot_gas,
                               filtered_type='gas', requires=["temperature"])
        ds.add_particle_filter("hot_gas")

    def data_generator():
        for projection_idx in tqdm(np.arange(number_of_projections)):
            print(f'\n\nCluster file: {cluster}')
            print(f'Cluster index: {cluster_idx}')
            print(f'Clusters done (or in the making) : {len(glob.glob(os.path.join(tfrecord_dir, "*")))}')
            print(f'Projection : {projection_idx + 1} / {number_of_projections}\n')

            _properties = properties.copy()

            # Rotate variables
            rot_mat = _random_special_ortho_matrix(3)
            # rot_mat = np.eye(3)
            _properties[:, :3] = (rot_mat @ _properties[:, :3].T).T
            _properties[:, 3:6] = (rot_mat @ _properties[:, 3:6].T).T
            center = (rot_mat @ np.array(sp.center.in_cgs()).T).T

            # Cut out box in 3D space
            lower_lim = center - 0.5 * cutout_box_size * np.array([1, 1, 1])
            upper_lim = center + 0.5 * cutout_box_size * np.array([1, 1, 1])
            indices = np.where((_properties[:, 0:3] < lower_lim) | (_properties[:, 0:3] > upper_lim))[0]
            _properties = np.delete(_properties, indices, axis=0)

            # Scale the variables
            _properties[:, 0:3] = (_properties[:, 0:3] - center) / units[0:3]
            _properties[:, 3:6] = _properties[:, 3:6] / units[3:6]
            _properties[:, 6:] = np.log10(_properties[:, 6:] / units[6:])
            center /= units[0:3]

            print(f'Properties :', _properties[0])
            print('Properties shape: ', _properties.shape)

            if plotting > 0:
                hot_gas_pos = ds.all_data()['hot_gas', 'position'].in_cgs().d
                hot_gas_pos = (rot_mat @ hot_gas_pos.T).T
                hot_gas_pos = (hot_gas_pos / units[0:3]) - center
            else:
                hot_gas_pos = None

            v = np.eye(3)
            vprime = rot_mat.T @ v

            north_vector = vprime[:, 1]
            viewing_vec = vprime[:, 2]

            # Finds the events along a certain line of sight
            cluster_projection_identity = number_of_projections * cluster_idx + projection_idx
            events_z = photons.project_photons(viewing_vec, sky_center, absorb_model="tbabs", nH=hydrogen_dens,
                                               north_vector=north_vector)

            events_z.write_simput_file(f'snap_{cluster_projection_identity}', overwrite=True)

            # Determine which events get detected by the AcisI instrument of Chandra
            soxs.instrument_simulator(f'snap_{cluster_projection_identity}_simput.fits',
                                      f'snap_{cluster_projection_identity}_evt.fits',
                                      exp_t,
                                      "chandra_acisi_cy0",
                                      sky_center,
                                      overwrite=True,
                                      ptsrc_bkgnd=False,
                                      foreground=False,
                                      instr_bkgnd=False)

            soxs.write_image(f'snap_{cluster_projection_identity}_evt.fits',
                             f'snap_{cluster_projection_identity}_img.fits',
                             emin=emin,
                             emax=emax,
                             overwrite=True)

            with fits.open(f'snap_{cluster_projection_identity}_img.fits') as hdu:
                xray_image = np.array(hdu[0].data, dtype='float32')[1358:3406, 1329:3377]  # [2048,2048]

            temp_fits_files = glob.glob(os.path.join(os.getcwd(), f'snap_{cluster_projection_identity}_*.fits'))
            for file in temp_fits_files:
                print(f'Removing : {os.path.basename(file)}')
                os.remove(file)

            xray_image = downsample(xray_image).numpy()[:, :, None]
            xray_image = np.log10(np.where(xray_image < 1e-5, 1e-5, xray_image))

            # For imshow the image is flipped
            plt_xray_image = xray_image[:, :, 0][::-1, :]

            # Create a graph with the positions and properties
            graph = generate_example_random_choice(_properties, plt_xray_image,
                                                   hot_gas_positions=hot_gas_pos,
                                                   number_of_virtual_nodes=number_of_virtual_nodes,
                                                   k=number_of_neighbours,
                                                   plot=plotting,
                                                   base_data_dir=base_data_dir,
                                                   center=center,
                                                   vprime=vprime,
                                                   dataset=ds,
                                                   sphere=sp)

            # This function is a generator, which has the advantage of not keeping used and upcoming data in memory.
            yield (graph, xray_image, cluster_idx, projection_idx, vprime)

    if good_cluster:
        # Save the data as tfrecords and return the filenames of the tfrecords
        save_examples(data_generator(),
                      save_dir=tfrecord_dir,
                      examples_per_file=number_of_projections,
                      num_examples=number_of_projections * len(cluster_dirs),
                      exp_time=exp_t[0],
                      prefix='train')


def main(magneticum_snap_directories,
         bahamas_snap_directories,
         multi_processing=False,
         addition=False,
         check_clusters=False,
         number_of_virtual_nodes=500,
         number_of_projections=26,
         exposure_time=5000.,
         redshift=0.20,
         plotting=0,
         cores=16,
         number_of_neighbours=6,
         move_to_front=None):
    yt.funcs.mylog.setLevel(40)  # Suppresses yt status output.
    soxs.utils.soxsLogger.setLevel(40)  # Suppresses soxs status output.
    pyxsim.utils.pyxsimLogger.setLevel(40)  # Suppresses pyxsim status output.

    # Define the directories containing the data
    if os.getcwd().split('/')[2] == 's2675544':
        base_data_dir = '/home/s2675544/data'
        print('Running on ALICE')
    else:
        base_data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data'
        print('Running at home')

    my_magneticum_data_dir = os.path.join(base_data_dir, 'Magneticum/Box2_hr')
    my_bahamas_data_dir = os.path.join(base_data_dir, 'Bahamas')
    my_tf_records_dir = os.path.join(base_data_dir, 'tf_records')

    magneticum_snap_paths = [os.path.join(my_magneticum_data_dir, snap_dir) for snap_dir in magneticum_snap_directories]
    bahamas_snap_paths = [os.path.join(my_bahamas_data_dir, snap_dir) for snap_dir in bahamas_snap_directories]

    defect_clusters = {'snap_128': {'split': [109, 16, 72, 48],
                                    'photon_max': [53, 78],
                                    'too_small': []},
                       'snap_132': {'split': [75, 50, 110, 18],
                                    'photon_max': [8, 52, 55, 93, 139, 289],
                                    'too_small': []},
                       'snap_136': {'split': [75, 107, 52, 15],
                                    'photon_max': [96, 137, 51, 315, 216, 55, 102, 101, 20, 3],
                                    'too_small': []},
                       'AGN_TUNED_nu0_L100N256_WMAP9': {'split': [],
                                                        'photon_max': [3],
                                                        'too_small': [4, 10] + list(
                                                            set(np.arange(20, 200)) - {20, 21, 22, 28})},
                       'AGN_TUNED_nu0_L400N1024_WMAP9': {'split': [62, 89, 108, 125, 130, 191],
                                                         'photon_max': [],
                                                         'too_small': []}}

    for snap_idx, snap_path in enumerate(magneticum_snap_paths + bahamas_snap_paths):
        print(f'Snapshot path : {snap_path}')
        snap_dir = os.path.basename(snap_path)

        if snap_dir[0:3] == 'AGN':
            cluster_dirs = glob.glob(os.path.join(snap_path, '*'))

            snapnum = 32
            gdata = g.Gadget(os.path.join(base_data_dir, snap_dir), 'subh', snapnum, sim='BAHAMAS')

            subhalo_ids = [int(id) for id in gdata.read_var('FOF/FirstSubhaloID', verbose=False)]

            centers = gdata.read_var('Subhalo/CentreOfPotential', verbose=False)
            centers = centers[subhalo_ids[:-1]]
            # Convert to codelength by going from cm to Mpc and from Mpc to codelength
            centers /= gdata.cm_per_mpc / 0.7
        else:
            cluster_dirs = glob.glob(os.path.join(snap_path, '*/*/*'))
            centers = []

        tfrecord_dir = os.path.join(my_tf_records_dir, snap_dir + '_tf_records')

        print(f'Tensorflow records will be saved in : {tfrecord_dir}')
        print(f'Number of clusters : {len(cluster_dirs)}')

        bad_cluster_idx = defect_clusters[snap_dir]['split'] + \
                          defect_clusters[snap_dir]['too_small'] + \
                          defect_clusters[snap_dir]['photon_max']

        print(f'Number of viable clusters : {len(cluster_dirs) - len(bad_cluster_idx)}')

        if check_clusters:
            if os.path.isdir(tfrecord_dir):
                tfrecords = glob.glob(os.path.join(tfrecord_dir, '*'))

                if os.path.isdir(tfrecords[0]):
                    tfrecords = glob.glob(os.path.join(tfrecord_dir, 'train', '*')) + \
                                glob.glob(os.path.join(tfrecord_dir, 'test', '*'))

                processed_cluster_idxs = []

                # Why does the downsample function not run if this is executed AND multiprocessing is on????
                existing_cluster_datasets = tf.data.TFRecordDataset(tfrecords).map(
                    lambda record_bytes: existing_clusters(record_bytes))

                os.makedirs(os.path.join(base_data_dir, 'images', snap_dir), exist_ok=True)

                for cluster_idx, projection_idx, image in tqdm(iter(existing_cluster_datasets)):
                    processed_cluster_idxs.append(cluster_idx.numpy())
                    if projection_idx.numpy() < 3:
                        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                        ax.imshow(image.numpy()[:, :, 0])
                        plt.savefig(os.path.join(base_data_dir, 'images', snap_dir, 'xray_' +
                                                 '{:03d}'.format(cluster_idx.numpy()) + '_' +
                                                 '{:02d}'.format(projection_idx.numpy()) + '.png'))
                        plt.close(fig=fig)

                bad_cluster_dirs = []

                for cluster_dir in cluster_dirs:
                    if get_index(cluster_dir) in bad_cluster_idx:
                        bad_cluster_dirs.append(cluster_dir)

                for bad_cluster_dir in bad_cluster_dirs:
                    cluster_dirs.remove(bad_cluster_dir)

                for cluster_dir in sorted(cluster_dirs):
                    if processed_cluster_idxs.count(get_index(cluster_dir)) != 26:
                        print('Unfinished cluster ', cluster_dir, ' : ', processed_cluster_idxs.count(get_index(cluster_dir)), ' images')

                print(f'Number of bad clusters: {len(bad_cluster_dirs)}')
                print(f'Already processed : {len(set(processed_cluster_idxs))}')
                print(f'Remaining cluster indices : {list(set([get_index(cluster) for cluster in cluster_dirs]) - set(processed_cluster_idxs))}')
            else:
                print('Can not check clusters because the cluster directory does not yet exist!')
        else:
            bad_cluster_dirs = []

            # print(f'All cluster indices : {[get_index(cluster) for cluster in cluster_dirs]}')

            if move_to_front is not None:
                cluster_dirs.insert(0, cluster_dirs.pop([get_index(cluster)
                                                         for cluster in cluster_dirs].index(move_to_front)))
            if addition:
                # Remove clusters which are already processed, lie on a periodic boundary
                # or will take too long to process
                # good_cluster_idxs = [53, 78]  # snap_128
                # good_cluster_idxs = [139, 289, 1]  # snap_132
                # good_cluster_idxs = [0, 7, 28, 59, 2, 53, 46, 152]  # snap_132
                # AGN 400 remaining big clusters [0, 1, 2, 3, 4, 40]
                good_cluster_idxs = [0, 1, 2, 3, 4, 40]

                for cluster_dir in cluster_dirs:
                    if get_index(cluster_dir) not in good_cluster_idxs:
                        bad_cluster_dirs.append(cluster_dir)
            else:
                for cluster_dir in cluster_dirs:
                    if get_index(cluster_dir) in bad_cluster_idx:
                        bad_cluster_dirs.append(cluster_dir)

            for bad_cluster_dir in bad_cluster_dirs:
                cluster_dirs.remove(bad_cluster_dir)

            # print(f'Remaining cluster indices : {[get_index(cluster) for cluster in cluster_dirs]}')

            if multi_processing:
                params = [(cluster,
                           tfrecord_dir,
                           base_data_dir,
                           cluster_dirs,
                           snap_dir,
                           centers,
                           number_of_projections,
                           exposure_time,
                           redshift,
                           number_of_virtual_nodes,
                           number_of_neighbours,
                           plotting) for cluster in cluster_dirs]

                pool = Pool(cores)
                pool.starmap(generate_data, params)
            else:
                for cluster in cluster_dirs:
                    generate_data(cluster=cluster,
                                  tfrecord_dir=tfrecord_dir,
                                  base_data_dir=base_data_dir,
                                  cluster_dirs=cluster_dirs,
                                  snap_dir=snap_dir,
                                  centers=centers,
                                  number_of_projections=number_of_projections,
                                  exp_time=exposure_time,
                                  redshift=redshift,
                                  number_of_virtual_nodes=number_of_virtual_nodes,
                                  number_of_neighbours=number_of_neighbours,
                                  plotting=plotting)


if __name__ == '__main__':
    # Determine which snapshots to use
    magneticum_snap_dirs = ['snap_132']
    # Possible Magneticum dirs ['snap_128', 'snap_132', 'snap_136']

    bahamas_snap_dirs = []
    # Possible Bahamas dirs : ['AGN_TUNED_nu0_L100N256_WMAP9', 'AGN_TUNED_nu0_L400N1024_WMAP9']

    main(magneticum_snap_directories=magneticum_snap_dirs,
         bahamas_snap_directories=bahamas_snap_dirs,
         multi_processing=True,
         addition=False,
         check_clusters=False,
         number_of_virtual_nodes=10000,
         number_of_projections=26,
         exposure_time=1000.,
         redshift=0.20,
         plotting=0,
         cores=8,
         number_of_neighbours=6,
         move_to_front=None)
