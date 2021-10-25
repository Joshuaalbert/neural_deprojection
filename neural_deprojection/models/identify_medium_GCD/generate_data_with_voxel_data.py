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

import matplotlib.pyplot as plt
import neural_deprojection.models.identify_medium_GCD.gadget as g
from tqdm import tqdm
from astropy.io import fits
from multiprocessing import Pool, Lock

mp_lock = Lock()


def grid_properties(positions, properties, n=128):
    # positions.shape = (N,D)
    # properties.shape = (N,P)
    voxels = []
    # bins are n+1 equally spaced edges
    bins = [np.linspace(positions[:, d].min(), positions[:, d].max(), n+1) for d in range(positions.shape[1])]
    for p in range(properties.shape[1]):
        sum_properties, _ = np.histogramdd(positions, bins=bins, weights=properties[:, p])
        count, _ = np.histogramdd(positions, bins=bins)
        mean_properties = np.where(count == 0, 0, sum_properties/count)
        voxels.append(mean_properties)
    # central point of bins is grid point center
    center_points = [(b[:-1] + b[1:]) / 2. for b in bins]
    return np.stack(voxels, axis=-1), center_points


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
        n: Size of matrix, draws from O(n) group.

    Returns: random [n,n] matrix with determinant = +-1
    """
    det = -1.
    while det < 0:
        Q = _random_ortho_matrix(n)
        det = np.linalg.det(Q)
    return Q


def get_box_size(positions):
    max_pos = np.max(positions.T, axis=1)
    min_pos = np.min(positions.T, axis=1)

    box_size = max_pos - min_pos
    return box_size


# Finds the center of the gas particles in the snapshot by taking the average of the position extrema
# Check if a cluster is split by a periodic boundary
def check_split(positions, simulation_box):
    box_size = get_box_size(positions)
    split_cluster = False

    for coord, side in enumerate(box_size):
        if side > 0.5 * simulation_box[coord]:
            split_cluster = True

    return split_cluster


def unsplit_positions(positions, simulation_box):
    """
    Move the positions to the center of the simulation box, so they are no longer
    split by a periodic boundary.
    """
    new_positions = positions
    box_size = get_box_size(new_positions)

    for coord, side in enumerate(box_size):
        half_sim_box = 0.5 * simulation_box[coord]
        if side > half_sim_box:
            new_positions[:, coord] = np.where(positions[:, coord] > half_sim_box,
                                               positions[:, coord] - half_sim_box,
                                               positions[:, coord] + half_sim_box)
    return new_positions


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
    return cluster_idx, projection_idx, image


@tf.function
def downsample(image):
    filter_2d = tf.ones((2, 2, 1, 1)) * 0.25
    return tf.nn.conv2d(image[None, :, :, None],
                        filters=filter_2d, strides=2,
                        padding='SAME')[0, :, :, 0]


def get_clusters(snap_path, defect_clusters):
    snap_dir = os.path.basename(snap_path)

    # Make a list of clusters in the snapshot
    if snap_dir[0:3] == 'AGN':
        cluster_dirs = glob.glob(os.path.join(snap_path, '*'))
    else:
        cluster_dirs = glob.glob(os.path.join(snap_path, '*/*/*'))

    # List the indices of bad clusters
    print(f'\nNumber of clusters : {len(cluster_dirs)}')
    bad_cluster_idx = defect_clusters[snap_dir]['too_small'] + \
                      defect_clusters[snap_dir]['photon_max']
    print(f'Number of viable clusters : {len(cluster_dirs) - len(bad_cluster_idx)}')

    # Remove bad clusters from cluster list
    bad_cluster_dirs = []
    for cluster_dir in cluster_dirs:
        if get_index(cluster_dir) in bad_cluster_idx:
            bad_cluster_dirs.append(cluster_dir)
    for bad_cluster_dir in bad_cluster_dirs:
        cluster_dirs.remove(bad_cluster_dir)

    return cluster_dirs


def get_dirs_and_filename(cluster):
    tail, head = os.path.split(cluster)
    while os.path.basename(tail) != 'data':
        tail, head = os.path.split(tail)

    data_path = tail

    if get_simulation_name(cluster) == 'Bahamas':
        snap_dir = cluster.split('/')[-2]
        cluster_file = os.path.join(cluster, os.path.basename(cluster) + '.hdf5')
    else:
        snap_dir = cluster.split('/')[-4]
        cluster_file = os.path.join(cluster, snap_dir)
    return data_path, snap_dir, cluster_file


def load_data_magneticum(cluster_dir):
    _, _, cluster_file = get_dirs_and_filename(cluster_dir)
    ds = yt.load(cluster_file, long_ids=True)
    ad = ds.all_data()

    positions = ad['Gas', 'Coordinates'].in_cgs().d
    velocities = ad['Gas', 'Velocities'].in_cgs().d
    rho = ad['Gas', 'Density'].in_cgs().d
    u = ad['Gas', 'InternalEnergy'].in_cgs().d
    mass = ad['Gas', 'Mass'].in_cgs().d
    smooth = ad['Gas', 'SmoothingLength'].in_cgs().d

    codelength_per_cm = ad['Gas', 'Coordinates'].d[0][0] / positions[0][0]

    # Dimension of the box in which the particles exist
    cluster_box = get_box_size(positions)

    # Dimensions of to whole simulation box
    simulation_box = ds.domain_width.in_cgs().d

    # Adjust positions for clusters on periodic boundaries
    on_periodic_boundary = check_split(positions, simulation_box)
    if on_periodic_boundary:
        print('Cluster is located on a periodic boundary')
        positions = unsplit_positions(positions, simulation_box)

    # Center of the positions
    positions_center = np.mean(positions.T, axis=1)

    # For making the xray image, we still need the center as it's defined in the original data
    # Since the mean of positions split by a periodic boundary is not located at the
    # center of the cluster, we calculate the mean on the offset positions and undo the offset
    # on the calculated center.
    cluster_center = np.mean(positions.T, axis=1)

    if on_periodic_boundary:
        for coord, cluster_box_side in enumerate(cluster_box):
            half_sim_box = 0.5 * simulation_box[coord]
            if cluster_box_side > half_sim_box:
                if cluster_center[coord] > half_sim_box:
                    cluster_center[coord] -= half_sim_box
                else:
                    cluster_center[coord] += half_sim_box

    # Convert to codelength units for pyxsim
    cluster_center = cluster_center * codelength_per_cm

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

    return properties, cluster_center, positions_center, ds


def load_data_bahamas(cluster_dir, centers):
    data_path, snap_dir, cluster_file = get_dirs_and_filename(cluster_dir)
    with h5py.File(cluster_file, 'r') as ds:
        positions = np.array(ds['PartType0']['Coordinates'])
        velocities = np.array(ds['PartType0']['Velocity'])
        rho = np.array(ds['PartType0']['Density'])
        u = np.array(ds['PartType0']['InternalEnergy'])
        mass = np.array(ds['PartType0']['Mass'])
        smooth = np.array(ds['PartType0']['SmoothingLength'])

    # For some reason the Bahamas snapshots are structured so that when you load one part of the snapshot,
    # you load the entire simulation box, so there is not a specific reason to choose the first element of filenames
    filenames = glob.glob(os.path.join(data_path, snap_dir, 'data/snapshot_032/*.hdf5'))
    snap_file = filenames[0]
    ds = yt.load(snap_file)

    # Dimensions of to whole simulation box
    simulation_box = ds.domain_width.in_cgs().d

    if check_split(positions, simulation_box):
        print('Cluster is located on a periodic boundary')
        positions = unsplit_positions(positions, simulation_box)

    # Create a sphere around the center of the snapshot, which captures the photons
    positions_center = np.mean(positions.T, axis=1)

    # We can get the Bahamas cluster centers from the data itself
    cluster_center = centers[get_index(cluster_dir)]

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

    return properties, cluster_center, positions_center, ds


def load_data(cluster):
    data_path, snap_dir, _ = get_dirs_and_filename(cluster)

    # Load in particle data and prepare for making an xray image.
    if get_simulation_name(cluster) == 'Bahamas':
        gdata = g.Gadget(os.path.join(data_path, snap_dir), 'subh', snapnum=32, sim='BAHAMAS')

        subhalo_ids = [int(idx) for idx in gdata.read_var('FOF/FirstSubhaloID', verbose=False)]

        centers = gdata.read_var('Subhalo/CentreOfPotential', verbose=False)
        centers = centers[subhalo_ids[:-1]]
        # Convert to codelength by going from cm to Mpc and from Mpc to codelength
        centers /= gdata.cm_per_mpc / 0.7

        properties, cluster_center, unsplit_positions_center, dataset = load_data_bahamas(cluster_dir=cluster,
                                                                                          centers=centers)
    else:
        properties, cluster_center, unsplit_positions_center, dataset = load_data_magneticum(cluster_dir=cluster)

    sim_box = dataset.domain_width.in_cgs()

    return properties, cluster_center, unsplit_positions_center,  dataset, sim_box


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
        prefix: string, prefix for the tf record file name

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
                    (voxels, image, cluster_idx, projection_idx, vprime) = next(data_iterable)
                except StopIteration:
                    data_left = False
                    break
                # Write the graph, image and example_idx to the tfrecord file
                # graph = get_graph(graph, 0)
                features = dict(
                    voxels=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(voxels, tf.float32)).numpy()])),
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
                            value=[tf.io.serialize_tensor(tf.cast(projection_idx, tf.int32)).numpy()]))
                )
                # Set the features up so they can be written to the tfrecord file
                features = tf.train.Features(feature=features)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                # Status bar update
                pbar.update(1)
    print("Saved in tfrecords: {}".format(files))
    return files


def decode_examples(record_bytes, voxels_shape=None, image_shape=None):
    """
    Decodes raw bytes as returned from tf.data.TFRecordDataset([example_path]) into a GraphTuple and image
    Args:
        record_bytes: raw bytes
        voxels_shape: shape of voxels if known.
        image_shape: shape of image if known.

    Returns: (GraphTuple, image)
    """
    parsed_example = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        dict(
            voxels=tf.io.FixedLenFeature([], dtype=tf.string),
            image=tf.io.FixedLenFeature([], dtype=tf.string),
            cluster_idx=tf.io.FixedLenFeature([], dtype=tf.string),
            projection_idx=tf.io.FixedLenFeature([], dtype=tf.string),
            vprime=tf.io.FixedLenFeature([], dtype=tf.string)
        )
    )
    voxels = tf.io.parse_tensor(parsed_example['voxels'], tf.float32)
    voxels.set_shape(voxels_shape)

    image = tf.io.parse_tensor(parsed_example['image'], tf.float32)
    image.set_shape(image_shape)

    vprime = tf.io.parse_tensor(parsed_example['vprime'], tf.float32)
    vprime.set_shape((3, 3))

    cluster_idx = tf.io.parse_tensor(parsed_example['cluster_idx'], tf.int32)
    cluster_idx.set_shape(())

    projection_idx = tf.io.parse_tensor(parsed_example['projection_idx'], tf.int32)
    projection_idx.set_shape(())

    return voxels, image, cluster_idx, projection_idx, vprime


def plot_data(voxels, xray_image, mayavi=False, mayavi_prop_idx=0, histograms=False, xray=False):
    if mayavi:
        from mayavi import mlab
        mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 800))
        mlab.clf()

        prop = np.where(voxels[..., mayavi_prop_idx] == 0, -6, np.log10(voxels[..., mayavi_prop_idx]))
        mayavi_voxels = mlab.pipeline.scalar_field(prop)

        mlab.pipeline.volume(mayavi_voxels)
        # mlab.pipeline.iso_surface(mayavi_voxels, contours=24, opacity=0.05)
        # mlab.pipeline.scalar_cut_plane(mayavi_voxels, line_width=2.0, plane_orientation='z_axes')
        mlab.show()

    if histograms:
        plot_titles = ['rho', 'U',
                       'rho gradient x', 'rho gradient y', 'rho gradient z',
                       'U gradient x', 'U gradient y', 'U gradient z',
                       'rho laplacian', 'U laplacian']

        for i in range(voxels.shape[-1]):
            # project the voxels along the z-axis (we take the mean value along the z-axis)
            projected_img = np.sum(voxels, axis=-2) / voxels.shape[2]

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            prop_plot = ax.imshow(projected_img[..., i])
            fig.colorbar(prop_plot, ax=ax)
            ax.set_title(f'{plot_titles[i]}')
            plt.savefig(f'{plot_titles[i]}.png')
            plt.show()

    if xray:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(xray_image)
        plt.show()


def generate_data(cluster,
                  tfrecord_dir,
                  plt_kwargs,
                  number_of_clusters,
                  number_of_projections=26,
                  exp_time=1000.,
                  redshift=0.20,
                  number_of_voxels_per_dimension=64):
    data_path, snap_dir, _ = get_dirs_and_filename(cluster)
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

    units = np.array([Mpc,
                      Mpc,
                      Mpc,
                      1e-4 * pc / yr,
                      1e-4 * pc / yr,
                      1e-4 * pc / yr,
                      36.689 * 1e-7 * M_sun / pc ** 3,  # 36.6891392703865
                      4.2057 * 1e-7 * (pc / yr) ** 2,  # 4.205786556116279
                      1e8 * M_sun,
                      1e5 * pc])

    properties, cluster_center, positions_center, dataset, simulation_box = load_data(cluster)
    photon_source = dataset.sphere(cluster_center, radius)

    if properties.shape[0] < 10000:
        print(f'\nThe cluster contains {properties.shape[0]} particles '
              f'which is less than the threshold of 10000.')
        good_cluster = False

    # Set a minimum temperature to ignore particles that shouldn't be X-ray emitting,
    # set metallicity to 0.3 * Zsolar (should maybe fix later)
    # The source model determines the photon energy distribution and which photon energies to look at
    source_model = pyxsim.ThermalSourceModel(spectral_model="apec",
                                             emin=emin,
                                             emax=emax,
                                             nchan=n_chan,
                                             Zmet=metallicty,
                                             kT_min=kt_min)

    # Simulate the photons from the source located at a certain redshift,
    # that pass through a certain area, during a certain exposure time
    photons = pyxsim.PhotonList.from_data_source(data_source=photon_source,
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

    def data_generator():
        for projection_idx in tqdm(np.arange(number_of_projections)):
            print(f'\n\nCluster file: {cluster}')
            print(f'Cluster index: {cluster_idx}')
            print(f'Clusters done (or in the making): {len(glob.glob(os.path.join(tfrecord_dir, "*")))}')
            print(f'Projection: {projection_idx + 1} / {number_of_projections}')

            _properties = properties.copy()

            print(f'Particles in cluster: {_properties.shape[0]}\n')

            # Rotate variables
            rot_mat = _random_special_ortho_matrix(3)
            _properties[:, :3] = (rot_mat @ _properties[:, :3].T).T
            _properties[:, 3:6] = (rot_mat @ _properties[:, 3:6].T).T
            center = (rot_mat @ np.array(positions_center).T).T

            # Cut out box in 3D space based on diameter of the xray image
            lower_lim = center - 0.5 * cutout_box_size * np.array([1, 1, 1])
            upper_lim = center + 0.5 * cutout_box_size * np.array([1, 1, 1])
            indices = np.where((_properties[:, 0:3] < lower_lim) | (_properties[:, 0:3] > upper_lim))[0]
            _properties = np.delete(_properties, indices, axis=0)

            # Use neural network friendly units
            _properties[:, 0:3] = (_properties[:, 0:3] - center) / units[0:3]
            _properties[:, 3:6] = _properties[:, 3:6] / units[3:6]
            _properties[:, 6:] = _properties[:, 6:] / units[6:]

            # Create voxels from the sph particle properties
            voxels, center_points = grid_properties(positions=_properties[:, 0:3],
                                                    properties=_properties[:, 6:8],
                                                    n=number_of_voxels_per_dimension)

            # Calculate gradients and laplacians for the voxel properties
            voxels_grads = []
            voxels_laplacians = []
            for p in range(voxels.shape[-1]):
                _voxels_grads = np.gradient(voxels[:, :, :, p],
                                            *center_points)  # tuple of three arrays of shape (n,n,n)
                voxels_grads.append(np.stack(_voxels_grads, axis=-1))  # stack into shape (n,n,n,3)
                _voxels_laplacian = [np.gradient(_voxels_grads[i], center_points[i], axis=i) for i in range(3)]
                voxels_laplacians.append(sum(_voxels_laplacian))

            # Add the gradients and laplacians as channels to the voxels
            voxels_grads = np.concatenate(voxels_grads, axis=-1)  # (n,n,n,3*P)
            voxels_laplacians = np.stack(voxels_laplacians, axis=-1)  # (n,n,n,P)

            # Recalculate the voxels with the scaled log of the properties
            # 0.7036346061741174, 0.6853849479729205
            _properties[:, 6:8] = np.log10(_properties[:, 6:8]) / np.array([0.7036, 0.6853])
            voxels, _ = grid_properties(positions=_properties[:, 0:3],
                                        properties=_properties[:, 6:8],
                                        n=number_of_voxels_per_dimension)

            voxels_all = np.concatenate([voxels, voxels_grads, voxels_laplacians], axis=-1)  # (n,n,n,P+3*P+P)

            voxel_center = int(number_of_voxels_per_dimension / 2)
            print(f'Center voxel data :', voxels_all[voxel_center, voxel_center, voxel_center, :])
            print('Voxels shape: ', voxels_all.shape)

            v = np.eye(3)
            vprime = rot_mat.T @ v
            north_vector = vprime[:, 1]
            viewing_vec = vprime[:, 2]

            # Finds the events along a certain line of sight
            events_z = photons.project_photons(viewing_vec, sky_center, absorb_model="tbabs", nH=hydrogen_dens,
                                               north_vector=north_vector)

            # Write the events to a simput file
            cluster_projection_identity = number_of_projections * cluster_idx + projection_idx
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

            # Soxs creates fits files to store the Chandra mock xray image
            soxs.write_image(f'snap_{cluster_projection_identity}_evt.fits',
                             f'snap_{cluster_projection_identity}_img.fits',
                             emin=emin,
                             emax=emax,
                             overwrite=True)

            # Crop the xray image to 2048x2048 and store it in a numpy array
            with fits.open(f'snap_{cluster_projection_identity}_img.fits') as hdu:
                xray_image = np.array(hdu[0].data, dtype='float32')[1358:3406, 1329:3377]  # [2048,2048]

            # Remove the fits files created by soxs (now that we have the array, the fits files are no longer needed)
            temp_fits_files = glob.glob(os.path.join(os.getcwd(), f'snap_{cluster_projection_identity}_*.fits'))
            for file in temp_fits_files:
                print(f'Removing : {os.path.basename(file)}')
                os.remove(file)

            # Downsample xray image from 2048x2048 to 256x256
            xray_image = downsample(xray_image)
            xray_image = downsample(xray_image)
            xray_image = downsample(xray_image)[:, :, None]

            # Take the log (base 10) and enforce a minimum value of 1e-5
            xray_image = np.log10(np.where(xray_image < 1e-5, 1e-5, xray_image))

            # Plot voxel properties
            plot_data(voxels, xray_image, **plt_kwargs)

            voxels_all = tf.convert_to_tensor(voxels_all, tf.float32)

            # This function is a generator, which has the advantage of not keeping used and upcoming data in memory.
            yield voxels_all, xray_image, cluster_idx, projection_idx, vprime

    if good_cluster:
        # Save the data as tfrecords and return the filenames of the tfrecords
        save_examples(data_generator(),
                      save_dir=tfrecord_dir,
                      examples_per_file=number_of_projections,
                      num_examples=number_of_projections * number_of_clusters,
                      exp_time=exp_t[0],
                      prefix='train')


def main(data_dir,
         magneticum_snap_directories,
         bahamas_snap_directories,
         plt_kwargs,
         multi_processing=False,
         number_of_voxels_per_dimension=64,
         number_of_projections=26,
         exposure_time=5000.,
         redshift=0.20,
         cores=16,
         move_to_front=None):
    yt.funcs.mylog.setLevel(40)  # Suppresses yt status output.
    soxs.utils.soxsLogger.setLevel(40)  # Suppresses soxs status output.
    pyxsim.utils.pyxsimLogger.setLevel(40)  # Suppresses pyxsim status output.

    # Define the data directories of each simulation
    magneticum_data_dir = os.path.join(data_dir, 'Magneticum/Box2_hr')
    bahamas_data_dir = os.path.join(data_dir, 'Bahamas')

    # Directory where tf records will be saved
    my_tf_records_dir = os.path.join(data_dir, 'tf_records')

    # Define the full paths of the snapshots in each simulation
    magneticum_snap_paths = [os.path.join(magneticum_data_dir, snap_dir) for snap_dir in magneticum_snap_directories]
    bahamas_snap_paths = [os.path.join(bahamas_data_dir, snap_dir) for snap_dir in bahamas_snap_directories]

    # Per snapshot, define the indices of cluster that are excluded from generating the tf records
    defect_clusters = {'snap_128': {'photon_max': [53, 78],
                                    'too_small': []},
                       'snap_132': {'photon_max': [8, 52, 55, 93, 139, 289],
                                    'too_small': []},
                       'snap_136': {'photon_max': [96, 137, 51, 315, 216, 55, 102, 101, 20, 3],
                                    'too_small': []},
                       'AGN_TUNED_nu0_L100N256_WMAP9': {'photon_max': [3],
                                                        'too_small': [4, 10] + list(
                                                            set(np.arange(20, 200)) - {20, 21, 22, 28})},
                       'AGN_TUNED_nu0_L400N1024_WMAP9': {'photon_max': [],
                                                         'too_small': []}}

    # Iterate over the cosmological simulation snapshots
    for snap_path in magneticum_snap_paths + bahamas_snap_paths:
        print(f'\nSnapshot path : {snap_path}')
        snap_dir = os.path.basename(snap_path)

        # Directory where the tf records for the specific snapshot will be saved
        tfrecord_dir = os.path.join(my_tf_records_dir, snap_dir + '_tf_records')
        print(f'Tensorflow records will be saved in : {tfrecord_dir}')

        clusters = get_clusters(snap_path, defect_clusters)
        n_clusters = len(clusters)

        if move_to_front is not None:
            clusters.insert(0, clusters.pop([get_index(cluster)
                                             for cluster in clusters].index(move_to_front)))

        # Generate tf records from the cluster files
        if multi_processing:
            params = [(cluster,
                       tfrecord_dir,
                       plt_kwargs,
                       n_clusters,
                       number_of_projections,
                       exposure_time,
                       redshift,
                       number_of_voxels_per_dimension) for cluster in clusters]

            pool = Pool(cores)
            pool.starmap(generate_data, params)
        else:
            for cluster in clusters:
                generate_data(cluster=cluster,
                              tfrecord_dir=tfrecord_dir,
                              plt_kwargs=plt_kwargs,
                              number_of_clusters=n_clusters,
                              number_of_projections=number_of_projections,
                              exp_time=exposure_time,
                              redshift=redshift,
                              number_of_voxels_per_dimension=number_of_voxels_per_dimension)


if __name__ == '__main__':
    # Define the directories containing the data
    if os.getcwd().split('/')[2] == 's2675544':
        print('Running on ALICE')
        main_data_dir = '/home/s2675544/data'

        # Determine which snapshots to use on ALICE
        magneticum_snap_dirs = ['snap_132']
        bahamas_snap_dirs = []

        # Possible Magneticum dirs ['snap_128', 'snap_132', 'snap_136']
        # Possible Bahamas dirs : ['AGN_TUNED_nu0_L100N256_WMAP9', 'AGN_TUNED_nu0_L400N1024_WMAP9']

        multi_proc = True

        plotting_kwargs = {'mayavi': False,
                           'mayavi_prop_idx': 0,
                           'histograms': False,
                           'xray': False}
        mv_to_front = None
    else:
        main_data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/data'
        print('Running at home')

        # Determine which snapshots to use at home
        magneticum_snap_dirs = ['snap_132']
        bahamas_snap_dirs = []
        multi_proc = False

        plotting_kwargs = {'mayavi': False,
                           'mayavi_prop_idx': 0,
                           'histograms': False,
                           'xray': False}
        mv_to_front = 18

    main(data_dir=main_data_dir,
         magneticum_snap_directories=magneticum_snap_dirs,
         bahamas_snap_directories=bahamas_snap_dirs,
         plt_kwargs=plotting_kwargs,
         multi_processing=multi_proc,
         number_of_voxels_per_dimension=128,
         number_of_projections=26,
         exposure_time=1000.,
         redshift=0.20,
         cores=8,
         move_to_front=mv_to_front)
