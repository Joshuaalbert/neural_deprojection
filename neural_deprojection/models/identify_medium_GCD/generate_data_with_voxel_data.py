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
# Check if a cluster is split by a periodic boundary
def check_split(dataset, position):
    max_pos = np.max(position.T, axis=1)
    min_pos = np.min(position.T, axis=1)

    box_size = max_pos - min_pos

    split_cluster = False

    for coord, side in enumerate(box_size):
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
    return cluster_idx, projection_idx, image


@tf.function
def downsample(image):
    filter = tf.ones((2, 2, 1, 1)) * 0.25
    return tf.nn.conv2d(image[None, :, :, None],
                        filters=filter, strides=2,
                        padding='SAME')[0, :, :, 0]


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


def load_data_bahamas(cluster, centers, data_dir):
    with h5py.File(os.path.join(cluster, os.path.basename(cluster) + '.hdf5'), 'r') as ds:
        positions = np.array(ds['PartType0']['Coordinates'])
        velocities = np.array(ds['PartType0']['Velocity'])
        rho = np.array(ds['PartType0']['Density'])
        u = np.array(ds['PartType0']['InternalEnergy'])
        mass = np.array(ds['PartType0']['Mass'])
        smooth = np.array(ds['PartType0']['SmoothingLength'])

    # For some reason the Bahamas snapshots are structured so that when you load one part of the snapshot,
    # you load the entire simulation box, so there is not a specific reason to choose the first element of filenames
    filenames = glob.glob(os.path.join(data_dir, cluster.split('/')[-2], 'data/snapshot_032/*.hdf5'))
    snap_file = filenames[0]
    ds = yt.load(snap_file)

    # Create a sphere around the center of the snapshot, which captures the photons
    c = centers[get_index(cluster)]

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
                  data_dir,
                  cluster_dirs,
                  snap_dir,
                  number_of_projections=26,
                  exp_time=1000.,
                  redshift=0.20,
                  number_of_voxels_per_dimension=64,
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
        gdata = g.Gadget(os.path.join(data_dir, snap_dir), 'subh', snapnum=32, sim='BAHAMAS')

        subhalo_ids = [int(id) for id in gdata.read_var('FOF/FirstSubhaloID', verbose=False)]

        centers = gdata.read_var('Subhalo/CentreOfPotential', verbose=False)
        centers = centers[subhalo_ids[:-1]]
        # Convert to codelength by going from cm to Mpc and from Mpc to codelength
        centers /= gdata.cm_per_mpc / 0.7

        properties, c, ds = load_data_bahamas(cluster=cluster,
                                              centers=centers,
                                              data_dir=data_dir)
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

    if properties.shape[0] < 10000:
        print(f'\nThe cluster contains {properties.shape[0]} particles '
              f'which is less than the threshold of 10000.')
        good_cluster = False

    # Set a minimum temperature to leave out that shouldn't be X-ray emitting,
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

    def data_generator():
        for projection_idx in tqdm(np.arange(number_of_projections)):
            print(f'\n\nCluster file: {cluster}')
            print(f'Cluster index: {cluster_idx}')
            print(f'Clusters done (or in the making) : {len(glob.glob(os.path.join(tfrecord_dir, "*")))}')
            print(f'Projection : {projection_idx + 1} / {number_of_projections}\n')

            _properties = properties.copy()

            # Rotate variables
            rot_mat = _random_special_ortho_matrix(3)
            _properties[:, :3] = (rot_mat @ _properties[:, :3].T).T
            _properties[:, 3:6] = (rot_mat @ _properties[:, 3:6].T).T
            center = (rot_mat @ np.array(sp.center.in_cgs()).T).T

            # Cut out box in 3D space based on diameter of the xray image
            lower_lim = center - 0.5 * cutout_box_size * np.array([1, 1, 1])
            upper_lim = center + 0.5 * cutout_box_size * np.array([1, 1, 1])
            indices = np.where((_properties[:, 0:3] < lower_lim) | (_properties[:, 0:3] > upper_lim))[0]
            _properties = np.delete(_properties, indices, axis=0)

            # Scale the variables to neural network friendly values
            _properties[:, 0:3] = (_properties[:, 0:3] - center) / units[0:3]
            _properties[:, 3:6] = _properties[:, 3:6] / units[3:6]
            _properties[:, 6:] = np.log10(_properties[:, 6:] / units[6:])
            center /= units[0:3]

            print(f'Properties :', _properties[0])
            print('Properties shape: ', _properties.shape)

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
            voxels_all = np.concatenate([voxels, voxels_grads, voxels_laplacians], axis=-1)  # (n,n,n,P+3*P+P)
            voxels_all = tf.convert_to_tensor(voxels_all, tf.float32)

            # Plot voxel properties
            if plotting > 0:
                plot_titles = ['rho', 'U',
                               'rho gradient x', 'rho gradient y', 'rho gradient z',
                               'U gradient x', 'U gradient y', 'U gradient z',
                               'rho laplacian', 'U laplacian']

                for i in range(voxels_all.shape[-1]):
                    # project the voxels along the z-axis (we take the mean value along the z-axis)
                    projected_img = tf.reduce_sum(voxels_all, axis=-2) / number_of_voxels_per_dimension

                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    prop_plot = ax.imshow(projected_img[..., i])
                    fig.colorbar(prop_plot, ax=ax)
                    ax.set_title(f'{plot_titles[i]}')
                    plt.savefig(f'{plot_titles[i]}.png')
                    plt.show()

            # This function is a generator, which has the advantage of not keeping used and upcoming data in memory.
            yield voxels_all, xray_image, cluster_idx, projection_idx, vprime

    if good_cluster:
        # Save the data as tfrecords and return the filenames of the tfrecords
        save_examples(data_generator(),
                      save_dir=tfrecord_dir,
                      examples_per_file=number_of_projections,
                      num_examples=number_of_projections * len(cluster_dirs),
                      exp_time=exp_t[0],
                      prefix='train')


def main(data_dir,
         magneticum_snap_directories,
         bahamas_snap_directories,
         multi_processing=False,
         number_of_voxels_per_dimension=64,
         number_of_projections=26,
         exposure_time=5000.,
         redshift=0.20,
         plotting=0,
         cores=16):
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

    # Iterate over the cosmological simulation snapshots
    for snap_path in magneticum_snap_paths + bahamas_snap_paths:
        print(f'\nSnapshot path : {snap_path}')
        snap_dir = os.path.basename(snap_path)

        # Make a list of clusters in the snapshot
        if snap_dir[0:3] == 'AGN':
            cluster_dirs = glob.glob(os.path.join(snap_path, '*'))
        else:
            cluster_dirs = glob.glob(os.path.join(snap_path, '*/*/*'))

        # Directory where the tf records for the specific snapshot will be saved
        tfrecord_dir = os.path.join(my_tf_records_dir, snap_dir + '_tf_records')
        print(f'Tensorflow records will be saved in : {tfrecord_dir}')

        # List the indices of bad clusters
        print(f'\nNumber of clusters : {len(cluster_dirs)}')
        bad_cluster_idx = defect_clusters[snap_dir]['split'] + \
                          defect_clusters[snap_dir]['too_small'] + \
                          defect_clusters[snap_dir]['photon_max']
        print(f'Number of viable clusters : {len(cluster_dirs) - len(bad_cluster_idx)}')

        # Remove bad clusters from cluster list
        bad_cluster_dirs = []
        for cluster_dir in cluster_dirs:
            if get_index(cluster_dir) in bad_cluster_idx:
                bad_cluster_dirs.append(cluster_dir)
        for bad_cluster_dir in bad_cluster_dirs:
            cluster_dirs.remove(bad_cluster_dir)

        # Generate tf records from the cluster files
        if multi_processing:
            params = [(cluster,
                       tfrecord_dir,
                       data_dir,
                       cluster_dirs,
                       snap_dir,
                       number_of_projections,
                       exposure_time,
                       redshift,
                       number_of_voxels_per_dimension,
                       plotting) for cluster in cluster_dirs]

            pool = Pool(cores)
            pool.starmap(generate_data, params)
        else:
            for cluster in cluster_dirs:
                generate_data(cluster=cluster,
                              tfrecord_dir=tfrecord_dir,
                              data_dir=data_dir,
                              cluster_dirs=cluster_dirs,
                              snap_dir=snap_dir,
                              number_of_projections=number_of_projections,
                              exp_time=exposure_time,
                              redshift=redshift,
                              number_of_voxels_per_dimension=number_of_voxels_per_dimension,
                              plotting=plotting)


if __name__ == '__main__':
    # Define the directories containing the data
    if os.getcwd().split('/')[2] == 's2675544':
        print('Running on ALICE')
        data_dir = '/home/s2675544/data'

        # Determine which snapshots to use on ALICE
        magneticum_snap_dirs = ['snap_128', 'snap_132', 'snap_136']
        bahamas_snap_dirs = ['AGN_TUNED_nu0_L400N1024_WMAP9']

        # Possible Magneticum dirs ['snap_128', 'snap_132', 'snap_136']
        # Possible Bahamas dirs : ['AGN_TUNED_nu0_L100N256_WMAP9', 'AGN_TUNED_nu0_L400N1024_WMAP9']

        multi_proc = True
    else:
        data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data'
        print('Running at home')

        # Determine which snapshots to use at home
        magneticum_snap_dirs = ['snap_132']
        bahamas_snap_dirs = []
        multi_proc = False

    main(data_dir=data_dir,
         magneticum_snap_directories=magneticum_snap_dirs,
         bahamas_snap_directories=bahamas_snap_dirs,
         multi_processing=multi_proc,
         number_of_voxels_per_dimension=128,
         number_of_projections=26,
         exposure_time=1000.,
         redshift=0.20,
         plotting=1,
         cores=8)
