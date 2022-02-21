import numpy as np
import os
import glob
import tensorflow as tf
import yt
import multiprocessing as mp
from tqdm import tqdm

yt.funcs.mylog.setLevel(40)  # Suppresses YT status output.


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


def save_examples(generator, snapshot, save_dir=None,
                  examples_per_file=26, num_examples=1, prefix='train'):
    """
    Save voxels to tfrecords
    """

    # make save dir
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    files = []
    data_iterable = iter(generator)
    data_left = True

    while data_left:
        file = os.path.join(save_dir, 'train_{:04d}.tfrecords'.format(snapshot))
        files.append(file)
        with tf.io.TFRecordWriter(file) as writer:
            for i in range(examples_per_file + 1):  # + 1 otherwise it makes a second tf rec file that is empty
                try:
                    print('try data...')
                    (voxels, proj_image, snapshot, projection, extra_info) = next(data_iterable)
                except StopIteration:
                    data_left = False
                    break
                # graph = get_graph(graph, 0)
                features = dict(
                    voxels=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(voxels, tf.float32)).numpy()])),
                    image=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(proj_image, tf.float32)).numpy()])),
                    snapshot=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(snapshot, tf.int32)).numpy()])),
                    projection=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(projection, tf.int32)).numpy()])),
                    extra_info=tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.cast(extra_info, tf.float32)).numpy()])),
                )
                features = tf.train.Features(feature=features)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

    return files


# taken from Matthijs' code
def grid_properties(positions, properties, n=128):
    """
    Create voxel grid from positions and properties.
    """
    # positions.shape = (N,D)
    # properties.shape = (N,P)

    voxels = []
    # bins are n+1 equally spaced edges
    bins = [np.linspace(positions[:, d].min(), positions[:, d].max(), n + 1) for d in range(positions.shape[1])]
    for p in range(properties.shape[1]):
        sum_properties, _ = np.histogramdd(positions, bins=bins, weights=properties[:, p])
        count, _ = np.histogramdd(positions, bins=bins)
        mean_properties = np.where(count == 0, 0, sum_properties / count)
        voxels.append(mean_properties)
    # central point of bins is grid point center
    center_points = [(b[:-1] + b[1:]) / 2. for b in bins]
    return np.stack(voxels, axis=-1), center_points


# altered from Matthijs' code, thanks Matthijs
def grad_and_laplace_voxels(voxels, center_points):
    # Calculate gradients and laplacians for the voxel properties

    # only for density
    _voxels_grads = np.gradient(voxels[:, :, :, -1],
                                *center_points)  # tuple of three arrays of shape (n,n,n)
    voxels_grads = np.stack(_voxels_grads, axis=-1)  # stack into shape (n,n,n,3)
    _voxels_laplacian = [np.gradient(_voxels_grads[i], center_points[i], axis=i) for i in range(3)]
    voxels_laplacians = sum(_voxels_laplacian)

    # voxels_grads = []
    # voxels_laplacians = []
    #
    # # loop through properties
    # for p in range(voxels.shape[-1]):
    #     _voxels_grads = np.gradient(voxels[:, :, :, p],
    #                                 *center_points)  # tuple of three arrays of shape (n,n,n)
    #     voxels_grads.append(np.stack(_voxels_grads, axis=-1))  # stack into shape (n,n,n,3)
    #     _voxels_laplacian = [np.gradient(_voxels_grads[i], center_points[i], axis=i) for i in range(3)]
    #     voxels_laplacians.append(sum(_voxels_laplacian))
    #
    # Add the gradients and laplacians as channels to the voxels
    # voxels_grads = np.concatenate(voxels_grads, axis=-1)  # (n,n,n,3*P)
    # voxels_laplacians = np.stack(voxels_laplacians, axis=-1)  # (n,n,n,P)

    voxels_all = np.concatenate([voxels, voxels_grads, voxels_laplacians[..., None]], axis=-1)  # (n,n,n,P+3+1)

    return voxels_all


def generate_voxel_grid(positions, properties, n=128):
    '''
    positions.shape = (N,D)
    properties.shape = (N,P)
    Returns:

    '''
    # generate voxels with density
    voxels, center_points = grid_properties(positions, properties, n)

    # calculate gradients & laplacians
    voxels = grad_and_laplace_voxels(voxels, center_points)

    # TODO: scale properties

    return voxels


def generate_voxel_data(positions, properties, proj_images, extra_info, save_dir, image_shape=(256, 256, 1),
                        voxels_per_dimension=128):
    """
        Routine for generating train data in tfrecords

        Returns: list of tfrecords.
        """

    # snapshot number
    snapshot = extra_info[0][0]

    def data_generator():
        # loop through projections
        for idx in range(len(positions)):
            # extract properties and projection
            _positions = positions[idx]
            _properties = properties[idx]
            proj_image = proj_images[idx].reshape(image_shape)
            snapshot = extra_info[idx][0]
            projection = extra_info[idx][1]

            # create voxel grid
            voxels = generate_voxel_grid(np.array(_positions), np.array(_properties), voxels_per_dimension)

            yield voxels, proj_image, snapshot, projection, np.array(extra_info)[idx]

    # save the examples to tfrecs
    train_tfrecords = save_examples(data_generator(),
                                    snapshot,
                                    save_dir=save_dir,
                                    examples_per_file=len(positions),
                                    num_examples=len(positions),
                                    prefix='train')
    return train_tfrecords


def snapshot_to_tfrec(params):
    """
    load snapshot plt file, rotate for different viewing angles, make projections and corresponding graph nets. Save to
    tfrec files.

    """

    # extract params
    snapshot_file, save_dir, num_of_projections, voxels_per_dimension = params

    print(f'snapshot to tfrec for {snapshot_file}')

    snapshot = int(snapshot_file[-4:])

    # skip if file already exists
    if os.path.isfile(os.path.join(save_dir, 'train_{:04d}.tfrecords'.format(snapshot))):
        return 0

    # load simulation data
    ds = yt.load(snapshot_file)  # loads in data into data set class. This is what we will use to plot field values
    ad = ds.all_data()  # generate data dictionary

    # simulation parameters
    field = 'density'
    width = np.max(ad[('gas', 'x')].to_value()) - np.min(ad[('gas', 'x')].to_value())
    resolution = 256

    # properties and units
    property_names = [('gas', 'x'), ('gas', 'y'), ('gas', 'z'), 'density']
    unit_names = ['pc', 'pc', 'pc', 'Msun/pc**3']

    # extract properties
    _values = []
    for name, unit in zip(property_names, unit_names):
        _values.append(ad[name].in_units(unit).to_value())

    property_values = np.array(_values).T  # n f

    # lists to store the different projections
    positions = []
    properties = []
    proj_images = []
    extra_info = []

    # rotate and project simulation
    projections = 0
    while projections < num_of_projections:
        # create rotation matrix
        V = np.eye(3)
        R = _random_special_ortho_matrix(3)
        Vprime = np.linalg.inv(R) @ V

        north_vector = Vprime[:, 1]
        viewing_vec = Vprime[:, 2]

        # store rotation and projection information
        _extra_info = [snapshot, projections,
                       viewing_vec[0], viewing_vec[1], viewing_vec[2],
                       north_vector[0], north_vector[1], north_vector[2],
                       resolution, width]

        # TODO: create image of laplacian?
        # project the image
        proj_image = yt.off_axis_projection(ds, center=[0, 0, 0], normal_vector=viewing_vec, north_vector=north_vector,
                                            item=field, width=width, resolution=resolution)

        # log scale and minimum value
        proj_image = np.log10(np.where(proj_image < 1e-5, 1e-5, proj_image))

        # rotate positions and velocities of simulation
        xyz = property_values[:, :3]  # n 3

        rotated_xyz = (R @ xyz.T).T

        # store properties
        _properties = property_values.copy()  # n f
        _properties[:, :3] = rotated_xyz  # n f

        _positions = xyz  # n 3

        # append to list of projections
        positions.append(_positions)
        properties.append(_properties)
        proj_images.append(proj_image)
        extra_info.append(_extra_info)

        projections += 1

    train_tfrecords = generate_voxel_data(positions, properties, proj_images, extra_info, save_dir,
                                          image_shape=(256, 256, 1),
                                          voxels_per_dimension=voxels_per_dimension)
    return 0


def main():
    # names of the simulation folders
    name_list = [
        'cournoyer/M4r5b',
        'cournoyer/M4r5b-3',
        'cournoyer/M4r5b-5',
        'cournoyer/M4r5s-2',
        'cournoyer/M4r5s-4',
        'cournoyer/M4r6b',
        'cournoyer/M4r6b-3',
        'cournoyer/M4r6s',
        'cournoyer/M4r5b-2',
        'cournoyer/M4r5b-4',
        'cournoyer/M4r5s',
        'cournoyer/M4r5s-3',
        'cournoyer/M4r5s-5',
        'cournoyer/M4r6b-2',
        'cournoyer/M4r6b-4',
        'lewis/run1',
        'lewis/run2',
        'lewis/run3',
        'lewis/run4'
    ]

    # directories
    folder_dir = '/disks/extern_collab_data/'
    save_dir = '/disks/extern_collab_data/hendrix/SCD_voxel_data/'

    # loop through simulations
    for n in name_list:
        # simulation directories
        sim_folder_path = os.path.join(folder_dir, n)
        sim_save_dir = os.path.join(save_dir, n)

        # because this run has too many snapshots, only select every third snapshot
        if n == 'lewis/run1':
            # make ordered array of all snapshots
            snapshot_list = np.array(
                [os.path.join(sim_folder_path, 'turbsph_hdf5_plt_cnt_{:04d}'.format(snap)) for snap in range(2118)]
            )

            # only keep every third snapshot
            snapshot_list = snapshot_list[[int(s[-4:]) % 3 == 0 for s in snapshot_list]]

        else:
            # read all snapshots
            snapshot_list = glob.glob(os.path.join(sim_folder_path, 'turbsph_hdf5_plt_cnt_*'))

        print(f'found {len(snapshot_list)} snapshots for {n}')

        # dataset parameters
        number_of_projections = 26  # number of random rotations of the simulation
        voxels_per_dimension = 64  # number of voxels per dimension of the simulation

        # multiprocessing parameters
        mp_params = [(snapshot,
                   sim_save_dir,
                   number_of_projections,
                   voxels_per_dimension) for snapshot in snapshot_list]

        # multiprocessing

        # num_workers = mp.cpu_count() - 1
        num_workers = 24

        print(f'generating data with {num_workers} workers...')
        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(snapshot_to_tfrec, mp_params),  # return results otherwise it doesn't work properly
                                total=len(mp_params)))


if __name__ == '__main__':
    main()
