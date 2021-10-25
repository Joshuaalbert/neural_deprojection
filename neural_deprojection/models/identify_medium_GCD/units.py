"""
README

This script stores the rho and U data of all particles in all clusters and
calculates the the unit and number for scaling these properties.
See the function calculate_unit_and_scaling() for how the units and numbers are calculated.
"""
import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

import os
import glob
import yt
import h5py
import soxs
import pyxsim
import numpy as np

import neural_deprojection.models.identify_medium_GCD.gadget as g


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


def calculate_unit_and_scaling(quantity_array):
    log_quantity = np.log10(quantity_array)
    unit = 10 ** np.mean(log_quantity)
    number = np.sqrt(np.mean(np.square(log_quantity - np.log10(unit))))
    return unit, number


def main(data_dir,
         magneticum_snap_directories,
         bahamas_snap_directories):
    yt.funcs.mylog.setLevel(40)  # Suppresses yt status output.
    soxs.utils.soxsLogger.setLevel(40)  # Suppresses soxs status output.
    pyxsim.utils.pyxsimLogger.setLevel(40)  # Suppresses pyxsim status output.

    # Define the data directories of each simulation
    magneticum_data_dir = os.path.join(data_dir, 'Magneticum/Box2_hr')
    bahamas_data_dir = os.path.join(data_dir, 'Bahamas')

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
    rhos = np.array([])
    Us = np.array([])

    # Iterate over the cosmological simulation snapshots
    for snap_path in magneticum_snap_paths + bahamas_snap_paths:
        clusters = get_clusters(snap_path, defect_clusters)
        for cluster in clusters:
            properties, _, _, _, _ = load_data(cluster)

            rho = properties[:, 6]
            U = properties[:, 7]

            print(rho.shape)

            # Store the rhos and Us of all particles in all clusters
            rhos = np.concatenate((rhos, rho), axis=0)
            Us = np.concatenate((Us, U), axis=0)

    yr = 3.15576e7  # in seconds
    pc = 3.085678e18  # in cm
    M_sun = 1.989e33  # in gram

    rho_unit = 1e-7 * M_sun / pc ** 3
    U_unit = 1e-7 * (pc / yr) ** 2

    for quantity_array, original_unit in zip([rhos, Us], [rho_unit, U_unit]):
        new_unit, number = calculate_unit_and_scaling(quantity_array)
        data = np.log10(quantity_array / new_unit) / number

        print('\nnew unit ', new_unit)
        print('number ', number)
        print('mean (should be 0)', np.mean(data))
        print('std (should be 1)', np.std(data))
        print('unit factor ', new_unit / original_unit)


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
    else:
        data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/data'
        print('Running at home')

        # Determine which snapshots to use at home
        magneticum_snap_dirs = ['snap_132']
        bahamas_snap_dirs = []

    main(data_dir=data_dir,
         magneticum_snap_directories=magneticum_snap_dirs,
         bahamas_snap_directories=bahamas_snap_dirs)
