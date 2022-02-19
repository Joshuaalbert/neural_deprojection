"""
README

This script stores the mean and std of rho and U data of all particles in all clusters and
calculates the unit and number for scaling these properties.
"""
import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

import os
import yt
import soxs
import pyxsim
import time
import numpy as np
from neural_deprojection.models.identify_medium_GCD.generate_data_with_voxel_data import grid_properties, \
    get_index, get_clusters, get_dirs_and_filename, load_data, smooth_voxels, add_gradients_and_laplacians


def global_mean_and_std(means, stds):
    """
    Given the mean and std of a property (rho or U) for a number of subsets (i.e. clusters),
    calculate the global mean of property and the global std of property of the union of the sets,
    so that (property - global_mean) / global_std has a mean of 0 and a std of 1

    This function assumes that all subsets have the same length (each cluster must have the same number of voxels)

    means: array, array containing the subset means, size [number of subsets/clusters]
    stds: array, array containing the subset stds, size [number of subsets/clusters]
    """
    global_mean = np.mean(means)
    global_std = np.sqrt(np.mean(stds**2 + (means - global_mean)**2))
    return global_mean, global_std


def global_mean_and_std_of_logs(means_of_logs, stds_of_logs):
    """
    Given the mean and std of some log(property) for a number of subsets (i.e. clusters),
    calculate the global mean of property and the global std of log(property) of the union of the sets,
    so that log(property / global_mean) / global_std_of_logs has a mean of 0 and a std of 1

    This function assumes that all subsets have the same length (each cluster must have the same number of voxels)

    means_of_logs: array, array with the mean of log(property) of each subset
    stds_of_logs: array, array with the std of log(property) of each subset
    """
    global_mean_of_logs = np.mean(means_of_logs)
    global_std_of_logs = np.sqrt(stds_of_logs**2 + (means_of_logs - global_mean_of_logs)**2)
    global_mean = 10**global_mean_of_logs
    return global_mean, global_std_of_logs


def store_means_and_stds(voxels, n_properties):
    """
    For each channel, store the its mean value and its standard deviation.
    For the property channels, first take the log

    voxels: 3D array, contains properties, gradients and laplacians
    n_properties: int, number of properties in voxels
    """
    n_channels = voxels.shape[-1]
    channel_means_stds = np.zeros((n_channels, 2))
    for i in np.arange(n_channels):
        if i >= n_properties:
            # gradient or laplacian
            channel = voxels[:, :, :, i].flatten()
        else:
            # property
            channel = np.log10(voxels[:, :, :, i].flatten())
        channel_means_stds[i][0] = np.mean(channel)
        channel_means_stds[i][1] = np.std(channel)
    return channel_means_stds


def main(data_path,
         magneticum_snap_directories,
         bahamas_snap_directories,
         number_of_voxels_per_dimension,
         n_properties,
         smooth_support=12):
    yt.funcs.mylog.setLevel(40)  # Suppresses yt status output.
    soxs.utils.soxsLogger.setLevel(40)  # Suppresses soxs status output.
    pyxsim.utils.pyxsimLogger.setLevel(40)  # Suppresses pyxsim status output.

    # Define the data directories of each simulation
    magneticum_data_path = os.path.join(data_path, 'Magneticum/Box2_hr')
    bahamas_data_path = os.path.join(data_path, 'Bahamas')

    # Define the full paths of the snapshots in each simulation
    magneticum_snap_paths = [os.path.join(magneticum_data_path, snap_dir) for snap_dir in magneticum_snap_directories]
    bahamas_snap_paths = [os.path.join(bahamas_data_path, snap_dir) for snap_dir in bahamas_snap_directories]

    n_channels = 5 * n_properties
    channel_means_and_stds_per_cluster = np.empty((n_channels, 2, 0))  # [n_channels, (mean, std), n_clusters]

    # Iterate over the cosmological simulation snapshots
    for snap_path in bahamas_snap_paths + magneticum_snap_paths:
        clusters = get_clusters(snap_path, existing_cluster_identities=None)

        # For each cluster:
        # - calculate voxels
        # - smooth voxels
        # - calculate gradients and laplacians
        # - store means and stds

        for cluster in clusters:
            t0 = time.time()
            properties, _, _, _, _ = load_data(cluster)
            _, snap_dir, _ = get_dirs_and_filename(cluster)

            print('snap:', snap_dir)
            print('cluster:', get_index(cluster))
            print('number of particles:', properties.shape[0])

            # Create voxels from the sph particle properties
            voxels, center_points = grid_properties(positions=properties[:, 0:3],
                                                    properties=properties[:, 3:5],
                                                    n=number_of_voxels_per_dimension)

            # Smooth the voxels
            voxels = smooth_voxels(voxels, smooth_support)

            # Calculate gradients and laplacians for the voxels and concatenate them together
            voxels_all = add_gradients_and_laplacians(voxels, center_points)

            # Store the mean and std of gradients, laplacian and log(property)
            channel_means_stds = store_means_and_stds(voxels_all, n_properties)

            channel_means_and_stds_per_cluster = np.append(channel_means_and_stds_per_cluster,
                                                           channel_means_stds[..., None], axis=-1)

            print('time', time.time() - t0)

    for i in np.arange(channel_means_and_stds_per_cluster.shape[0]):
        print('\nchannel index ', i)
        if i >= n_properties:
            # gradient or laplacian
            new_unit, number = global_mean_and_std(channel_means_and_stds_per_cluster[i][0],
                                                   channel_means_and_stds_per_cluster[i][1])
        else:
            # property
            new_unit, number = global_mean_and_std_of_logs(channel_means_and_stds_per_cluster[i][0],
                                                           channel_means_and_stds_per_cluster[i][1])
        print('new unit ', new_unit)
        print('number ', number)
        with open('units.txt', 'a') as units_file:
            units_file.write(f'channel index {i}\n')
            units_file.write(f'unit {new_unit}\n')
            units_file.write(f'number {number}\n\n')


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

    main(data_path=data_dir,
         magneticum_snap_directories=magneticum_snap_dirs,
         bahamas_snap_directories=bahamas_snap_dirs,
         number_of_voxels_per_dimension=64,
         n_properties=2,
         smooth_support=12)
