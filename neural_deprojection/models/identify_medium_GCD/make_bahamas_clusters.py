import os
import gadget as g
import h5py
import numpy as np


def make_bahamas_clusters_box(data_dir, save_dir, number_of_clusters=200, starting_cluster=0):
    """
    Creates in the save_dir a npy file for a number of clusters in the snapshot located in data_dir.
    The npy files contain the particle data of their corresponding cluster. The particles are selected on the basis
    of being inside a fixed (physical) size box centered on the cluster. The size is determined by the field of view
    of the Chandra Acis-I instrument, and the fixed redshift at which the clusters will be viewed.

    Args:
        data_dir: Directory containing the bahamas data (e.g. 'AGN_TUNED_nu0_L100N256_WMAP9')
        save_dir: Directory to save the cluster hdf5 files in
        number_of_clusters: Number of clusters for which to make hdf5 files
        starting_cluster: The *number_of_clusters* largest clusters are selected, with index 0 the largest cluster.
    """
    if os.path.basename(data_dir) == 'AGN_TUNED_nu0_L400N1024_WMAP9':
        simulation_box_size = 400  # in simulation distance units
    elif os.path.basename(data_dir) == 'AGN_TUNED_nu0_L100N256_WMAP9':
        simulation_box_size = 100
    else:
        raise Exception(f'Snapshot {os.path.basename(data_dir)} is unknown!')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    gdata = g.Gadget(data_dir, 'subh', snapnum=32, sim='BAHAMAS')
    subhalo_ids = [int(idx) for idx in gdata.read_var('FOF/FirstSubhaloID', verbose=False)]

    centers = gdata.read_var('Subhalo/CentreOfPotential', verbose=False)
    centers = centers[subhalo_ids[:-1]]

    partdata = g.Gadget(data_dir, "particles", snapnum=32, sim="BAHAMAS")

    coordinates = partdata.read_var('PartType0' + '/Coordinates', verbose=False)
    internal_energy = partdata.read_var('PartType0' + '/InternalEnergy', verbose=False)
    rho = partdata.read_var('PartType0' + '/Density', verbose=False)

    properties = np.concatenate((coordinates, rho[..., None], internal_energy[..., None]), axis=1)

    h = 0.7

    # The factor sqrt(2) is to ensure that any projection of the simulation box still fills the fov
    cutout_box_size = 4.7048754823854417e+24 * np.sqrt(2)
    cluster_idx = starting_cluster
    clusters_saved = 0
    while clusters_saved < number_of_clusters:
        lower_lim = centers[cluster_idx] - 0.5 * cutout_box_size * np.array([1, 1, 1])
        upper_lim = centers[cluster_idx] + 0.5 * cutout_box_size * np.array([1, 1, 1])
        indices = np.where((coordinates < lower_lim) | (coordinates > upper_lim))[0]
        cluster_properties = np.delete(properties, indices, axis=0)

        print(cluster_idx, cluster_properties.shape)
        # Currently only use clusters that are not on a periodic boundary
        if any(lower_lim / gdata.cm_per_mpc * h < 0) or any(upper_lim / gdata.cm_per_mpc * h > simulation_box_size):
            print('split cluster')
            print(lower_lim / gdata.cm_per_mpc * h, upper_lim / gdata.cm_per_mpc * h)
        else:
            cluster_dir = os.path.join(save_dir, f'cluster_{cluster_idx:03}')
            if not os.path.isdir(cluster_dir):
                os.makedirs(cluster_dir, exist_ok=True)
            np.save(os.path.join(cluster_dir, f'cluster_{cluster_idx:03}.npy'), cluster_properties)
            clusters_saved += 1
        cluster_idx += 1


def make_bahamas_clusters_fof(data_dir,
                              save_dir,
                              number_of_clusters=200,
                              starting_cluster=0):
    """
    Creates in the save_dir a hdf5 file for a number of clusters in the snapshot located in data_dir.
    The hdf5 files contain the particle data of their corresponding cluster. The particles are selected based on their
    FoF group number, which we treat as clusters.

    Args:
        data_dir: Directory containing the bahamas data (e.g. 'AGN_TUNED_nu0_L100N256_WMAP9')
        save_dir: Directory to save the cluster hdf5 files in
        number_of_clusters: Number of clusters for which to make hdf5 files
        starting_cluster: The *number_of_clusters* largest clusters are selected, with index 0 the largest cluster.
    """

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # The original file contains all the particles in the simulation
    # Eventually we want to make a new hdf5 file that contains only the particles from a certain cluster
    path = os.path.join(data_dir, 'data/particledata_032/eagle_subfind_particles_032.0.hdf5')
    original_file = h5py.File(path, 'r')
    groups = list(original_file.keys())

    # Determine which groups contain particles and which don't.
    particle_groups = []
    non_particle_groups = []
    for group in groups:
        if 'PartType' in group:
            particle_groups.append(group)
        else:
            non_particle_groups.append(group)

    # Determine particle sub- and subsubgroups.
    # For every group that contains particles,
    # a selection needs to be taken that only contains the particles belonging to a particular cluster
    particle_subgroups = {group: list(original_file[group].keys()) for group in particle_groups}
    particle_subsubgroups = {group: {subgroup: list(original_file[group][subgroup].keys()) for subgroup in
                                     ['ElementAbundance', 'SmoothedElementAbundance']} for group in
                             ['PartType0', 'PartType4']}

    print('Particle groups made')

    # Load the particle data
    snapnum = 32
    # snapdata = g.Gadget(data_dir, "snap", snapnum, sim="BAHAMAS")
    partdata = g.Gadget(data_dir, "particles", snapnum, sim="BAHAMAS")

    print('Data loaded')

    # Find the FoF group number for every particle for every particle type
    fof_group_number_per_particle = {ptype: partdata.read_var(ptype + '/GroupNumber', verbose=False) for ptype in
                                     particle_groups}

    # Find the different FoF groups for every particle type by taking the set of the previous variable,
    # taking the length of the set gives an estimate of the number of 'clusters' per particle type
    fof_sets = {ptype: list(set(fof_group_number_per_particle[ptype])) for ptype in particle_groups}

    print('Data prepared')

    for cluster in range(starting_cluster, starting_cluster + number_of_clusters):

        cluster_dir = os.path.join(save_dir, f'cluster_{cluster:03}')
        if not os.path.isdir(cluster_dir):
            os.makedirs(cluster_dir, exist_ok=True)
        # Create a new file in which the cluster particles will be saved
        with h5py.File(os.path.join(cluster_dir, f'cluster_{cluster:03}.hdf5'), 'w') as f2:

            # Non-particle groups can just be copied
            for group in non_particle_groups:
                original_file.copy(group, f2)

            # Take a subset of all particles which are present in the cluster
            # and add their properties to the new hdf5 file
            for group in particle_groups:
                # Indices of the particles in the cluster subset for a certain particle type
                inds = np.where(fof_group_number_per_particle[group] == fof_sets[group][cluster])[0]
                # Make sure there are particles of the current type in the cluster subset
                if len(inds) != 0:
                    for subgroup in particle_subgroups[group]:
                        # These subgroups are actual groups instead of datasets, so we need to go one layer deeper
                        if subgroup in ['ElementAbundance', 'SmoothedElementAbundance']:
                            for subsubgroup in particle_subsubgroups[group][subgroup]:
                                field = group + '/' + subgroup + '/' + subsubgroup
                                # Create a new dataset with the subset of the particles
                                f2.create_dataset(field, data=partdata.read_var(field, verbose=False)[inds])
                                # Also add the attributes
                                for attr in list(original_file[field].attrs.items()):
                                    f2[field].attrs.create(attr[0], attr[1])
                        else:
                            # These 'subgroups' are datasets and can be added directly
                            field = group + '/' + subgroup
                            f2.create_dataset(field, data=partdata.read_var(field, verbose=False)[inds])
                            # Again also add the attributes
                            for attr in list(original_file[field].attrs.items()):
                                f2[field].attrs.create(attr[0], attr[1])
        print(f'cluster {cluster} particles are done')


if __name__ == '__main__':
    if os.getcwd().split('/')[2] == 's2675544':
        print('Running on ALICE')
        _data_dir = '/home/s2675544/data/AGN_TUNED_nu0_L400N1024_WMAP9'
        _save_dir = '/home/s2675544/data/Bahamas/AGN_TUNED_nu0_L400N1024_WMAP9'
    else:
        print('Running at home')
        _data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/data/' \
                    'AGN_TUNED_nu0_L100N256_WMAP9'
        _save_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/data/Bahamas/' \
                    'AGN_TUNED_nu0_L100N256_WMAP9'

    make_bahamas_clusters_box(_data_dir, _save_dir, number_of_clusters=200, starting_cluster=0)
    # make_bahamas_clusters_fof(data_dir, save_dir, number_of_clusters=5, starting_cluster=200)
