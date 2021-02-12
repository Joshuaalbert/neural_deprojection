import h5py
import yt
from pyxsim import ThermalSourceModel, PhotonList
import soxs
import gadget as g
import numpy as np
import os
import glob

def make_bahamas_clusters_and_fits(simulation_dir,
                                   save_dir,
                                   number_of_clusters=200,
                                   starting_cluster=0,
                                   xray=False,
                                   numbers_of_xray_images=1):
    """

    Args:
        simulation_dir: Directory containing the bahamas data (e.g. 'AGN_TUNED_nu0_L100N256_WMAP9')
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
    path = os.path.join(simulation_dir, 'data/particledata_032/eagle_subfind_particles_032.0.hdf5')
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
    pdata = g.Gadget(simulation_dir, "particles", snapnum, sim="BAHAMAS")

    #Find the FoF group number for every particle for every particle type
    fof_group_number_per_particle = {ptype: pdata.read_var(ptype + '/GroupNumber', verbose=False) for ptype in particle_groups}

    #Find the different FoF groups for every particle type by taking the set of the previous variable,
    #taking the length of the set gives an estimate of the number of 'clusters' per particle type
    fof_sets = {ptype: list(set(fof_group_number_per_particle[ptype])) for ptype in particle_groups}

    #Parameters for making the xray images
    exp_time = (200., "ks") # exposure time
    area = (2000.0, "cm**2") # collecting area
    redshift = 0.05
    min_E = 0.05 #Minimum energy of photons in keV
    max_E = 11.0 #Maximum energy of photons in keV
    Z = 0.3 #Metallicity in units of solar metallicity
    kT_min = 0.05 #Minimum temperature to solve emission for
    n_chan = 1000 #Number of channels in the spectrum
    nH = 0.04 #The foreground column density in units of 10^22 cm^{-2}. Only used if absorption is applied.
    radius = 5.0 #Radius of the sphere which captures photons
    sky_center = [45., 30.] #Ra and dec coordinates of the cluster (which are currently dummy values)

    #The line of sight is oriented such that the north vector
    #would be projected on the xray image as a line from the center to the top of the image.
    north_vector = np.array([0., 1., 0.])

    #Converts a line_of_sight vector to a string: shows the first 2 decimals of every dimension
    #This is added to the xray fits filename
    def line_of_sight_to_string(normal_vector):
        normal_string = ''
        for i in normal_vector:
            angle_string = str(i).replace('.', '')
            if angle_string[0] == '-':
                angle_string = '-' + angle_string[2:4]
            else:
                angle_string = angle_string[1:3]
            normal_string += '_' + angle_string
        return normal_string

    #Define the centers of clusters as the center of potential of friends-of-friends groups
    #'subh' stands for subhalos
    snapnum = 32
    gdata = g.Gadget(simulation_dir, 'subh', snapnum, sim='BAHAMAS')
    centers = gdata.read_var('FOF/GroupCentreOfPotential', verbose=False)
    #Convert to codelength by going from cm to Mpc and from Mpc to codelength
    centers /= gdata.cm_per_mpc * 1.42855

    # Set a minimum temperature to leave out that shouldn't be X-ray emitting, set metallicity to 0.3 Zsolar (should maybe fix later)
    # The source model determines the energy distribution of photons that are emitted
    source_model = ThermalSourceModel("apec", min_E, max_E, n_chan, Zmet=Z)

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

        if xray:
            # For some reason the Bahamas snapshots are structured so that when you load one snapshot, you load the entire simulation box
            # so there is not a specific reason to choose the first element of filenames
            filenames = glob.glob(os.path.join(simulation_dir, 'data/snapshot_032/*'))
            snap_file = filenames[0]
            ds = yt.load(snap_file)

            # Create a sphere around the center of the snapshot, which captures the photons
            sphere_center = centers[cluster]
            sp = ds.sphere(sphere_center, radius)

            # Create the photonlist
            photons = PhotonList.from_data_source(sp, redshift, area, exp_time, source_model)

            # Take a number of random lines of sight to create the xray images
            lines_of_sight = [list(2.0 * (np.random.random(3) - 0.5)) for i in range(numbers_of_xray_images)]

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
                                 os.path.join(cluster_dir,
                                              f"img_{cluster:03}" + line_of_sight_to_string(line_of_sight) + ".fits"),
                                 emin=min_E,
                                 emax=max_E,
                                 overwrite=True)
            print(f'cluster {cluster} image is done')

if __name__ == '__main__':
    # Directory containing the bahamas data
    data_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data'

    # Directory to save the cluster hdf5 files in
    save_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Bahamas'

    # Determine which simulation to use
    sim_dir = 'AGN_TUNED_nu0_L100N256_WMAP9'
    # sim_dir = 'AGN_TUNED_nu0_L400N1024_WMAP9/'

    simulation_dir = os.path.join(data_dir, sim_dir)

    # Specify the number of clusters for which to make hdf5 files
    # The centers list will be ordered, so by taking the number of fits as e.g. 5, the 5 biggest clusters will be used.
    starting_cluster = 0
    number_of_clusters = 200
    xray = False
    number_of_xray_images = 1

    make_bahamas_clusters_and_fits(simulation_dir,
                                   save_dir,
                                   number_of_clusters=number_of_clusters,
                                   starting_cluster=starting_cluster,
                                   xray=xray,
                                   numbers_of_xray_images=number_of_xray_images)





