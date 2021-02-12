import yt
import pyxsim
import glob
import soxs
import numpy as np
import os

def make_magneticum_fits(data_dir, snap_file, alice=True, number_of_xray_images=1):
    """

    Args:
        snap_file: The snapshot filename (e.g. 'snap_128' or 'snap_132')
        alice: Whether this function is run on Alice (directories are different),
        if False, data_dir and snap_dir are not used.
        number_of_xray_images: How many xray images per cluster

    Returns:
        Creates a number of xray images for every

    """
    if alice:
        snap_dir = os.path.join(data_dir, snap_file + '/*/simcut/*/' + snap_file)
        clusters = glob.glob(snap_dir)
    else:
        # Home
        #Directory containing magneticum snapshots
        snap_dir = '/home/matthijs/PycharmProjects/TensorflowTest/pyXSIM/snaps/'

        #Directory to save the xray fits files in
        save_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/Magneticum/snap_xrays'

        #Select the the snapshots you want to use
        clusters = glob.glob(snap_dir + 'snap_???_?')

    #Parameters for making the xray images
    exp_time = (200., "ks") # exposure time
    area = (2000.0, "cm**2") # collecting area
    redshift = 0.20
    min_E = 0.05 #Minimum energy of photons in keV
    max_E = 11.0 #Maximum energy of photons in keV
    Z = 0.3 #Metallicity in units of solar metallicity
    kT_min = 0.05 #Minimum temperature to solve emission for
    n_chan = 1000 #Number of channels in the spectrum
    nH = 0.04 #The foreground column density in units of 10^22 cm^{-2}. Only used if absorption is applied.
    radius = (2.0,"Mpc") #Radius of the sphere which captures photons
    sky_center = [45., 30.] #Ra and dec coordinates of the cluster (which are currently dummy values)

    #The line of sight is oriented such that the north vector
    #would be projected on the xray image, as a line from the center to the top of the image
    north_vector = np.array([0., 1., 0.])

    #Converts a line_of_sight vector to a string: shows the first 2 decimals of every dimension
    #This is added to the xray fits filename
    def line_of_sight_to_string(normal_vector, decimals: int = 2):
        normal_string = ''
        for i in normal_vector:
            angle_string = str(i).replace('.', '')
            if angle_string[0] == '-':
                angle_string = '-' + angle_string[2:2+decimals]
            else:
                angle_string = angle_string[1:1+decimals]
            normal_string += '_' + angle_string
        return normal_string

    #Finds the center of the gas particles in the snapshot by taking the average of the position extrema
    def find_center(position, offset=[0., 0., 0.]):
        x, y, z = position.T

        x_max, x_min = max(x), min(x)
        y_max, y_min = max(y), min(y)
        z_max, z_min = max(z), min(z)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        return [x_center + offset[0], y_center + offset[1], z_center + offset[2]], [x_max - x_min, y_max - y_min, z_max - z_min]

    # SimCut files (until November 24, 2020) have fields in this order, so we need to specify them
    my_field_def = (
        "Coordinates",
        "Velocities",
        "Mass",
        "ParticleIDs",
        ("InternalEnergy", "Gas"),
        ("Density", "Gas"),
        ("SmoothingLength", "Gas"),
    )

    #For every selected snapshot, create an xray image for one or more lines of sight
    for cluster in clusters:
        # Set long_ids = True because the IDs are 64-bit ints
        ds = yt.load(cluster, long_ids=True, field_spec=my_field_def)

        #Use the gas particle positions to find the center of the snapshot
        ad = ds.all_data()
        pos = ad['Gas', 'Coordinates'].d
        c = find_center(pos, [0., 0., 0.])[0]

        #Create a sphere around the center of the snapshot, which captures the photons
        sp = ds.sphere(c, radius)

        # Set a minimum temperature to leave out that shouldn't be X-ray emitting, set metallicity to 0.3 Zsolar (should maybe fix later)
        #The source model determines the distribution of photons that are emitted
        source_model = pyxsim.ThermalSourceModel("apec", min_E, max_E, n_chan, Zmet=Z, kT_min=kT_min)

        #Create the photonlist
        photons = pyxsim.PhotonList.from_data_source(sp, redshift, area, exp_time, source_model)

        # Take a number of random lines of sight to create the xray images
        lines_of_sight = [list(2.0 * (np.random.random(3) - 0.5)) for i in range(number_of_xray_images)]

        #Make an xray image for a set of lines of sight
        for line_of_sight in lines_of_sight:
            #Finds the events along a certain line of sight
            events_z = photons.project_photons(line_of_sight, sky_center, absorb_model="tbabs", nH=nH, north_vector=north_vector)

            events_z.write_simput_file("magneticum", overwrite=True)

            #Determine which events get detected by the AcisI intstrument of Chandra
            soxs.instrument_simulator("magneticum_simput.fits", "magneticum_evt.fits", exp_time, "chandra_acisi_cy0",
                                      sky_center, overwrite=True, ptsrc_bkgnd=False, foreground=False, instr_bkgnd=False)

            if alice:
                soxs.write_image("magneticum_evt.fits",
                                 os.path.join(os.path.dirname(cluster),
                                              "xray_image" + line_of_sight_to_string(line_of_sight) + ".fits"),
                                 emin=min_E,
                                 emax=max_E,
                                 overwrite=True)
            else:
                #Write the detections to a fits file
                soxs.write_image("magneticum_evt.fits",
                                 os.path.join(save_dir,
                                              cluster.split('/')[-1] + line_of_sight_to_string(line_of_sight) + ".fits"),
                                 emin=min_E,
                                 emax=max_E,
                                 overwrite=True)

if __name__ == '__main__':
    data_dir = '/home/s2675544/data/Magneticum/Box2_hr'
    snap_file = 'snap_128'
    make_magneticum_fits(data_dir=data_dir, snap_file=snap_file, alice=False)