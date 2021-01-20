import yt
import numpy as np
from tqdm import tqdm

yt.funcs.mylog.setLevel(40)  # Surpresses YT status output.

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

folder_path = '~/Desktop/SCD/SeanData/'
# folder_path = '~/data/SeanData/M3f2/'
snapshot = 3136
filename = 'turbsph_hdf5_plt_cnt_{}'.format(snapshot)  # only plt file, will automatically find part file

file_path = folder_path + filename
ds = yt.load(file_path)  # loads in data into data set class. This is what we will use to plot field values
ad = ds.all_data()  # Can call on the data set's property .all_data() to generate a dictionary
# containing all data available to be parsed through.
# e.g. print ad['mass'] will print the list of all cell masses.
# if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]

print("making positions & properties...")
# one_scale = ad.cut_region("obj['grid_level'] == 6")

print('finding amount of particles per cell...')
alloc_list = [0, []]
for cell_ind in range(0, 2250):
    if len(ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))['grid_indices']) != 0:
        particle_gridsize = round(len(ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))['x'])**(1/3.))
        print('particle_gridsize = {}'.format(particle_gridsize), '\nmaking allocation list...')
        alloc_list[0] = particle_gridsize
        for x in range(particle_gridsize):
            for y in range(particle_gridsize):
                for z in range(particle_gridsize):
                    alloc_list[1].append([x, y, z])
        print('done!')
        break


field ='density' # dens, temp, and pres are some shorthand strings recognized by yt.
# ax = 'y' # the axis our slice plot will be "looking down on".
L = [1,0,0] # vector normal to cutting plane
v_elements = [-1, 0, 1]
folder_path = '~/Desktop/SCD/SeanData/test_pos_prop_im/'

print('making projections and pos_prop_array...')
counter = 0
for x_e in v_elements:
    for y_e in v_elements:
        for z_e in v_elements:
            if x_e == y_e == z_e == 0:
                continue
            counter += 1
            L = [x_e, y_e, z_e]
            print('projection {}, [ {} / {} ]'.format(L, counter, len(v_elements)**3))
            plot_ = yt.OffAxisProjectionPlot(ds, L, field)
            im_name = folder_path + 'axis_{}_snapshot_{}.png'.format(str(x_e)+str(y_e)+str(z_e), snapshot)
            plot_.save(im_name)
            rot_mat = rotation_matrix_from_vectors([1, 0, 0], L)

            positions = []
            properties = []

            for cell_ind in tqdm(range(0, 2250)):
                if len(ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))['grid_indices']) != 0:
                    particles_in_cell = ad.cut_region("obj['grid_indices'] == {}".format(cell_ind))
                    feature_array = np.zeros(shape=(alloc_list[0], alloc_list[0], alloc_list[0], 10))
                    for i, alloc in enumerate(alloc_list[1]):
                        i_x, i_y, i_z = alloc[0], alloc[1], alloc[2]

                        pos = [particles_in_cell['x'][i].to_value(),
                               particles_in_cell['y'][i].to_value(),
                               particles_in_cell['z'][i].to_value()]

                        vel = [particles_in_cell['velocity_x'][i].to_value(),
                               particles_in_cell['velocity_y'][i].to_value(),
                               particles_in_cell['velocity_z'][i].to_value()]

                        pos_rot = rot_mat.dot(pos)      # rotate to off axis coordinate frame
                        vel_rot = rot_mat.dot(vel)

                        feature_array[i_x, i_y, i_z, 0] = pos_rot[0]
                        feature_array[i_x, i_y, i_z, 1] = pos_rot[1]
                        feature_array[i_x, i_y, i_z, 2] = pos_rot[2]
                        feature_array[i_x, i_y, i_z, 3] = vel_rot[0]
                        feature_array[i_x, i_y, i_z, 4] = vel_rot[1]
                        feature_array[i_x, i_y, i_z, 5] = vel_rot[2]
                        feature_array[i_x, i_y, i_z, 6] = np.log10(particles_in_cell['density'][i].to_value())
                        feature_array[i_x, i_y, i_z, 7] = np.log10(particles_in_cell['temperature'][i].to_value())
                        feature_array[i_x, i_y, i_z, 8] = np.log10(particles_in_cell['cell_mass'][i].to_value())
                        feature_array[i_x, i_y, i_z, 9] = particles_in_cell['gravitational_potential'][i].to_value()

                    m_pos = [np.mean(particles_in_cell['x'].to_value()),
                             np.mean(particles_in_cell['y'].to_value()),
                             np.mean(particles_in_cell['z'].to_value())]

                    m_vel = [np.mean(particles_in_cell['velocity_x'].to_value()),
                             np.mean(particles_in_cell['velocity_y'].to_value()),
                             np.mean(particles_in_cell['velocity_z'].to_value())]

                    m_pos_rot = rot_mat.dot(m_pos)      # rotate to off axis coordinate frame
                    m_vel_rot = rot_mat.dot(m_vel)

                    mean_x, mean_y, mean_z = m_pos_rot[0], m_pos_rot[1], m_pos_rot[2]
                    mean_vx, mean_vy, mean_vz = m_vel_rot[0], m_vel_rot[1], m_vel_rot[2]
                    mean_density = np.mean(particles_in_cell['density'].to_value())
                    mean_temperature = np.mean(particles_in_cell['temperature'].to_value())
                    mean_pot = np.mean(particles_in_cell['gravitational_potential'].to_value())
                    sum_mass = np.sum(particles_in_cell['cell_mass'].to_value())

                    features = [mean_x, mean_y, mean_z,
                                mean_vx, mean_vy, mean_vz,
                                mean_density, mean_temperature, mean_pot,
                                sum_mass, np.log10(particles_in_cell['cell_volume'][0].to_value()),
                                feature_array]

                    properties.append(features)
                    positions.append([mean_x, mean_y, mean_z])

            pos_array = np.array(positions)
            pos_array.save(folder_path + 'pos_axis_{}_snapshot_{}.npy')
            prop_array = np.array(properties)
            prop_array.save(folder_path + 'prop_axis_{}_snapshot_{}.npy')