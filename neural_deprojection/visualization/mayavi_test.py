from mayavi import mlab
import numpy as np
import h5py
import pysph.tools.interpolator as interp
from pysph.base.particle_array import ParticleArray
from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction
from mayavi.tools.pipeline import scalar_field

#my_file2 = '/home/julius/Desktop/SCD/hydro_gas_particles_i00905.hdf5'
my_file2 = '/home/matthijs/PycharmProjects/TensorflowTest/hydro_gas_particles_i00905.hdf5'

with h5py.File(my_file2, 'r') as hdf:
    data = hdf['particles']['0000000001']['attributes']
    print(list(data))

    particle_array = ParticleArray(name='a_name')
    for name in list(data):
        particle_array.add_property(name=name, data=data[name])
    particle_array.add_constant(name='h', data=data['h_smooth'])
    interp3d = interp.Interpolator([particle_array], num_points=125000)
    dens = interp3d.interpolate('density')
    pres = interp3d.interpolate('pressure')

    x = data['x']  # [2000:2500]
    y = data['y']  # [2000:2500]
    z = data['z']  # [2000:2500]
    u = data['vx']  # [2000:2500]
    v = data['vy']  # [2000:2500]
    w = data['vz']  # [2000:2500]
    rho = data['density']  # [2000:2500]

    fig = mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0))

    def grid_data_plot(data, dataname):

        log = True
        if log:
            grid_data = np.log10(data)
            dynamic_range = 6.0
            vmin = grid_data.max() - dynamic_range
            vmax = grid_data.max()
        else:
            grid_data = data
            vmin = np.min(grid_data)
            vmax = np.max(grid_data)

        #Removes the zeros problem
        grid_data[grid_data < vmin] = vmin
        grid_data[grid_data > vmax] = vmax

        sf = scalar_field(grid_data)
        V = mlab.pipeline.volume(sf, color=(1.0, 1.0, 1.0), vmin=vmin, vmax=vmax, name=dataname)
        print(type(V))
        V.trait_get('volume_mapper')['volume_mapper'].blend_mode = 'maximum_intensity'

        #Change the values below to change the color
        #Doing this manually with the mayavi app is hard for small values of vmin and vmax
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(vmin, 107. / 255, 124. / 255, 132. / 255)
        ctf.add_rgb_point(vmin + (vmax - vmin) * 0.8, 200. / 255, 178. / 255, 164. / 255)
        ctf.add_rgb_point(vmin + (vmax - vmin) * 0.9, 1.0, 210. / 255, 149. / 255)
        ctf.add_rgb_point(vmax, 1.0, 222. / 255, 141. / 255)

        V._volume_property.set_color(ctf)
        V._ctf = ctf
        V.update_ctf = True

        #Regulates opacity
        otf = PiecewiseFunction()
        otf.add_point(vmin, 0.0)
        otf.add_point(vmax, 1.0)

        V._volume_property.set_scalar_opacity(otf)
        V._otf = otf
        return V

    V1 = grid_data_plot(dens, 'density')
    V2 = grid_data_plot(pres, 'pressure')

    #3D contours
    mlab.contour3d(dens, contours=10, transparent=True)

    # #particle locations
    # mlab.points3d(x, y, z, scale_factor=2e15)

    # #particle velocities
    # mlab.quiver3d(x,y,z,u,v,w)
    mlab.show()
