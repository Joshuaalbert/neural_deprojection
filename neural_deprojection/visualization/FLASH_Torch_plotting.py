import yt
import numpy as np
yt.funcs.mylog.setLevel(40) # Surpresses YT status output.

folder_path = '~/Desktop/SCD/SeanData/'
filename = 'turbsph_hdf5_plt_cnt_3136'  # only plt file, will automatically find part file
# folder_path = '~/Desktop/SCD/ClaudeData/M4r6b/'
# filename = 'turbsph_hdf5_part_0265'

file_path = folder_path + filename
print(file_path)
ds = yt.load(file_path) # loads in data into data set class. This is what we will use to plot field values
ad = ds.all_data() # Can call on the data set's property .all_data() to generate a dictionary
                   # containing all data available to be parsed through.
                   # e.g. print ad['mass'] will print the list of all cell masses.
                   # if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]

# print(ad['temperature'])

# To see all the possible field values contained within the hydro (plt) and particle (part) files:
for e in ds.derived_field_list:
    print(e)
ds.print_stats()
# print(ds.derived_field_list)
# print("-"*20 + "\n", ad['x'])
# print(len(ad['x']))
# print(len(ad['temperature']))


# Say I want to make a slice plot of the density field.
field ='density' # dens, temp, and pres are some shorthand strings recognized by yt.
# ax = 'y' # the axis our slice plot will be "looking down on".

L = [1,0,0] # vector normal to cutting plane
v_elements = [-1, 0, 1]
folder_path = '~/Desktop/SCD/SeanData/offaxisplots/'

for x_e in v_elements:
    for y_e in v_elements:
        for z_e in v_elements:
            if x_e == y_e == z_e == 0: continue
            L = [x_e, y_e, z_e]
            plot_ = yt.OffAxisProjectionPlot(ds, L, field)
            im_name = folder_path + '{}_offaxis_testplot1.png'.format(str(x_e)+str(y_e)+str(z_e))
            plot_.save(im_name)
# plot_ = yt.SlicePlot(ds, ax, field)
# plot_ = yt.ProjectionPlot(ds, ax, field)
# plot_.set_cmap(field, "binary")
#
# plot_.annotate_timestamp()
# plot_.annotate_grids()
#
# plot_.annotate_title('Testplot')
# plot_.annotate_scale()

#plot_.annotate_magnetic_field()

#plot_.show()





