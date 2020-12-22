import yt
yt.funcs.mylog.setLevel(40) # Surpresses YT status output.

folder_path = '~/Desktop/SCD/SeanData/'
filename = 'turbsph_hdf5_plt_cnt_2117'  # only plt file, will automatically find part file

file_path = folder_path + filename
print(file_path)
ds = yt.load(file_path) # loads in data into data set class. This is what we will use to plot field values
ad = ds.all_data() # Can call on the data set's property .all_data() to generate a dictionary
                   # containing all data available to be parsed through.
                   # e.g. print ad['mass'] will print the list of all cell masses.
                   # if particles exist, print ad['particle_position'] gives a list of each particle [x,y,z]

print(ad['x'])

# To see all the possible field values contained within the hydro (plt) and particle (part) files:
#print(ds.derived_field_list)

# Say I want to make a slice plot of the density field.
field ='pressure' # dens, temp, and pres are some shorthand strings recognized by yt.
ax = 'x' # the axis our slice plot will be "looking down on".
# plot_ = yt.SlicePlot(ds, ax, field)
plot_ = yt.ProjectionPlot(ds, ax, field)
plot_.set_cmap(field, "binary")

plot_.annotate_timestamp()
#plot_.annotate_grids()

plot_.annotate_title('Testplot')
plot_.annotate_scale()
#plot_.annotate_magnetic_field()

plot_.save('~/Desktop/SCD/SeanData/testplot2.png')
#plot_.show()





