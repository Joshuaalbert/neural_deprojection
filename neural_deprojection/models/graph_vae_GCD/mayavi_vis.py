from mayavi import mlab
import numpy as np
import glob, os



def main(eval_dir, plot_property):
    data_files = glob.glob(os.path.join(eval_dir, '*.npz'))

    for data_file in data_files:
        data = np.load(data_file)
        positions = data['positions']
        input_prop = data[f'prop_{plot_property}_input']
        decoded_prop = data[f'prop_{plot_property}_decoded']

        for prop, source in zip([input_prop, decoded_prop],['input','decoded']):
            mlab.figure(1, bgcolor=(0, 0, 0))
            mlab.clf()
            colors = prop
            pts = mlab.points3d(positions[:,0], positions[:,1], positions[:,2], colors,
                                scale_factor=0.015, resolution=10, scale_mode='none')
            mlab.savefig(f'{data_file.replace(".npz",f"{source}.png")}')
            mlab.show()
            # exit(0)

if __name__ == '__main__':
    main('output_evaluations', 'rho')