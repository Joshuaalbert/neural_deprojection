import numpy as np
import os, glob
import pylab as plt
from matplotlib.widgets import Slider

def plot_voxel(image, rec_voxels, actual_voxels):
    fig, axs = plt.subplots(1, 3)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    axs[0].imshow(image[:,:,0])
    axs[0].set_title('Image')
    img_actual = axs[1].imshow(actual_voxels[:, :, 0, 0])
    axs[1].margins(x=0)
    axs[2].set_title("Actual voxels")
    img_rec = axs[2].imshow(rec_voxels[:, :, 0, 0])
    axs[2].margins(x=0)
    axs[2].set_title("Reconstructed voxels")

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'z-axis slice', 0, actual_voxels.shape[-2]-1, valinit=0, valstep=1)

    def update(val):
        idx = int(sfreq.val)
        img_actual.set_data(actual_voxels[:, :, idx, 0])
        img_rec.set_data(rec_voxels[:, :, idx, 0])
        fig.canvas.draw_idle()

    sfreq.on_changed(update)
    plt.show()

def main(eval_dir):
    files = glob.glob(os.path.join(eval_dir, '*.npz'))
    for f in files:
        image = np.load(f)['image']
        mu_3d = np.load(f)['mu_3d']
        actual_voxels = np.load(f)['actual_voxels']

        plot_voxel(image, mu_3d, actual_voxels)
        # Visualize the local atomic density


        # break

if __name__ == '__main__':
    main("/home/albert/data/deprojections")