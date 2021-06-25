import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

import os
import tensorflow as tf
import glob
from neural_deprojection.models.Simple_complete_model_GCD.main import build_dataset
from functools import partial
from neural_deprojection.models.Simple_complete_model.model_utils import SimpleCompleteModel
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from mayavi import mlab
from scipy import interpolate
from neural_deprojection.graph_net_utils import histogramdd
import matplotlib.pyplot as plt
import numpy as np

def decode_property(scm_model,
                    img,
                    positions,
                    property_index,
                    component=None,
                    debug=False,
                    saved_model=True):
    if debug and not saved_model:
        properties = scm_model._im_to_components(img, positions, 10)  # [num_components, num_positions, num_properties]
    else:
        properties = scm_model.im_to_components(img, positions, 10)  # [num_components, num_positions, num_properties]

    if component is not None:
        properties_one_component = properties[component]  # [num_positions, num_properties]
    else:
        properties_one_component = tf.reduce_sum(properties, axis=0)  # [num_positions, num_properties]
    data_flat_after = properties_one_component[:, property_index]  # [num_positions] (3=density)
    return data_flat_after


def visualization_3d(scm_model,
                     image,
                     graph,
                     property_index,
                     grid_resolution,
                     component=None,
                     decode_on_interp_pos=True,
                     debug=False,
                     saved_model=True):
    x = tf.linspace(-1.7, 1.7, grid_resolution)[..., None]
    x, y, z = tf.meshgrid(x, x, x, indexing='ij')  # 3 x [grid_resolution, grid_resolution, grid_resolution]
    grid_positions = (x, y, z)  # [3, grid_resolution, grid_resolution, grid_resolution]

    node_positions = (graph.nodes[0, :, 0].numpy(),
                      graph.nodes[0, :, 1].numpy(),
                      graph.nodes[0, :, 2].numpy())  # [3, num_positions]

    # Interpolate 3D property to grid positions
    histogram_positions_before = graph.nodes[0, :, :2]
    prop = graph.nodes[0, :, 3 + property_index]  # [num_positions]
    interp_data_before = interpolate.griddata(node_positions,
                                              prop.numpy(),
                                              xi=grid_positions,
                                              method='linear',
                                              fill_value=0.)  # [grid_resolution, grid_resolution, grid_resolution]
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    z = tf.reshape(z, [-1])
    grid_positions_tensor = tf.concat([x[:, None], y[:, None], z[:, None]], axis=-1)  # [num_grid_positions, 3]

    if decode_on_interp_pos:
        # Directly calculate the decoded 3D property for the grid positions
        histogram_positions_after = grid_positions_tensor[:, :2]
        decoded_prop = decode_property(scm_model, image, grid_positions_tensor, property_index, component, debug, saved_model)
        interp_data_after = tf.reshape(decoded_prop, [grid_resolution, grid_resolution, grid_resolution]).numpy()
    else:
        # Calculate the decoded 3D property for the node positions and interpolate to grid positions
        histogram_positions_after = graph.nodes[0, :, :2]
        decoded_prop = decode_property(scm_model, image, graph.nodes[0, :, :3], property_index, component, debug, saved_model)
        interp_data_after = interpolate.griddata(node_positions,
                                                 decoded_prop.numpy(),
                                                 xi=grid_positions,
                                                 method='linear',
                                                 fill_value=0.)  # [grid_resolution, grid_resolution, grid_resolution]
    prop_interp = tf.convert_to_tensor(np.reshape(interp_data_after, (-1)))
    histogram_positions_interp = grid_positions_tensor[:, :2]

    # the reverse switches x and y, this way my images and histograms line up
    graph_hist_before, _ = histogramdd(tf.reverse(histogram_positions_before, [1]), bins=32, weights=prop)
    graph_hist_after, _ = histogramdd(tf.reverse(histogram_positions_after, [1]), bins=32, weights=decoded_prop)
    graph_hist_interp, _ = histogramdd(tf.reverse(histogram_positions_interp, [1]), bins=32, weights=prop_interp)

    return interp_data_before, interp_data_after, graph_hist_before, graph_hist_after, graph_hist_interp


def load_saved_models(scm_saved_model_dir):
    simple_complete_model = tf.saved_model.load(scm_saved_model_dir)
    discrete_image_vae = simple_complete_model.discrete_image_vae
    return simple_complete_model, discrete_image_vae


def load_checkpoint_models(scm_checkpoint_dir, scm_kwargs):
    simple_complete_model = SimpleCompleteModel(**scm_kwargs)
    encoder_cp = tf.train.Checkpoint(decoder=simple_complete_model.decoder,
                                     field_reconstruction=simple_complete_model.field_reconstruction,
                                     discrete_image_vae=simple_complete_model.discrete_image_vae)
    model_cp = tf.train.Checkpoint(_model=encoder_cp)
    checkpoint = tf.train.Checkpoint(module=model_cp)
    manager = tf.train.CheckpointManager(checkpoint, scm_checkpoint_dir, max_to_keep=3,
                                         checkpoint_name=simple_complete_model.__class__.__name__)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    return simple_complete_model, simple_complete_model.discrete_image_vae


def main(scm_saved_model_dir,
         scm_checkpoint_dir,
         scm_kwargs,
         image,
         input_graph,
         prop_index: int,
         grid_res: int,
         component=None,
         decode_on_interp_pos=True,
         saved_model=True,
         debug=False):
    if saved_model:
        simple_complete_model, discrete_image_vae = load_saved_models(scm_saved_model_dir)
    else:
        simple_complete_model, discrete_image_vae = load_checkpoint_models(scm_checkpoint_dir, scm_kwargs)

    data_before, data_after, image_before, image_after, image_interp = visualization_3d(scm_model=simple_complete_model,
                                                                                        image=image,
                                                                                        graph=input_graph,
                                                                                        property_index=prop_index,
                                                                                        grid_resolution=grid_res,
                                                                                        component=component,
                                                                                        decode_on_interp_pos=decode_on_interp_pos,
                                                                                        debug=debug,
                                                                                        saved_model=saved_model)
    if debug and not saved_model:
        decoded_img = discrete_image_vae._sample_decoder(discrete_image_vae._sample_encoder(img), 10, 4)
    else:
        decoded_img = discrete_image_vae.sample_decoder(discrete_image_vae.sample_encoder(img), 10, 4)

    H, xedges, yedges = np.histogram2d(graph.nodes[0, :, 0].numpy(),
                                       -graph.nodes[0, :, 1].numpy(),
                                       bins=([i for i in np.linspace(-1.7, 1.7, 32)],
                                             [i for i in np.linspace(-1.7, 1.7, 32)]))

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    graph_before = ax[0, 0].imshow(image_before.numpy())
    fig.colorbar(graph_before, ax=ax[0, 0])
    ax[0, 0].set_title('graph before')
    graph_after = ax[0, 1].imshow(image_after.numpy())
    fig.colorbar(graph_after, ax=ax[0, 1])
    ax[0, 1].set_title('graph after')
    graph_interp = ax[0, 2].imshow(image_interp.numpy())
    fig.colorbar(graph_interp, ax=ax[0, 2])
    ax[0, 2].set_title('graph interp')
    image_before = ax[1, 0].imshow(image[0, :, :, 1])
    fig.colorbar(image_before, ax=ax[1, 0])
    ax[1, 0].set_title('image before')
    image_after = ax[1, 1].imshow(decoded_img[0, 0, :, :, 1].numpy())
    fig.colorbar(image_after, ax=ax[1, 1])
    ax[1, 1].set_title('image after')
    hist2d = ax[1, 2].imshow(H.T, interpolation='nearest', origin='lower',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    fig.colorbar(hist2d, ax=ax[1, 2])
    ax[1, 2].set_title('particles per bin')
    plt.show()

    mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 800))
    mlab.clf()

    source_before = mlab.pipeline.scalar_field(data_before)
    source_after = mlab.pipeline.scalar_field(data_after)

    # v_min_before = np.min(data_before) + 0.25 * (np.max(data_before) - np.min(data_before))
    # v_max_before = np.min(data_before) + 0.75 * (np.max(data_before) - np.min(data_before))
    # mlab.pipeline.volume(source_before, vmin=v_min_before, vmax=v_max_before)
    #
    # v_min_after = np.min(data_after) + 0.25 * (np.max(data_after) - np.min(data_after))
    # v_max_after = np.min(data_after) + 0.75 * (np.max(data_after) - np.min(data_after))
    # mlab.pipeline.volume(source_after, vmin=v_min_after, vmax=v_max_after)

    # mlab.pipeline.iso_surface(source_before, contours=16, opacity=0.05)
    mlab.pipeline.iso_surface(source_after, contours=16, opacity=0.05)

    # mlab.pipeline.scalar_cut_plane(source_before, line_width=2.0, plane_orientation='z_axes')
    # mlab.pipeline.scalar_cut_plane(source_after, line_width=2.0, plane_orientation='z_axes')

    mlab.view(180, 180, 100 * grid_res / 40, [0.5 * grid_res,
                                              0.5 * grid_res,
                                              0.5 * grid_res])
    mlab.show()


if __name__ == '__main__':
    if os.getcwd().split('/')[2] == 's2675544':
        tfrec_base_dir = '/home/s2675544/data/tf_records'
        base_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD'
        # autoencoder_checkpoint_dir = os.path.join(base_dir, 'autoencoder_2d_checkpointing')
        scm_cp_dir = os.path.join(base_dir, 'simple_complete_checkpointing')
        # autoencoder_saved_model_dir = os.path.join(base_dir, 'autoencoder_2d_saved_model')
        scm_saved_dir = os.path.join(base_dir, 'simple_complete_saved_model')
        print('Running on ALICE')
    else:
        tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
        base_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD'
        # autoencoder_checkpoint_dir = os.path.join(base_dir, 'autoencoder_2d_checkpointing')
        scm_cp_dir = os.path.join(base_dir, 'simple_complete_checkpointing')
        # autoencoder_saved_model_dir = os.path.join(base_dir, 'autoencoder_2d_saved_model')
        scm_saved_dir = os.path.join(base_dir, 'simple_complete_saved_model')
        print('Running at home')

    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')
    tfrecords = glob.glob(os.path.join(tfrec_dir, 'train', '*.tfrecords'))
    dataset = build_dataset(tfrecords, batch_size=1)
    counter = 0
    thing = iter(dataset)
    (graph, img, cluster_id, proj_id) = next(thing)
    while cluster_id != 56:
        (graph, img, cluster_id, proj_id) = next(thing)
        counter += 1
        print(counter, cluster_id.numpy()[0], proj_id.numpy()[0])

    _scm_kwargs = dict(num_properties=7,
                       num_components=4,
                       component_size=128,
                       num_embedding_3d=512,
                       edge_size=8,
                       global_size=16,
                       n_node_per_graph=256,
                       discrete_image_vae=DiscreteImageVAE(embedding_dim=64,
                                                           num_embedding=1024,
                                                           hidden_size=64,
                                                           num_token_samples=4,
                                                           num_channels=2),
                       num_token_samples=2,
                       batch=2,
                       name='simple_complete_model')

    # property_index 0: vx, 1: vy, 2: vz, 3: rho, 4: U, 5: particle_mass, 6: smoothing length

    main(scm_saved_model_dir=scm_saved_dir,
         scm_checkpoint_dir=glob.glob(os.path.join(scm_cp_dir,'*'))[0],
         scm_kwargs=_scm_kwargs,
         image=img,
         input_graph=graph,
         prop_index=3,
         grid_res=32,
         component=0,
         decode_on_interp_pos=False,
         saved_model=False,
         debug=True)
