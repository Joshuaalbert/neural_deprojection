import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

import os
import tensorflow as tf
import glob
from neural_deprojection.models.Simple_complete_model_GCD.main import double_downsample
from neural_deprojection.models.Simple_complete_model.model_utils import SimpleCompleteModel, VoxelisedModel
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from mayavi import mlab
from scipy import interpolate
from neural_deprojection.graph_net_utils import histogramdd
import matplotlib.pyplot as plt
import numpy as np
from graph_nets.graphs import GraphsTuple
from functools import partial
from tensorflow_addons.image import gaussian_filter2d
from neural_deprojection.models.identify_medium_GCD.generate_data import decode_examples


def build_dataset(tfrecords, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(10,),
                                                             edge_shape=(2,),
                                                             image_shape=(1024, 1024, 1)))  # (graph, image, idx)

    dataset = dataset.map(lambda graph_data_dict,
                                 img,
                                 cluster_idx,
                                 projection_idx,
                                 vprime: (GraphsTuple(**graph_data_dict),
                                          tf.concat([double_downsample(img),
                                                     gaussian_filter2d(double_downsample(img),
                                                                       filter_shape=[6, 6])],
                                                    axis=-1), cluster_idx, projection_idx)).shuffle(buffer_size=50).batch(batch_size=batch_size)
    return dataset


def decode_property(scm_model,
                    img,
                    positions,
                    property_index,
                    temperature,
                    component=None,
                    debug=False,
                    saved_model=True):
    if debug and not saved_model:
        decoded_properties, _ = scm_model._im_to_components(img, tf.tile(positions[None, None, :, :], (scm_model.num_token_samples, scm_model.batch, 1, 1)), temperature)  # [num_components, num_positions, num_properties]
    else:
        decoded_properties = scm_model.im_to_components(img, tf.tile(positions[None, None, :, :], (scm_model.num_token_samples, scm_model.batch, 1, 1)), temperature)  # [num_components, num_positions, num_properties]

    if scm_model.name == 'simple_complete_model':
        if component is not None:
            properties_one_component = decoded_properties[component]  # [num_positions, num_properties]
        else:
            properties_one_component = tf.reduce_mean(decoded_properties, axis=0)  # [num_positions, num_properties]
        decoded_property = properties_one_component[:, property_index]  # [num_positions]
    else:
        decoded_property = decoded_properties[:, property_index]  # [num_positions]
    return decoded_property


def visualization_3d(scm_model,
                     image,
                     graph,
                     property_index,
                     grid_resolution,
                     temperature,
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
        decoded_prop = decode_property(scm_model, image, grid_positions_tensor, property_index, temperature, component, debug, saved_model)
        interp_data_after = tf.reshape(decoded_prop, [grid_resolution, grid_resolution, grid_resolution]).numpy()
    else:
        # Calculate the decoded 3D property for the node positions and interpolate to grid positions
        histogram_positions_after = graph.nodes[0, :, :2]
        decoded_prop = decode_property(scm_model, image, graph.nodes[0, :, :3], property_index, temperature, component, debug, saved_model)
        interp_data_after = interpolate.griddata(node_positions,
                                                 decoded_prop.numpy(),
                                                 xi=grid_positions,
                                                 method='linear',
                                                 fill_value=0.)  # [grid_resolution, grid_resolution, grid_resolution]

    prop_interp_before = tf.convert_to_tensor(np.reshape(interp_data_before, (-1)))
    prop_interp_after = tf.convert_to_tensor(np.reshape(interp_data_after, (-1)))
    histogram_positions_interp = grid_positions_tensor[:, :2]

    # decoded_prop = tf.random.uniform((10000,))
    # the reverse switches x and y, this way my images and histograms line up
    graph_hist_before, _ = histogramdd(tf.reverse(histogram_positions_before, [1]), bins=grid_resolution, weights=prop)
    graph_hist_after, _ = histogramdd(tf.reverse(histogram_positions_after, [1]), bins=grid_resolution, weights=decoded_prop)

    interp_hist_before, _ = histogramdd(tf.reverse(histogram_positions_interp, [1]), bins=grid_resolution, weights=prop_interp_before)
    interp_hist_after, _ = histogramdd(tf.reverse(histogram_positions_interp, [1]), bins=grid_resolution, weights=prop_interp_after)

    return interp_data_before, \
           interp_data_after, \
           graph_hist_before, \
           graph_hist_after, \
           interp_hist_before, \
           interp_hist_after, prop.numpy(), decoded_prop.numpy()


def load_saved_models(scm_saved_model_dir):
    simple_complete_model = tf.saved_model.load(scm_saved_model_dir)
    return simple_complete_model


def load_checkpoint_models(scm_checkpoint_dir, model, cp_kwargs):
    objects_cp = tf.train.Checkpoint(**cp_kwargs)
    model_cp = tf.train.Checkpoint(_model=objects_cp)
    checkpoint = tf.train.Checkpoint(module=model_cp)
    manager = tf.train.CheckpointManager(checkpoint, scm_checkpoint_dir, max_to_keep=3,
                                         checkpoint_name=model.__class__.__name__)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    return model


def main(scm_saved_model_dir,
         scm_checkpoint_dir,
         scm_model,
         cp_kwargs,
         image,
         input_graph,
         temperature,
         prop_index: int,
         grid_res: int,
         component=None,
         decode_on_interp_pos=True,
         saved_model=True,
         debug=False):
    if saved_model:
        simple_complete_model = load_saved_models(scm_saved_model_dir)
    else:
        simple_complete_model = load_checkpoint_models(scm_checkpoint_dir, scm_model, cp_kwargs)

    data_before, \
    data_after, \
    hist_before, \
    hist_after, \
    image_interp_before, \
    image_interp_after,\
    prop, \
    decoded_prop = visualization_3d(scm_model=simple_complete_model,
                                    image=image,
                                    graph=input_graph,
                                    property_index=prop_index,
                                    grid_resolution=grid_res,
                                    temperature=temperature,
                                    component=component,
                                    decode_on_interp_pos=decode_on_interp_pos,
                                    debug=debug,
                                    saved_model=saved_model)

    if debug and not saved_model:
        decoded_img = simple_complete_model.discrete_image_vae._sample_decoder(
            simple_complete_model.discrete_image_vae._sample_encoder(img), temperature, 1)
    else:
        decoded_img = simple_complete_model.discrete_image_vae.sample_decoder(
            simple_complete_model.discrete_image_vae.sample_encoder(img), temperature, 4)

    H, xedges, yedges = np.histogram2d(graph.nodes[0, :, 0].numpy(),
                                       -graph.nodes[0, :, 1].numpy(),
                                       bins=([i for i in np.linspace(-1.7, 1.7, grid_res)],
                                             [i for i in np.linspace(-1.7, 1.7, grid_res)]))

    fig, ax = plt.subplots(2, 4, figsize=(24, 12))

    # image channel 1 is smoothed image
    image_before = ax[0, 0].imshow(image[0, :, :, 1])
    fig.colorbar(image_before, ax=ax[0, 0])
    ax[0, 0].set_title('input image')

    # decoded_img [S, batch, H, W, channels]
    # image channel 1 is decoded from the smoothed image
    image_after = ax[1, 0].imshow(decoded_img[0, 0, :, :, 1].numpy())
    fig.colorbar(image_after, ax=ax[1, 0])
    ax[1, 0].set_title('decoded image')

    graph_before = ax[0, 1].imshow(hist_before.numpy())
    fig.colorbar(graph_before, ax=ax[0, 1])
    ax[0, 1].set_title('property histogram')

    graph_after = ax[1, 1].imshow(hist_after.numpy())
    fig.colorbar(graph_after, ax=ax[1, 1])
    ax[1, 1].set_title('reconstructed property histogram')

    hist2d = ax[0, 2].imshow(H.T, interpolation='nearest', origin='lower',
                             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    fig.colorbar(hist2d, ax=ax[0, 2])
    ax[0, 2].set_title('particles per histogram bin')

    ax[1, 2].hist(prop, bins=32, histtype='step', label='input data')
    ax[1, 2].hist(decoded_prop, bins=32, histtype='step', label='reconstructed')

    offset = (np.min(decoded_prop) * np.max(prop) - np.max(decoded_prop) * np.min(prop)) / (np.max(prop) - np.min(prop))
    scale = np.max(prop) / np.max(decoded_prop - offset)

    ax[1, 2].hist((decoded_prop - offset) * scale, bins=32, histtype='step', label='reconstructed scaled')
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_xlabel('property value')
    ax[1, 2].set_ylabel('counts')
    ax[1, 2].set_title('property value distributions')
    ax[1, 2].legend()

    interp_before = ax[0, 3].imshow(image_interp_before.numpy())
    fig.colorbar(interp_before, ax=ax[0, 3])
    ax[0, 3].set_title('property interpolated to grid points')

    interp_after = ax[1, 3].imshow(image_interp_after.numpy())
    fig.colorbar(interp_after, ax=ax[1, 3])
    ax[1, 3].set_title('reconstructed property interpolated to grid points')

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

    mlab.pipeline.iso_surface(source_before, contours=16, opacity=0.4)
    mlab.pipeline.iso_surface(source_after, contours=16, opacity=0.4)

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
        print('Running on ALICE')
    else:
        tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
        base_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD'
        print('Running at home')

    scm_cp_dir = os.path.join(base_dir, 'voxelised_model_checkpointing')
    scm_saved_dir = os.path.join(base_dir, 'voxelised_model_saved')

    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')
    tfrecords = glob.glob(os.path.join(tfrec_dir, 'train', '*.tfrecords'))
    dataset = build_dataset(tfrecords, batch_size=1)
    counter = 0
    thing = iter(dataset)
    (graph, img, cluster_id, proj_id) = next(thing)
    # 56 is a cluster that has two clear blobs which is nice for comparisons
    while cluster_id != 56:
        (graph, img, cluster_id, proj_id) = next(thing)
        counter += 1
        print(counter, cluster_id.numpy()[0], proj_id.numpy()[0])

    # _simple_complete_model = SimpleCompleteModel(num_properties=7,
    #                                              num_components=8,
    #                                              component_size=64,
    #                                              num_embedding_3d=512,
    #                                              edge_size=8,
    #                                              global_size=16,
    #                                              n_node_per_graph=256,
    #                                              num_heads=4,
    #                                              multi_head_output_size=64,
    #                                              discrete_image_vae=DiscreteImageVAE(embedding_dim=64,
    #                                                                                  num_embedding=1024,
    #                                                                                  hidden_size=64,
    #                                                                                  num_token_samples=4,
    #                                                                                  num_channels=2),
    #                                              num_token_samples=2,
    #                                              batch=2,
    #                                              name='simple_complete_model')

    _voxelised_model = VoxelisedModel(num_properties=7,
                                      voxel_per_dimension=4,
                                      component_size=16,
                                      num_embedding_3d=1024,
                                      edge_size=8,
                                      global_size=16,
                                      n_node_per_graph=256,
                                      num_heads=2,
                                      multi_head_output_size=16,
                                      decoder_3d_hidden_size=4,
                                      discrete_image_vae=DiscreteImageVAE(embedding_dim=64,
                                                                          num_embedding=1024,
                                                                          hidden_size=64,
                                                                          num_token_samples=4,
                                                                          num_channels=2),
                                      num_token_samples=2,
                                      batch=1,
                                      name='voxelised_model')

    # property_index 0: vx, 1: vy, 2: vz, 3: rho, 4: U, 5: particle_mass, 6: smoothing length

    main(scm_saved_model_dir=scm_saved_dir,
         scm_checkpoint_dir=glob.glob(os.path.join(scm_cp_dir, '*'))[0],
         scm_model=_voxelised_model,
         cp_kwargs=dict(autoregressive_prior=_voxelised_model.autoregressive_prior,
                        decoder_3d=_voxelised_model.decoder_3d,
                        discrete_image_vae=_voxelised_model.discrete_image_vae
                        ),
         image=img,
         input_graph=graph,
         temperature=10.,
         prop_index=3,
         grid_res=64,
         component=None,
         decode_on_interp_pos=True,
         saved_model=False,
         debug=True)
