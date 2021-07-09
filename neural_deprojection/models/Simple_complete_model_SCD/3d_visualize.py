import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

import tensorflow as tf
import sonnet as snt
from functools import partial
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, \
    build_log_dir, build_checkpoint_dir, get_distribution_strategy
from neural_deprojection.models.Simple_complete_model.model_utils import SimpleCompleteModel, VoxelisedModel
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.identify_medium_SCD.generate_data import decode_examples_old, decode_examples
import glob, os, json
from graph_nets.graphs import GraphsTuple
from scipy import interpolate
from neural_deprojection.graph_net_utils import histogramdd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# from mayavi import mlab


def build_dataset(data_dir, batch_size):
    dataset = _build_dataset(data_dir)

    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def _build_dataset(data_dir):
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))

    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(11,),
                                                             image_shape=(256, 256, 1),
                                                             k=6))  # (graph, image, spsh, proj)

    dataset = dataset.map(lambda graph_data_dict, img, spsh, proj, e: (GraphsTuple(**graph_data_dict), img))

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

    plt.tight_layout()
    plt.savefig('3d_vis.pdf')

    # if mayavi==True:
    #     mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 800))
    #     mlab.clf()
    #
    #     source_before = mlab.pipeline.scalar_field(data_before)
    #     source_after = mlab.pipeline.scalar_field(data_after)
    #
    #     # v_min_before = np.min(data_before) + 0.25 * (np.max(data_before) - np.min(data_before))
    #     # v_max_before = np.min(data_before) + 0.75 * (np.max(data_before) - np.min(data_before))
    #     # mlab.pipeline.volume(source_before, vmin=v_min_before, vmax=v_max_before)
    #     #
    #     # v_min_after = np.min(data_after) + 0.25 * (np.max(data_after) - np.min(data_after))
    #     # v_max_after = np.min(data_after) + 0.75 * (np.max(data_after) - np.min(data_after))
    #     # mlab.pipeline.volume(source_after, vmin=v_min_after, vmax=v_max_after)
    #
    #     mlab.pipeline.iso_surface(source_before, contours=16, opacity=0.4)
    #     mlab.pipeline.iso_surface(source_after, contours=16, opacity=0.4)
    #
    #     # mlab.pipeline.scalar_cut_plane(source_before, line_width=2.0, plane_orientation='z_axes')
    #     # mlab.pipeline.scalar_cut_plane(source_after, line_width=2.0, plane_orientation='z_axes')
    #
    #     mlab.view(180, 180, 100 * grid_res / 40, [0.5 * grid_res,
    #                                               0.5 * grid_res,
    #                                               0.5 * grid_res])
    #     mlab.show()


if __name__ == '__main__':
    data_dir = '/home/s1825216/data/dataset/test/'
    scm_cp_dir = 'single_voxelised_checkpointing'
    scm_saved_dir = 'single_voxelised_saved_models'

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

    _voxelised_model = VoxelisedModel(num_properties=1,
                                        # num_components=8,
                                        voxel_per_dimension=4,
                                        decoder_3d_hidden_size=4,
                                        component_size=16,
                                        num_embedding_3d=128,
                                        edge_size=4,
                                        global_size=16,
                                        num_heads=2,
                                        multi_head_output_size=16,
                                      discrete_image_vae=DiscreteImageVAE(hidden_size=64,
                                                                          embedding_dim=64,
                                                                          num_embedding=1024,
                                                                          num_channels=1,
                                                                          num_token_samples=4,),
                                      num_token_samples=2,
                                      n_node_per_graph=256,
                                      batch=1,
                                      name='voxelised_model')

    dataset = build_dataset(data_dir, batch_size=1)
    cluster = 0
    iter_ds = iter(dataset)

    for i in range(20):
        (graph, img) = next(iter_ds)
        if i == cluster:
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
                 prop_index=0,
                 grid_res=256,
                 component=None,
                 decode_on_interp_pos=True,
                 saved_model=False,
                 debug=True)
            break
