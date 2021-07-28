import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

import os
import tensorflow as tf
import glob
from neural_deprojection.models.Simple_complete_model_GCD.main import double_downsample
# from neural_deprojection.models.Simple_complete_model.model_utils import SimpleCompleteModel, VoxelisedModel
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoregressive_prior import AutoRegressivePrior
from mayavi import mlab
from scipy import interpolate
from neural_deprojection.graph_net_utils import histogramdd, get_shape, temperature_schedule, grid_graphs, grid_graph_smoothing
import matplotlib.pyplot as plt
import numpy as np
from graph_nets.graphs import GraphsTuple
from functools import partial
from tensorflow_addons.image import gaussian_filter2d
from neural_deprojection.models.identify_medium_GCD.generate_data import decode_examples
import json
import tensorflow_probability as tfp
import tensorflow_graphics as tfg

# tf.random.set_seed(1)

def tf_graph_dict_to_voxels(graph, voxels_per_dimension):
    tf_voxels = tf.py_function(graph_dict_to_voxels,
                               (graph, voxels_per_dimension),
                               tf.float32)
    return tf_voxels


def graph_dict_to_voxels(nodes, voxels_per_dimension):
    _x = tf.linspace(-1.7, 1.7, voxels_per_dimension)[..., None]
    x, y, z = tf.meshgrid(_x, _x, _x, indexing='ij')  # 3 x [grid_resolution, grid_resolution, grid_resolution]
    grid_positions = (x, y, z)  # [3, grid_resolution, grid_resolution, grid_resolution]

    node_positions = (nodes[:, 0].numpy(),
                      nodes[:, 1].numpy(),
                      nodes[:, 2].numpy())  # [3, num_positions]

    voxels = np.zeros((voxels_per_dimension,
                       voxels_per_dimension,
                       voxels_per_dimension,
                       2), dtype=np.float32)

    for i in range(2):
        prop = nodes[:, 3 + i].numpy() # [num_positions]
        voxels[..., i] = interpolate.griddata(node_positions,
                                              prop,
                                              xi=grid_positions,
                                              method='linear',
                                              fill_value=0.)  # [grid_resolution, grid_resolution, grid_resolution]

    return tf.convert_to_tensor(voxels, dtype=tf.float32)

def build_dataset(tfrecords, voxels_per_dimension, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(10,),
                                                             edge_shape=(2,),
                                                             image_shape=(256, 256, 1)))  # (graph, image, idx)

    dataset = dataset.map(lambda graph_data_dict,
                                 img,
                                 cluster_idx,
                                 projection_idx,
                                 vprime: (
        GraphsTuple(**graph_data_dict).replace(nodes=tf.concat([GraphsTuple(**graph_data_dict).nodes[:, :3],
                                                                GraphsTuple(**graph_data_dict).nodes[:, 6:8]],
                                                               axis=-1)),
        gaussian_filter2d(img), cluster_idx, projection_idx)).batch(batch_size=batch_size)

    # dataset = dataset.map(lambda graph_data_dict,
    #                              img,
    #                              cluster_idx,
    #                              projection_idx,
    #                              vprime: (tf_graph_dict_to_voxels(
    #     GraphsTuple(**graph_data_dict).replace(nodes=tf.concat([GraphsTuple(**graph_data_dict).nodes[:, :3],
    #                                                             GraphsTuple(**graph_data_dict).nodes[:, 6:8]],
    #                                                            axis=-1)).nodes, voxels_per_dimension),
    #                                       gaussian_filter2d(img), cluster_idx,
    #                              projection_idx)).batch(batch_size=batch_size)

    return dataset


def decode_property(model,
                    img,
                    graph,
                    positions,
                    property_index,
                    temperature,
                    component=None,
                    debug=False,
                    saved_model=True,
                    test_decoder=False):
    if model.name != 'auto_regressive_prior':
        if debug and not saved_model:
            decoded_properties, _ = model._im_to_components(img, tf.tile(positions[None, None, :, :], (model.num_token_samples, model.batch, 1, 1)), temperature)  # [num_components, num_positions, num_properties]
        else:
            decoded_properties = model.im_to_components(img, tf.tile(positions[None, None, :, :], (model.num_token_samples, model.batch, 1, 1)), temperature)  # [num_components, num_positions, num_properties]
    else:
        latent_logits_2d = model.discrete_image_vae.compute_logits(img)  # [batch, H, W, num_embeddings]
        log_token_samples_onehot_2d, token_samples_onehot_2d, latent_tokens_2d = model.discrete_image_vae.sample_latent(
            latent_logits_2d,
            0.01,
            model.num_token_samples)  # [num_samples, batch, H, W, num_embeddings], [num_samples, batch, H, W, embedding_size]

        latent_logits_3d = model.discrete_voxel_vae.compute_logits(graph)  # [batch, H, W, D, num_embeddings]
        log_token_samples_onehot_3d, token_samples_onehot_3d, latent_tokens_3d = model.discrete_voxel_vae.sample_latent(
            latent_logits_3d,
            0.01,
            model.num_token_samples)  # [num_samples, batch, H, W, D, num_embeddings], [num_samples, batch, H, W, D, embedding_size]

        mu_2d, _ = model.discrete_image_vae.compute_likelihood_parameters(latent_tokens_2d)
        if test_decoder:
            mu_3d, _ = model.discrete_voxel_vae.compute_likelihood_parameters(latent_tokens_3d)
        else:
            mu_3d, _ = model.deproject_images(img)

        decoded_properties = mu_3d
        decoded_img = mu_2d

        [batch3, H3, W3, D3, _] = get_shape(decoded_properties)
        decoded_properties = tf.reshape(decoded_properties, (batch3 * H3 * W3 * D3, 2))

    if model.name == 'simple_complete_model':
        if component is not None:
            properties_one_component = decoded_properties[component]  # [num_positions, num_properties]
        else:
            properties_one_component = tf.reduce_mean(decoded_properties, axis=0)  # [num_positions, num_properties]
        decoded_property = properties_one_component[:, property_index]  # [num_positions]
    else:
        decoded_property = decoded_properties[..., property_index]  # [num_positions]
    return decoded_property, decoded_img


def visualization_3d(model,
                     image,
                     graph,
                     property_index,
                     temperature,
                     component=None,
                     decode_on_interp_pos=True,
                     debug=False,
                     saved_model=True,
                     test_decoder=False):
    grid_resolution = model.discrete_voxel_vae.voxels_per_dimension
    _x = tf.linspace(-1.7, 1.7, grid_resolution)[..., None]
    x, y, z = tf.meshgrid(_x, _x, _x, indexing='ij')  # 3 x [grid_resolution, grid_resolution, grid_resolution]
    grid_positions = (x, y, z)  # [3, grid_resolution, grid_resolution, grid_resolution]

    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    z = tf.reshape(z, [-1])
    grid_positions_tensor = tf.concat([x[:, None], y[:, None], z[:, None]], axis=-1)  # [num_grid_positions, 3]



    # Interpolate 3D property to grid positions
    node_positions = (graph.nodes[0, :, 0].numpy(),
                      graph.nodes[0, :, 1].numpy(),
                      graph.nodes[0, :, 2].numpy())  # [3, num_positions]

    prop = graph.nodes[0, :, 3 + property_index]
    interp_data_before = interpolate.griddata(node_positions,
                                              prop.numpy(),
                                              xi=grid_positions,
                                              method='linear',
                                              fill_value=0.)  # [grid_resolution, grid_resolution, grid_resolution]

    # interp_data_before = grid_graphs(graph, model.discrete_voxel_vae.voxels_per_dimension)[0, :, :, :, property_index].numpy()
    histogram_positions_before = graph.nodes[0, :, :2]

    # nodes = graph
    # interp_data_before = nodes[0, :, :, :, property_index].numpy()  # [num_positions]
    # prop = tf.reshape(interp_data_before, [-1])
    # histogram_positions_before = grid_positions_tensor[:, :2]

    if decode_on_interp_pos:
        # Directly calculate the decoded 3D property for the grid positions
        histogram_positions_after = grid_positions_tensor[:, :2]
        decoded_prop, decoded_img = decode_property(model,
                                                    image,
                                                    graph,
                                                    grid_positions_tensor,
                                                    property_index,
                                                    temperature,
                                                    component,
                                                    debug=debug,
                                                    saved_model=saved_model,
                                                    test_decoder=test_decoder)
        interp_data_after = tf.reshape(decoded_prop, [grid_resolution, grid_resolution, grid_resolution]).numpy()
    else:
        # Calculate the decoded 3D property for the node positions and interpolate to grid positions
        histogram_positions_after = graph.nodes[0, :, :2]
        decoded_prop, decoded_img = decode_property(model,
                                                    image,
                                                    graph,
                                                    graph.nodes[0, :, :3],
                                                    property_index,
                                                    temperature,
                                                    component,
                                                    debug=debug,
                                                    saved_model=saved_model,
                                                    test_decoder=test_decoder)
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
           interp_hist_after, prop.numpy(), decoded_prop.numpy(), decoded_img.numpy()


def load_saved_models(saved_model_dir):
    model = tf.saved_model.load(saved_model_dir)
    return model


def load_checkpoint_models(checkpoint_dir, model, cp_kwargs):
    objects_cp = tf.train.Checkpoint(**cp_kwargs)
    model_cp = tf.train.Checkpoint(_model=objects_cp)
    checkpoint = tf.train.Checkpoint(module=model_cp)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3,
                                         checkpoint_name=model.__class__.__name__)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    return model


def main(saved_model_dir,
         checkpoint_dir,
         model,
         cp_kwargs,
         image,
         input_graph,
         temperature,
         prop_index: int,
         component=None,
         decode_on_interp_pos=True,
         saved_model=True,
         debug=False,
         test_decoder=False):
    if type(model) is not dict:
        if saved_model:
            simple_complete_model = load_saved_models(saved_model_dir)
        else:
            simple_complete_model = load_checkpoint_models(checkpoint_dir, model, cp_kwargs)

    grid_res = model.discrete_voxel_vae.voxels_per_dimension

    data_before, \
    data_after, \
    hist_before, \
    hist_after, \
    image_interp_before, \
    image_interp_after,\
    prop, \
    decoded_prop, \
    decoded_img = visualization_3d(model=model,
                                   image=image,
                                   graph=input_graph,
                                   property_index=prop_index,
                                   temperature=temperature,
                                   component=component,
                                   decode_on_interp_pos=decode_on_interp_pos,
                                   debug=debug,
                                   saved_model=saved_model,
                                   test_decoder=test_decoder)

    if model.name != 'auto_regressive_prior':
        if debug and not saved_model:
            decoded_img = model.discrete_image_vae._sample_decoder(
                model.discrete_image_vae._sample_encoder(img), temperature, 1)
        else:
            decoded_img = model.discrete_image_vae.sample_decoder(
                model.discrete_image_vae.sample_encoder(img), temperature, 4)

    # H, xedges, yedges = np.histogram2d(input_nodes[0, :, 0].numpy(),
    #                                    -input_nodes[0, :, 1].numpy(),
    #                                    bins=([i for i in np.linspace(-1.7, 1.7, grid_res)],
    #                                          [i for i in np.linspace(-1.7, 1.7, grid_res)]))

    fig, ax = plt.subplots(2, 4, figsize=(24, 12))

    # image channel 1 is smoothed image
    image_before = ax[0, 0].imshow(image[0, :, :, 0])
    fig.colorbar(image_before, ax=ax[0, 0])
    ax[0, 0].set_title('input image')

    # decoded_img [S, batch, H, W, channels]
    # image channel 1 is decoded from the smoothed image
    image_after = ax[1, 0].imshow(decoded_img[0, 0, :, :, 0])
    fig.colorbar(image_after, ax=ax[1, 0])
    ax[1, 0].set_title('decoded image')

    graph_before = ax[0, 1].imshow(hist_before.numpy())
    fig.colorbar(graph_before, ax=ax[0, 1])
    ax[0, 1].set_title('property histogram')

    graph_after = ax[1, 1].imshow(hist_after.numpy())
    fig.colorbar(graph_after, ax=ax[1, 1])
    ax[1, 1].set_title('reconstructed property histogram')

    # hist2d = ax[0, 2].imshow(H.T, interpolation='nearest', origin='lower',
    #                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # fig.colorbar(hist2d, ax=ax[0, 2])
    # ax[0, 2].set_title('particles per histogram bin')

    ax[0, 2].hist(prop, bins=32, histtype='step', label='input data')
    ax[0, 2].hist(decoded_prop, bins=32, histtype='step', label='reconstructed')

    prop = data_before.flatten()

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

    mlab.pipeline.scalar_cut_plane(source_before, line_width=2.0, plane_orientation='z_axes')
    mlab.pipeline.scalar_cut_plane(source_after, line_width=2.0, plane_orientation='z_axes')

    mlab.view(180, 180, 100 * grid_res / 40, [0.5 * grid_res,
                                              0.5 * grid_res,
                                              0.5 * grid_res])
    mlab.show()


if __name__ == '__main__':
    if os.getcwd().split('/')[2] == 's2675544':
        tfrec_base_dir = '/home/s2675544/data/tf_records'
        base_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model'
        print('Running on ALICE')
    else:
        tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
        base_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model'
        print('Running at home')

    scm_cp_dir = os.path.join(base_dir, 'checkpointing')
    scm_saved_dir = os.path.join(base_dir, 'saved_model')

    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_136_tf_records')
    tfrecords = glob.glob(os.path.join(tfrec_dir, 'train', '*.tfrecords'))
    dataset = build_dataset(tfrecords, 4*16, batch_size=1)
    counter = 0
    thing = iter(dataset)
    (graph, img, cluster_id, proj_id) = next(thing)
    datapoint = 543
     # 66
     # 543
    while counter < datapoint:
        (graph, img, cluster_id, proj_id) = next(thing)
        counter += 1

    discrete_image_vae_checkpoint = glob.glob(os.path.join(scm_cp_dir, '*disc_image_vae*'))[0]
    discrete_voxel_vae_checkpoint = glob.glob(os.path.join(scm_cp_dir, '*disc_voxel_vae*'))[0]
    auto_regressive_prior_checkpoint = glob.glob(os.path.join(scm_cp_dir, '*auto_regressive_prior*'))[0]

    with open(os.path.join(discrete_image_vae_checkpoint, 'config.json'), 'r') as f:
        discrete_image_vae_kwargs = json.load(f)['model_parameters']
        discrete_image_vae = DiscreteImageVAE(**discrete_image_vae_kwargs)
        checkpoint = tf.train.Checkpoint(module=discrete_image_vae)
        manager = tf.train.CheckpointManager(checkpoint, discrete_image_vae_checkpoint, max_to_keep=3,
                                             checkpoint_name=discrete_image_vae.__class__.__name__)
        if manager.latest_checkpoint is not None:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(f"Restored from {manager.latest_checkpoint}")

    with open(os.path.join(discrete_voxel_vae_checkpoint, 'config.json'), 'r') as f:
        discrete_voxel_vae_kwargs = json.load(f)['model_parameters']
        discrete_voxel_vae = DiscreteVoxelsVAE(**discrete_voxel_vae_kwargs)
        checkpoint = tf.train.Checkpoint(module=discrete_voxel_vae)
        manager = tf.train.CheckpointManager(checkpoint, discrete_voxel_vae_checkpoint, max_to_keep=3,
                                             checkpoint_name=discrete_voxel_vae.__class__.__name__)
        if manager.latest_checkpoint is not None:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(f"Restored from {manager.latest_checkpoint}")

    with open(os.path.join(auto_regressive_prior_checkpoint, 'config.json'), 'r') as f:
        auto_regressive_prior_kwargs = json.load(f)['model_parameters']
        auto_regressive_prior = AutoRegressivePrior(discrete_image_vae=discrete_image_vae,
                                                    discrete_voxel_vae=discrete_voxel_vae,
                                                    **auto_regressive_prior_kwargs)
        checkpoint = tf.train.Checkpoint(module=auto_regressive_prior)
        manager = tf.train.CheckpointManager(checkpoint, auto_regressive_prior_checkpoint, max_to_keep=3,
                                             checkpoint_name=auto_regressive_prior.__class__.__name__)
        if manager.latest_checkpoint is not None:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(f"Restored from {manager.latest_checkpoint}")


    main(saved_model_dir=scm_saved_dir,
         checkpoint_dir=glob.glob(os.path.join(scm_cp_dir, '*'))[0],
         model=auto_regressive_prior,
         cp_kwargs=dict(#autoregressive_prior=_voxelised_model.autoregressive_prior,
                        #decoder_3d=_voxelised_model.decoder_3d,
                        #discrete_image_vae=_voxelised_model.discrete_image_vae
                        ),
         image=img,
         input_graph=graph,
         temperature=1.,
         prop_index=0,
         component=None,
         decode_on_interp_pos=True,
         saved_model=False,
         debug=True,
         test_decoder=False)
