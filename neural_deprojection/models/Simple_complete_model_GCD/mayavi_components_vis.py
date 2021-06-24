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


if os.getcwd().split('/')[2] == 's2675544':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    base_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD'
    autoencoder_checkpoint_dir = os.path.join(base_dir, 'autoencoder_2d_checkpointing')
    scm_checkpoint_dir = os.path.join(base_dir, 'simple_complete_checkpointing')
    autoencoder_saved_model_dir = os.path.join(base_dir, 'autoencoder_2d_saved_model')
    scm_saved_model_dir = os.path.join(base_dir, 'simple_complete_saved_model')
    print('Running on ALICE')
else:
    tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
    base_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD'
    autoencoder_checkpoint_dir = os.path.join(base_dir, 'autoencoder_2d_checkpointing')
    scm_checkpoint_dir = os.path.join(base_dir, 'simple_complete_checkpointing')
    autoencoder_saved_model_dir = os.path.join(base_dir, 'autoencoder_2d_saved_model')
    scm_saved_model_dir = os.path.join(base_dir, 'simple_complete_saved_model')
    print('Running at home')

tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')
tfrecords = glob.glob(os.path.join(tfrec_dir, 'train', '*.tfrecords'))
dataset = build_dataset(tfrecords, batch_size=1)
(graph, img) = next(iter(dataset))

# discrete_image_vae = DiscreteImageVAE(embedding_dim=64,  # 64
#                                       num_embedding=1024,  # 1024
#                                       hidden_size=64,  # 64
#                                       num_token_samples=4,  # 4
#                                       num_channels=2)
#
# # dummy = discrete_image_vae.encoder(tf.zeros((1, 256, 256, 2)))
# # print(discrete_image_vae.encoder.trainable_variables)
#
# encoder_cp = tf.train.Checkpoint(encoder=discrete_image_vae.encoder,
#                                  decoder=discrete_image_vae.decoder)
# model_cp = tf.train.Checkpoint(_model=encoder_cp)
# checkpoint = tf.train.Checkpoint(module=model_cp)
# status = tf.train.latest_checkpoint(autoencoder_checkpoint_dir)
# checkpoint.restore(status).expect_partial()


# simple_complete_model = SimpleCompleteModel(num_properties=7,
#                                             num_components=4,
#                                             component_size=128,
#                                             num_embedding_3d=512,
#                                             edge_size=8,
#                                             global_size=16,
#                                             discrete_image_vae=discrete_image_vae,
#                                             num_token_samples=1,
#                                             batch=1,
#                                             beta=6.6,
#                                             name=None)
# encoder_cp = tf.train.Checkpoint(decoder=simple_complete_model.decoder,
#                                  field_reconstruction=simple_complete_model.field_reconstruction)
# model_cp = tf.train.Checkpoint(_model=encoder_cp)
# checkpoint = tf.train.Checkpoint(module=model_cp)
# status = tf.train.latest_checkpoint(scm_checkpoint_dir)
# checkpoint.restore(status).expect_partial()


def decode_property(scm_model, img, graph, property_index, component=None):
    properties = scm_model.im_to_components(img, graph.nodes[0, :, :3],
                                                        10)  # [num_components, num_positions, num_properties]

    if component is not None:
        properties_one_component = properties[component]  # [num_positions, num_properties]
    else:
        properties_one_component = tf.reduce_sum(properties, axis=0)  # [num_positions, num_properties]
    data_flat_after = properties_one_component[:, property_index]  # [num_positions] (3=density)
    return data_flat_after


def graph_to_graph(scm_model, img, graph, property_index, grid_resolution, component=None):
    x = tf.linspace(-1.7, 1.7, grid_resolution)[..., None]
    x, y, z = tf.meshgrid(x, x, x, indexing='ij')  # 3 x [grid_resolution, grid_resolution, grid_resolution]
    interp_pos = (x, y, z)  # [3, grid_resolution, grid_resolution, grid_resolution]
    positions = (graph.nodes[0, :, 0].numpy(),
                 graph.nodes[0, :, 1].numpy(),
                 graph.nodes[0, :, 2].numpy())  # [3, num_positions]
    positions_2d = graph.nodes[0, :, :2]

    prop = graph.nodes[0, :, 3 + property_index]  # [num_positions]
    decoded_prop = decode_property(scm_model, img, graph, property_index, component)

    # the reverse switches x and y, this way my images and histograms line up
    graph_hist_before, _ = histogramdd(tf.reverse(positions_2d, [1]), bins=64, weights=prop)
    graph_hist_after, _ = histogramdd(tf.reverse(positions_2d, [1]), bins=64, weights=decoded_prop)

    interp_data_before = interpolate.griddata(positions,
                                              prop.numpy(),
                                              xi=interp_pos,
                                              method='linear',
                                              fill_value=0.)  # [grid_resolution, grid_resolution, grid_resolution]
    interp_data_after = interpolate.griddata(positions,
                                             decoded_prop.numpy(),
                                             xi=interp_pos,
                                             method='linear',
                                             fill_value=0.)  # [grid_resolution, grid_resolution, grid_resolution]

    return interp_data_before, interp_data_after, graph_hist_before, graph_hist_after

prop_index = 3
grid_res = 40
component = 1

simple_complete_model = tf.saved_model.load(scm_saved_model_dir)
data_before, data_after, image_before, image_after = graph_to_graph(scm_model=simple_complete_model,
                                                                    img=img,
                                                                    graph=graph,
                                                                    property_index=prop_index,
                                                                    grid_resolution=grid_res,
                                                                    component=component)

discrete_image_vae = tf.saved_model.load(autoencoder_saved_model_dir)
decoded_img = discrete_image_vae.sample_decoder(discrete_image_vae.sample_encoder(img), 10, 4)

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0, 0].imshow(image_before.numpy())
ax[0, 0].set_title('graph before')
ax[0, 1].imshow(image_after.numpy())
ax[0, 1].set_title('graph after')
ax[1, 0].imshow(img[0, :, :, 1])
ax[1, 0].set_title('image before')
ax[1, 1].imshow(decoded_img[0, :, :, 1].numpy())
ax[1, 1].set_title('image after')
plt.show()

mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 800))
mlab.clf()

source_before = mlab.pipeline.scalar_field(data_before)
source_after = mlab.pipeline.scalar_field(data_after)

# v_min_before = np.min(data_before) + 0.25 * np.max(data_before) - np.min(data_before)
# v_max_before = np.min(data_before) + 0.75 * np.max(data_before) - np.min(data_before)
# vol_before = mlab.pipeline.volume(source_before, vmin=v_min_before, vmax=v_max_before)

# v_min_after = np.min(data_after) + 0.25 * np.max(data_after) - np.min(data_after)
# v_max_after = np.min(data_after) + 0.75 * np.max(data_after) - np.min(data_after)
# vol_after = mlab.pipeline.volume(source_after, vmin=v_min_after, vmax=v_max_after)

# iso_before = mlab.pipeline.iso_surface(source_before, contours=20, opacity=0.05)
# iso_after = mlab.pipeline.iso_surface(source_after, contours=20, opacity=0.05)

scalar_cut_plane_before = mlab.pipeline.scalar_cut_plane(source_before, line_width=2.0, plane_orientation='z_axes')
scalar_cut_plane_after = mlab.pipeline.scalar_cut_plane(source_after, line_width=2.0, plane_orientation='z_axes')


mlab.view(180, 180, 100 * grid_res / 40, [0.5 * grid_res,
                                          0.5 * grid_res,
                                          0.5 * grid_res])

mlab.show()
