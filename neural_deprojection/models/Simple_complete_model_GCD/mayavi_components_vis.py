import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

import os
import tensorflow as tf
import glob
from graph_nets.graphs import GraphsTuple
from neural_deprojection.models.identify_medium_GCD.generate_data import decode_examples
from neural_deprojection.models.Simple_complete_model_GCD.main import double_downsample
from tensorflow_addons.image import gaussian_filter2d
from functools import partial
from neural_deprojection.models.Simple_complete_model.model_utils import SimpleCompleteModel
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from mayavi import mlab
from scipy import interpolate
from neural_deprojection.graph_net_utils import histogramdd
import matplotlib.pyplot as plt


if os.getcwd().split('/')[2] == 's2675544':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    autoencoder_checkpoint_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/autoencoder_2d_checkpointing'
    scm_checkpoint_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/simple_complete_checkpointing'
    autoencoder_saved_model_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/autoencoder_2d_saved_model'
    print('Running on ALICE')
else:
    tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
    autoencoder_checkpoint_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/autoencoder_2d_checkpointing'
    scm_checkpoint_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/simple_complete_checkpointing'
    autoencoder_saved_model_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/autoencoder_2d_saved_model'
    print('Running at home')

tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

def build_dataset(tfrecords, batch_size):
    """
    Build data set from a directory of tfrecords. With graph batching

    Args:
        data_dir: str, path to *.tfrecords

    Returns: Dataset obj.
    """
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
                                                    axis=-1))).shuffle(buffer_size=50).batch(batch_size=batch_size)
    return dataset

tfrecords = glob.glob(os.path.join(tfrec_dir, 'train', '*.tfrecords'))
dataset = build_dataset(tfrecords, batch_size=1)
(graph, img) = next(iter(dataset))
print(img)

discrete_image_vae = DiscreteImageVAE(embedding_dim=64,  # 64
                                      num_embedding=1024,  # 1024
                                      hidden_size=64,  # 64
                                      num_token_samples=1,  # 4
                                      num_channels=2)

dummy = discrete_image_vae.encoder(tf.zeros((1, 256, 256, 2)))
# print(discrete_image_vae.encoder.trainable_variables)

encoder_cp = tf.train.Checkpoint(encoder=discrete_image_vae.encoder,
                                 decoder=discrete_image_vae.decoder)
model_cp = tf.train.Checkpoint(_model=encoder_cp)
checkpoint = tf.train.Checkpoint(module=model_cp)
status = tf.train.latest_checkpoint(autoencoder_checkpoint_dir)
checkpoint.restore(status).expect_partial()

print(discrete_image_vae.encoder.trainable_variables)

# imported = tf.saved_model.load(autoencoder_saved_model_dir)
#
# decoded_img = imported.sample_decoder(imported.sample_encoder(img), 10, 1)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.imshow(decoded_img[0, :, :, 0].numpy())
# plt.show()
#
# exit()

simple_complete_model = SimpleCompleteModel(num_properties=7,
                                            num_components=4,
                                            component_size=128,
                                            num_embedding_3d=512,
                                            edge_size=8,
                                            global_size=16,
                                            discrete_image_vae=discrete_image_vae,
                                            num_token_samples=1,
                                            batch=1,
                                            beta=6.6,
                                            name=None)
encoder_cp = tf.train.Checkpoint(decoder=simple_complete_model.decoder,
                                 field_reconstruction=simple_complete_model.field_reconstruction)
model_cp = tf.train.Checkpoint(_model=encoder_cp)
checkpoint = tf.train.Checkpoint(module=model_cp)
status = tf.train.latest_checkpoint(scm_checkpoint_dir)
checkpoint.restore(status).expect_partial()



grid_resolution = 40
x = tf.linspace(-1.7, 1.7, grid_resolution)[..., None]
x, y, z = tf.meshgrid(x, x, x, indexing='ij')  # 3 x [grid_resolution, grid_resolution, grid_resolution]

original = False

if original:
    print('Plotting original properties')
    original_data = graph.nodes[0, :, 6]  # [num_positions]
    original_pos = (graph.nodes[0, :, 0].numpy(), graph.nodes[0, :, 1].numpy(), graph.nodes[0, :, 2].numpy())  # [3, num_positions]
    interp_pos = (x, y, z)  # [3, grid_resolution, grid_resolution, grid_resolution]
    data = interpolate.griddata(original_pos,
                                original_data.numpy(),
                                xi=interp_pos,
                                fill_value=0.0)  # [grid_resolution, grid_resolution, grid_resolution]
    data_flat = original_data
    # pos = tf.reverse(graph.nodes[0, :, :2], [1])
    pos = graph.nodes[0, :, :2]
else:
    print('Plotting reconstructed properties')
    x = tf.reshape(x, shape=[-1])
    y = tf.reshape(y, shape=[-1])
    z = tf.reshape(z, shape=[-1])
    positions = tf.cast(tf.concat([x[..., None], y[..., None], z[..., None]], axis=1), dtype=tf.float32) # [8000, 3]
    # random_positions = tf.random.truncated_normal(shape=(grid_resolution**3, 3), stddev=0.85)
    # positions = random_positions
    # print(positions)
    properties = simple_complete_model._im_to_components(img, positions, 10)  # [num_components, num_positions, num_properties]
    # properties_one_component = tf.reduce_sum(properties, axis=0)  # [num_positions, num_properties]
    properties_one_component = properties[0]  # [num_positions, num_properties]
    data_flat = properties_one_component[:, 3]  # [num_positions] (3=density)
    data = tf.reshape(data_flat, [grid_resolution, grid_resolution, grid_resolution]).numpy()  # [grid_resolution, grid_resolution, grid_resolution]
    pos = positions[:, :2]

image_after, _ = histogramdd(tf.reverse(pos, [1]), bins=64, weights=data_flat)
# image_after, _ = histogramdd(pos, bins=50, weights=tf.random.truncated_normal((10000, )))
image_after -= tf.reduce_min(image_after)
image_after /= tf.reduce_max(image_after)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image_after.numpy())
ax[1].imshow(img[0, :, :, 1])
plt.show()

mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 800))
mlab.clf()

source = mlab.pipeline.scalar_field(data)
min = data.min()
max = data.max()
#0.65
#0.9
# vol = mlab.pipeline.volume(source, vmin=min + 0.25 * (max - min),
#                                    vmax=min + 0.8 * (max - min))

iso = mlab.pipeline.iso_surface(source, contours=10, opacity=0.2)

mlab.view(180, 180, 100 * grid_resolution / 40, [0.5 * grid_resolution, 0.5 * grid_resolution, 0.5 * grid_resolution])

mlab.show()
