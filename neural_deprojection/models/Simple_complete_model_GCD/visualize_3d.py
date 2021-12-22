import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model_with_voxel_data.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model_with_voxel_data.autoregressive_prior import AutoRegressivePrior
from mayavi import mlab
from neural_deprojection.models.Simple_complete_model_GCD.complete_model.main_with_voxel_data import build_dataset


def decode_property(model,
                    voxels,
                    img,
                    property_index,
                    test_decoder=False):
    latent_logits_2d = model.discrete_image_vae.compute_logits(img)  # [batch, H, W, num_embeddings]
    # [num_samples, batch, H, W, num_embeddings], [num_samples, batch, H, W, embedding_size]
    _, _, latent_tokens_2d = model.discrete_image_vae.sample_latent(latent_logits_2d,
                                                                    0.1,
                                                                    model.discrete_image_vae.num_token_samples)
    mu_2d, _ = model.discrete_image_vae.compute_likelihood_parameters(latent_tokens_2d)

    if test_decoder:
        latent_logits_3d = model.discrete_voxel_vae.compute_logits(voxels)  # [batch, H, W, D, num_embeddings]

        # [num_samples, batch, H, W, D, num_embeddings], [num_samples, batch, H, W, D, embedding_size]
        _, _, latent_tokens_3d = model.discrete_voxel_vae.sample_latent(latent_logits_3d,
                                                                        0.1,
                                                                        model.discrete_voxel_vae.num_token_samples)
        mu_3d, _ = model.discrete_voxel_vae.compute_likelihood_parameters(latent_tokens_3d)
        mu_3d = mu_3d[0]
    else:
        mu_3d, _ = model.deproject_images(img)

    decoded_properties = mu_3d
    decoded_img = mu_2d

    decoded_property = decoded_properties[0, ..., property_index].numpy()  # [num_positions]
    return decoded_property, decoded_img


def main(model,
         voxels,
         image,
         prop_index: int,
         test_decoder=False):
    grid_res = model.discrete_voxel_vae.voxels_per_dimension

    decoded_voxels, decoded_img = decode_property(model,
                                                  voxels,
                                                  image,
                                                  prop_index,
                                                  test_decoder)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # image channel 1 is smoothed image
    image_before = ax[0].imshow(image[0, :, :, 0])
    fig.colorbar(image_before, ax=ax[0])

    # decoded_img [S, batch, H, W, channels]
    # image channel 1 is decoded from the smoothed image
    image_after = ax[1].imshow(decoded_img[0, 0, :, :, 0], vmin=np.min(image), vmax=np.max(image))
    fig.colorbar(image_after, ax=ax[1])

    plt.show()

    mlab.figure(1, bgcolor=(0, 0, 0), size=(800, 800))
    mlab.clf()

    print(decoded_voxels.shape)

    source_before = mlab.pipeline.scalar_field(voxels[..., prop_index].numpy()[0])
    source_after = mlab.pipeline.scalar_field(decoded_voxels)

    # v_min_before = np.min(data_before) + 0.25 * (np.max(data_before) - np.min(data_before))
    # v_max_before = np.min(data_before) + 0.75 * (np.max(data_before) - np.min(data_before))
    # mlab.pipeline.volume(source_before, vmin=v_min_before, vmax=v_max_before)
    #
    # v_min_after = np.min(data_after) + 0.25 * (np.max(data_after) - np.min(data_after))
    # v_max_after = np.min(data_after) + 0.75 * (np.max(data_after) - np.min(data_after))
    # mlab.pipeline.volume(source_after, vmin=v_min_after, vmax=v_max_after)

    mlab.pipeline.iso_surface(source_before, contours=24, opacity=0.3)
    mlab.pipeline.iso_surface(source_after, contours=24, opacity=0.3, vmin=np.min(voxels[..., prop_index]),
                              vmax=np.max(voxels[..., prop_index]))

    mlab.pipeline.scalar_cut_plane(source_before, line_width=2.0, plane_orientation='z_axes')
    mlab.pipeline.scalar_cut_plane(source_after, line_width=2.0, plane_orientation='z_axes')

    mlab.view(180, 180, 100 * grid_res / 40, [0.5 * grid_res,
                                              0.5 * grid_res,
                                              0.5 * grid_res])
    mlab.show()


if __name__ == '__main__':
    if os.getcwd().split('/')[2] == 's2675544':
        tfrec_base_dir = '/home/s2675544/data/tf_records'
        base_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/' \
                   'models/Simple_complete_model_GCD/complete_model'
        print('Running on ALICE')
    else:
        tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/data/tf_records'
        base_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/' \
                   'models/Simple_complete_model_GCD/complete_model'
        print('Running at home')

    scm_cp_dir = os.path.join(base_dir, 'checkpointing')
    scm_saved_dir = os.path.join(base_dir, 'saved_model')

    tfrec_dirs = [os.path.join(tfrec_base_dir, 'snap_132_tf_records')]
    dataset = build_dataset(tfrec_dirs, batch_size=1, type='train')
    counter = 0
    thing = iter(dataset)
    (vox, img) = next(thing)
    datapoint = 2
    # 2, 20, 15
    # 66
    # 543
    while counter < datapoint * 26:
        (vox, img) = next(thing)
        counter += 1

    discrete_image_vae_checkpoint = glob.glob(os.path.join(scm_cp_dir, '*disc_image_vae*'))[0]
    discrete_voxel_vae_checkpoint = glob.glob(os.path.join(scm_cp_dir, '*disc_voxel_vae*'))[0]
    auto_regressive_prior_checkpoint = glob.glob(os.path.join(scm_cp_dir, '*auto_regressive_prior*'))[0]

    print('image vae: ', os.path.basename(discrete_image_vae_checkpoint))
    print('voxel vae: ', os.path.basename(discrete_voxel_vae_checkpoint))
    print('autoregressive prior: ', os.path.basename(auto_regressive_prior_checkpoint))

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

    main(model=auto_regressive_prior,
         voxels=vox,
         image=img,
         prop_index=0,
         test_decoder=True)
