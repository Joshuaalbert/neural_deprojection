import sys
sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')

from neural_deprojection.models.Simple_complete_model_with_voxel_data.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model_with_voxel_data.autoregressive_prior import AutoRegressivePrior
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_log_dir, \
    build_checkpoint_dir, temperature_schedule
import os
import tensorflow as tf
import json
import sonnet as snt
import random
import glob
from neural_deprojection.models.identify_medium_GCD.generate_data_with_voxel_data import decode_examples
from functools import partial
from tensorflow_addons.image import gaussian_filter2d

random.seed(1)

MODEL_MAP = {'auto_regressive_prior': AutoRegressivePrior,
             'disc_image_vae': DiscreteImageVAE,
             'disc_voxel_vae': DiscreteVoxelsVAE}


def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None,
                   **kwargs) -> TrainOneEpoch:
    model_cls = MODEL_MAP[model_type]
    model = model_cls(**model_parameters, **kwargs)

    def build_opt(**kwargs):
        opt_type = kwargs.get('opt_type')
        if opt_type == 'adam':
            learning_rate = kwargs.get('learning_rate')
            opt = snt.optimizers.Adam(learning_rate)
        else:
            raise ValueError('Opt {} invalid'.format(opt_type))
        return opt

    def build_loss(**loss_parameters):
        def loss(model_outputs, batch):
            return model_outputs['loss']

        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training

def build_dataset(tfrecords_dirs, batch_size, type='train'):
    """
    Build data set from a directory of tfrecords. With graph batching

    Args:
        data_dir: str, path to *.tfrecords

    Returns: Dataset obj.
    """

    tfrecords = []

    for tfrecords_dir in tfrecords_dirs:
        tfrecords += glob.glob(os.path.join(tfrecords_dir, type, '*.tfrecords'))

    random.shuffle(tfrecords)

    print(f'Number of {type} tfrecord files : {len(tfrecords)}')

    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             voxels_shape=(64, 64, 64, 7),
                                                             image_shape=(256, 256, 1)))  # (voxels, image, idx)

    dataset = dataset.map(lambda voxels,
                                 img,
                                 cluster_idx,
                                 projection_idx,
                                 vprime: (voxels[:, :, :, 3:5],
                                          gaussian_filter2d(img))).shuffle(buffer_size=52).batch(batch_size=batch_size)

    return dataset


def train_discrete_image_vae(data_dirs, config, kwargs, batch_size=1, num_epochs=100):
    print('\n')

    train_one_epoch = build_training(**config, **kwargs)
    train_dataset = build_dataset(data_dirs, batch_size=batch_size, type='train')
    test_dataset = build_dataset(data_dirs, batch_size=batch_size, type='test')

    print(f'Number of epochs: {num_epochs}')
    print('Training discrete image VAE\n')

    # drop the graph as the model expects only images
    train_dataset = train_dataset.map(lambda voxels, images: (images,))
    test_dataset = test_dataset.map(lambda voxels, images: (images,))

    # run on first input to set variable shapes
    for batch in iter(train_dataset):
        train_one_epoch.model(*batch)
        break

    log_dir = build_log_dir('log_dir', config)
    checkpoint_dir = build_checkpoint_dir('checkpointing', config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=num_epochs,
                          early_stop_patience=20,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          variables=train_one_epoch.model.trainable_variables,
                          debug=False)

    return train_one_epoch.model, checkpoint_dir


def train_discrete_voxel_vae(data_dirs, config, kwargs, batch_size=1, num_epochs=100):
    print('\n')

    train_one_epoch = build_training(**config, **kwargs)
    train_dataset = build_dataset(data_dirs, batch_size=batch_size, type='train')
    test_dataset = build_dataset(data_dirs, batch_size=batch_size, type='test')

    print(f'Number of epochs: {num_epochs}')
    print('Training discrete voxel VAE\n')

    # drop the image as the model expects only graphs
    train_dataset = train_dataset.map(lambda voxels, images: (voxels,))
    test_dataset = test_dataset.map(lambda voxels, images: (voxels,))

    # run on first input to set variable shapes
    for batch in iter(train_dataset):
        train_one_epoch.model(*batch)
        break

    log_dir = build_log_dir('log_dir', config)
    checkpoint_dir = build_checkpoint_dir('checkpointing', config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=num_epochs,
                          early_stop_patience=20,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          debug=False)
    return train_one_epoch.model, checkpoint_dir


def train_auto_regressive_prior(data_dirs, config, kwargs, batch_size=1, num_epochs=100):
    print('\n')

    train_one_epoch = build_training(**config, **kwargs)
    train_dataset = build_dataset(data_dirs, batch_size=batch_size, type='train')
    test_dataset = build_dataset(data_dirs, batch_size=batch_size, type='test')

    print(f'Number of epochs: {num_epochs}')
    print('Training autoregressive prior\n')

    # run on first input to set variable shapes
    for batch in iter(train_dataset):
        train_one_epoch.model(*batch)
        break

    log_dir = build_log_dir('log_dir', config)
    checkpoint_dir = build_checkpoint_dir('checkpointing', config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    exclude_variables = [variable.name for variable in kwargs['discrete_image_vae'].trainable_variables] \
                        + [variable.name for variable in kwargs['discrete_voxel_vae'].trainable_variables]
    trainable_variables = list(filter(lambda variable: (variable.name not in exclude_variables),
                                      train_one_epoch.model.trainable_variables))

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=num_epochs,
                          early_stop_patience=40,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          variables=trainable_variables,
                          debug=False)


def main():
    if os.getcwd().split('/')[2] == 's2675544':
        tfrec_base_dir = '/home/s2675544/data/tf_records'
        print('Running on ALICE')
    else:
        tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
        print('Running at home')

    tfrec_dirs = glob.glob(os.path.join(tfrec_base_dir, '*tf_records'))

    num_epochs_img = 30
    num_epochs_vox = 30
    num_epochs_auto = 30

    config = dict(model_type='disc_image_vae',
                  model_parameters=dict(embedding_dim=64,  # 64
                                        num_embedding=32,  # 1024
                                        hidden_size=32,
                                        num_groups=5,
                                        num_channels=1
                                        ),
                  optimizer_parameters=dict(learning_rate=4e-4, opt_type='adam'),
                  loss_parameters=dict())
    get_temp = temperature_schedule(config['model_parameters']['num_embedding'], num_epochs_img)
    kwargs = dict(num_token_samples=4,
                  compute_temperature=get_temp,
                  beta=1.)
    discrete_image_vae, discrete_image_vae_checkpoint = train_discrete_image_vae(tfrec_dirs,
                                                                                 config,
                                                                                 kwargs,
                                                                                 batch_size=4,
                                                                                 num_epochs=num_epochs_img)

    config = dict(model_type='disc_voxel_vae',
                  model_parameters=dict(voxels_per_dimension=64,
                                        embedding_dim=64,  # 64
                                        num_embedding=32,  # 1024
                                        hidden_size=32,
                                        num_groups=4,
                                        num_channels=2),
                  optimizer_parameters=dict(learning_rate=4e-4, opt_type='adam'),
                  loss_parameters=dict())
    get_temp = temperature_schedule(config['model_parameters']['num_embedding'], num_epochs_vox)
    kwargs = dict(num_token_samples=4,
                  compute_temperature=get_temp,
                  beta=1.)
    discrete_voxel_vae, discrete_voxel_vae_checkpoint = train_discrete_voxel_vae(tfrec_dirs,
                                                                                 config,
                                                                                 kwargs,
                                                                                 batch_size=4,
                                                                                 num_epochs=num_epochs_vox)

    config = dict(model_type='auto_regressive_prior',
                  model_parameters=dict(num_heads=4,
                                        num_layers=2
                                        ),
                  optimizer_parameters=dict(learning_rate=4e-4, opt_type='adam'),
                  loss_parameters=dict())

    kwargs = dict(discrete_image_vae=discrete_image_vae,
                  discrete_voxel_vae=discrete_voxel_vae,
                  embedding_dim=32,
                  num_token_samples=1)

    train_auto_regressive_prior(tfrec_dirs,
                                config,
                                kwargs,
                                batch_size=4,
                                num_epochs=num_epochs_auto)


if __name__ == '__main__':
    main()
