import sys
sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')

from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model.autoregressive_prior import AutoRegressivePrior
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_log_dir, \
    build_checkpoint_dir, temperature_schedule
import os
import tensorflow as tf
import json
import sonnet as snt
import random
import glob
from neural_deprojection.models.identify_medium_GCD.generate_data import decode_examples
from graph_nets.graphs import GraphsTuple
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
        gaussian_filter2d(img))).shuffle(buffer_size=52).batch(batch_size=batch_size)

    return dataset


def train_discrete_image_vae(data_dirs, config, kwargs, batch_size=1, num_epochs=100):

    train_one_epoch = build_training(**config, **kwargs)
    train_dataset = build_dataset(data_dirs, batch_size=batch_size, type='train')
    test_dataset = build_dataset(data_dirs, batch_size=batch_size, type='test')

    # drop the graph as the model expects only images
    train_dataset = train_dataset.map(lambda graphs, images: (images,))
    test_dataset = test_dataset.map(lambda graphs, images: (images,))

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

    train_one_epoch = build_training(**config, **kwargs)
    train_dataset = build_dataset(data_dirs, batch_size=batch_size, type='train')
    test_dataset = build_dataset(data_dirs, batch_size=batch_size, type='test')

    # drop the image as the model expects only graphs
    train_dataset = train_dataset.map(lambda graphs, images: (graphs,))
    test_dataset = test_dataset.map(lambda graphs, images: (graphs,))

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

    train_one_epoch = build_training(**config, **kwargs)
    train_dataset = build_dataset(data_dirs, batch_size=batch_size, type='train')
    test_dataset = build_dataset(data_dirs, batch_size=batch_size, type='test')

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

    data_dirs = ['AGN_TUNED_nu0_L400N1024_WMAP9_tf_records',
                 'snap_128_tf_records',
                 'snap_132_tf_records',
                 'snap_136_tf_records']

    tfrec_dirs = [os.path.join(tfrec_base_dir, data_dir) for data_dir in data_dirs]

    num_epochs_img = 20
    num_epochs_vox = 0
    num_epochs_auto = 0

    print("Training the discrete image VAE")
    config = dict(model_type='disc_image_vae',
                  model_parameters=dict(embedding_dim=32,  # 64
                                        num_embedding=32,  # 1024
                                        hidden_size=32,
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

    print("Training the discrete voxel VAE.")
    config = dict(model_type='disc_voxel_vae',
                  model_parameters=dict(voxels_per_dimension=8 * 8,
                                        embedding_dim=32,  # 64
                                        num_embedding=32,  # 1024
                                        hidden_size=32,
                                        num_groups=4,
                                        num_channels=2),
                  optimizer_parameters=dict(learning_rate=5e-4, opt_type='adam'),
                  loss_parameters=dict())
    get_temp = temperature_schedule(config['model_parameters']['num_embedding'], num_epochs_vox)
    kwargs = dict(num_token_samples=3,
                  compute_temperature=get_temp,
                  beta=1.)
    discrete_voxel_vae, discrete_voxel_vae_checkpoint = train_discrete_voxel_vae(tfrec_dirs,
                                                                                     config,
                                                                                     kwargs,
                                                                                     batch_size=3,
                                                                                     num_epochs=num_epochs_vox)

    print("Training auto-regressive prior.")
    config = dict(model_type='auto_regressive_prior',
                  model_parameters=dict(num_heads=4,
                                        num_layers=2
                                        ),
                  optimizer_parameters=dict(learning_rate=4e-4, opt_type='adam'),
                  loss_parameters=dict())

    get_temp = temperature_schedule(discrete_image_vae.num_embedding + discrete_voxel_vae.num_embedding, num_epochs_auto)
    kwargs = dict(discrete_image_vae=discrete_image_vae,
                  discrete_voxel_vae=discrete_voxel_vae,
                  num_token_samples=2,
                  compute_temperature=get_temp,
                  beta=1.)

    train_auto_regressive_prior(tfrec_dirs,
                                config,
                                kwargs,
                                batch_size=2,
                                num_epochs=num_epochs_auto)


if __name__ == '__main__':
    main()
