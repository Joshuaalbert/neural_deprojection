import sys
sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model.autoregressive_prior import AutoRegressivePrior
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_example_dataset, \
    grid_graphs, build_log_dir, build_checkpoint_dir, temperature_schedule, batch_graph_data_dict,\
    graph_unbatch_reshape, get_distribution_strategy
from neural_deprojection.models.identify_medium_SCD.generate_data import decode_examples
from neural_deprojection.models.Simple_complete_model.plot_evaluations import plot_voxel


from graph_nets.graphs import GraphsTuple
import os
import tensorflow as tf
import json
import glob
from functools import partial

import pylab as plt
import sonnet as snt
import numpy as np


MODEL_MAP = {'auto_regressive_prior': AutoRegressivePrior,
             'disc_image_vae': DiscreteImageVAE,
             'disc_voxel_vae': DiscreteVoxelsVAE}

def build_dataset(data_dir, batch_size):
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))

    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(11,),
                                                             image_shape=(256, 256, 1),
                                                             k=6))  # (graph, image, spsh, proj)

    dataset = dataset.map(lambda graph_data_dict, img, spsh, proj, e: (graph_data_dict, img))

    dataset = dataset.batch(batch_size)
    #batch fixing mechanism
    dataset = dataset.map(lambda data_dict, image: (batch_graph_data_dict(data_dict), image))
    dataset = dataset.map(lambda data_dict, image: (GraphsTuple(**data_dict,
                                                                edges=None, receivers=None, senders=None, globals=None), image))
    dataset = dataset.map(lambda batched_graphs, image: (graph_unbatch_reshape(batched_graphs), image))
    # dataset = dataset.cache()
    return dataset

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


def train_discrete_image_vae(data_dir, batch_size, config, kwargs, num_epochs=100):
    train_one_epoch = build_training(**config, **kwargs)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=batch_size)
    test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=batch_size)

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
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          variables=train_one_epoch.model.trainable_variables,
                          debug=False)

    return train_one_epoch.model, checkpoint_dir

def train_discrete_voxel_vae(data_dir, batch_size, config, kwargs, num_epochs=100):
    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=batch_size)
    test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=batch_size)

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
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          debug=False)
    return train_one_epoch.model, checkpoint_dir

def train_auto_regressive_prior(data_dir, batch_size, config, kwargs, num_epochs=100):
    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=batch_size)
    test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=batch_size)

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
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          variables=trainable_variables,
                          debug=False)
    return train_one_epoch.model, checkpoint_dir

def main():
    data_dir = '/home/s1825216/data/dataset'
    batch_size = 2

    print("Training the discrete image VAE")
    config = dict(model_type='disc_image_vae',
                  model_parameters=dict(embedding_dim=24,  # 64
                                        num_embedding=24,  # 1024
                                        hidden_size=32,
                                        num_channels=1,
                                        num_groups=5 # 256/16 -> 16 -> 256 nodes
                                        ),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())
    get_temp = temperature_schedule(config['model_parameters']['num_embedding'], 100)
    kwargs = dict(num_token_samples=4,
                  compute_temperature=get_temp,
                  beta=1.)
    # set num_epochs=0 to load but not train
    discrete_image_vae, discrete_image_vae_checkpoint = train_discrete_image_vae(data_dir, batch_size, config, kwargs, num_epochs=100)

    print("Training the discrete voxel VAE.")
    config = dict(model_type='disc_voxel_vae',
                  model_parameters=dict(voxels_per_dimension=8*6,#128
                                        embedding_dim=64,  # 64
                                        num_embedding=128,  # 1024
                                        hidden_size=8,
                                        num_channels=1,
                                        num_groups=4 #128 -> 8 -> 512 nodes  # reduction 2^(n-1)
                                        ),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())
    get_temp = temperature_schedule(config['model_parameters']['num_embedding'], 100)
    kwargs = dict(num_token_samples=4,
                  compute_temperature=get_temp,
                  beta=1.)
    # set num_epochs=0 to load but not train
    discrete_voxel_vae, discrete_voxel_vae_checkpoint = train_discrete_voxel_vae(data_dir, batch_size, config, kwargs, num_epochs=100)

    print("Training auto-regressive prior.")
    config = dict(model_type='auto_regressive_prior',
                  model_parameters=dict(num_heads=2,
                                        num_layers=2,
                                        embedding_dim=64
                                        ),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())

    kwargs = dict(discrete_image_vae=discrete_image_vae,
                  discrete_voxel_vae=discrete_voxel_vae,
                  num_token_samples=1)
    # set num_epochs=0 to load but not train
    autoregressive_prior, autoregressive_prior_checkpoint = train_auto_regressive_prior(data_dir, batch_size, config, kwargs, num_epochs=100)

    print("Evaluating auto-regressive prior")
    evaluate_auto_regressive_prior(data_dir, batch_size, autoregressive_prior, 'eval_dir')


def evaluate_auto_regressive_prior(data_dir, batch_size, autoregressive_prior: AutoRegressivePrior, output_dir):

    dataset = build_dataset(os.path.join(data_dir, 'eval'), batch_size=batch_size)

    tf_grid_graphs = tf.function(
        lambda graphs: grid_graphs(graphs, autoregressive_prior.discrete_voxel_vae.voxels_per_dimension))
    fig_dir = os.path.join(output_dir, 'evaluated_pairs')
    os.makedirs(fig_dir, exist_ok=True)
    pair_idx = 0
    for (graphs, images) in iter(dataset):
        actual_voxels = tf_grid_graphs(graphs)
        actual_voxels = actual_voxels.numpy()

        # batch, H,W,D,C
        mu_3d, b_3d = autoregressive_prior._deproject_images(images[0:1])
        mu_3d = mu_3d.numpy()
        b_3d = b_3d.numpy()

        for _actual_voxels, _mu_3d, _b_3d, _image in zip(actual_voxels, mu_3d, b_3d, images):
            np.savez(os.path.join(fig_dir, "evaluation_{:05}.npz".format(pair_idx)),
                     actual_voxels=_actual_voxels,
                     mu_3d=_mu_3d,
                     b_3d=_b_3d,
                     image=_image)
            # plot_voxel(_image, _mu_3d, _actual_voxels)

            pair_idx += 1


def load_checkpoints(discrete_image_vae_checkpoint, discrete_voxel_vae_checkpoint):
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
    return discrete_image_vae, discrete_voxel_vae


if __name__ == '__main__':
    main()
