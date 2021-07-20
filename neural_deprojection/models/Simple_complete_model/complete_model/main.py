from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model.autoregressive_prior import AutoRegressivePrior
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_example_dataset, \
    grid_graphs, build_log_dir, build_checkpoint_dir, temperature_schedule
import os
import tensorflow as tf
import json
import pylab as plt
import sonnet as snt

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


def train_discrete_image_vae(config, kwargs, num_epochs=100):
    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    dataset = build_example_dataset(100, batch_size=2, num_blobs=3, num_nodes=64 ** 3, image_dim=256)

    # show example of image
    for graphs, image in iter(dataset):
        assert image.numpy().shape == (2, 256, 256, 1)
        plt.imshow(image[0,...,0].numpy())
        plt.colorbar()
        plt.show()
        break

    # drop the graph as the model expects only images
    dataset = dataset.map(lambda graphs, images: (images,))

    # run on first input to set variable shapes
    for batch in iter(dataset):
        train_one_epoch.model(*batch)
        break

    log_dir = build_log_dir('log_dir', config)
    checkpoint_dir = build_checkpoint_dir('checkpointing', config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=dataset,
                          num_epochs=num_epochs,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          variables=train_one_epoch.model.trainable_variables,
                          debug=False)

    return train_one_epoch.model, checkpoint_dir


def train_discrete_voxel_vae(config, kwargs, num_epochs=100):
    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    dataset = build_example_dataset(1000, batch_size=2, num_blobs=3, num_nodes=64 ** 3, image_dim=256)

    # the model will call grid_graphs internally to learn the 3D autoencoder.
    # we show here what that produces from a batch of graphs.
    for graphs, image in iter(dataset):
        assert image.numpy().shape == (2, 256, 256, 1)
        plt.imshow(image[0,...,0].numpy())
        plt.colorbar()
        plt.show()
        voxels = grid_graphs(graphs, 64)
        assert voxels.numpy().shape == (2, 64, 64, 64, 1)
        plt.imshow(tf.reduce_mean(voxels[0,...,0], axis=-1))
        plt.colorbar()
        plt.show()
        break

    # drop the image as the model expects only graphs
    dataset = dataset.map(lambda graphs, images: (graphs,))

    # run on first input to set variable shapes
    for batch in iter(dataset):
        train_one_epoch.model(*batch)
        break

    log_dir = build_log_dir('log_dir', config)
    checkpoint_dir = build_checkpoint_dir('checkpointing', config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=dataset,
                          num_epochs=num_epochs,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          debug=False)
    return train_one_epoch.model, checkpoint_dir


def train_auto_regressive_prior(config, kwargs, num_epochs=100):
    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    dataset = build_example_dataset(100, batch_size=2, num_blobs=3, num_nodes=64 ** 3, image_dim=256)

    # the model will call grid_graphs internally to learn the 3D autoencoder.
    # we show here what that produces from a batch of graphs.
    for graphs, image in iter(dataset):
        assert image.numpy().shape == (2, 256, 256, 1)
        plt.imshow(image[0,...,0].numpy())
        plt.colorbar()
        plt.show()
        voxels = grid_graphs(graphs, 64)
        assert voxels.numpy().shape == (2, 64, 64, 64, 1)
        plt.imshow(tf.reduce_mean(voxels[0,...,0], axis=-1))
        plt.colorbar()
        plt.show()
        break

    # run on first input to set variable shapes
    for batch in iter(dataset):
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
                          training_dataset=dataset,
                          num_epochs=num_epochs,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          variables=trainable_variables,
                          debug=False)


def main():
    print("Training the discrete image VAE")
    config = dict(model_type='disc_image_vae',
                  model_parameters=dict(embedding_dim=64,  # 64
                                        num_embedding=16,  # 1024
                                        hidden_size=32,
                                        num_channels=1
                                        ),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())
    get_temp = temperature_schedule(config['model_parameters']['num_embedding'], 10)
    for epoch in range(0, 15):
        kwargs = dict(num_token_samples=4,
                      temperature=get_temp(epoch),
                      beta=1.)
        discrete_image_vae, discrete_image_vae_checkpoint = train_discrete_image_vae(config, kwargs, num_epochs=1)

    print("Training the discrete voxel VAE.")
    config = dict(model_type='disc_voxel_vae',
                  model_parameters=dict(voxels_per_dimension=4 * 8,
                                        embedding_dim=64,  # 64
                                        num_embedding=16,  # 1024
                                        hidden_size=4,
                                        num_channels=1),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())
    get_temp = temperature_schedule(config['model_parameters']['num_embedding'], 10)
    for epoch in range(0, 15):
        kwargs = dict(num_token_samples=4,
                      temperature=get_temp(epoch),
                      beta=1.)
        discrete_voxel_vae, discrete_voxel_vae_checkpoint = train_discrete_voxel_vae(config, kwargs, num_epochs=1)

    print("Training auto-regressive prior.")
    config = dict(model_type='auto_regressive_prior',
                  model_parameters=dict(num_heads=1,
                                        num_layers=1
                                        ),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())

    # can load checkpoints if desired.
    # discrete_image_vae, discrete_voxel_vae = load_checkpoints(discrete_image_vae_checkpoint, discrete_voxel_vae_checkpoint)

    get_temp = temperature_schedule(discrete_image_vae.num_embedding + discrete_voxel_vae.num_embedding, 10)
    for epoch in range(0, 15):
        kwargs = dict(discrete_image_vae=discrete_image_vae,
                      discrete_voxel_vae=discrete_voxel_vae,
                      num_token_samples=4,
                      temperature=get_temp(epoch),
                      beta=1.)

        train_auto_regressive_prior(config, kwargs, num_epochs=1)


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
