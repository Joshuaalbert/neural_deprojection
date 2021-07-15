from neural_deprojection.models.Simple_complete_model.autoregressive_prior import AutoRegressivePrior
from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model.autoregressive_prior import AutoRegressivePrior
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_example_dataset, grid_graphs, build_log_dir, build_checkpoint_dir
import os
import tensorflow as tf
import json
import pylab as plt
import sonnet as snt

MODEL_MAP = {'auto_regressive_prior': AutoRegressivePrior}

def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None, **kwargs) -> TrainOneEpoch:
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

def train_auto_regressive_prior(config, kwargs):
    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    dataset = build_example_dataset(100, batch_size=2, num_blobs=3, num_nodes=64**3, image_dim=256)

    # the model will call grid_graphs internally to learn the 3D autoencoder.
    # we show here what that produces from a batch of graphs.
    for graphs, image in iter(dataset):
        assert image.numpy().shape == (2, 256, 256, 1)
        plt.imshow(image[0].numpy())
        plt.colorbar()
        plt.show()
        voxels = grid_graphs(graphs, 64)
        assert voxels.numpy().shape == (2, 64, 64, 64, 1)
        plt.imshow(tf.reduce_mean(voxels[0], axis=-2))
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

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=dataset,
                          num_epochs=100,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          debug=False)


def main(discrete_image_vae_checkpoint, discrete_voxel_vae_checkpoint):

    config = dict(model_type='auto_regressive_prior',
                  model_parameters=dict(num_heads=1,
                                        num_layers=1
                                        ),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())

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

    kwargs = dict(discrete_image_vae=discrete_image_vae,
                  discrete_voxel_vae=discrete_voxel_vae,
                  num_token_samples=1,
                  temperature=1.,
                  beta=1.)

    train_auto_regressive_prior(config, kwargs)


if __name__ == '__main__':
    main(discrete_image_vae_checkpoint='/home/albert/git/neural_deprojection/neural_deprojection/models/Simple_complete_model/train_autoencoder_2d/checkpointing/|disc_image_vae||embddngdm=64,hddnsz=32,nmchnnls=1,nmembddng=128||lrnngrt=1.0e-03,opttyp=adam|||',
         discrete_voxel_vae_checkpoint='/home/albert/git/neural_deprojection/neural_deprojection/models/Simple_complete_model/train_autoencoder_3d/checkpointing/|disc_voxel_vae||embddngdm=64,hddnsz=4,nmchnnls=1,nmembddng=128||lrnngrt=1.0e-03,opttyp=adam|||')
