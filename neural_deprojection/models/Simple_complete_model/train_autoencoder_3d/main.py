from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_example_dataset, grid_graphs, graph_unbatch_reshape, build_log_dir, build_checkpoint_dir
import os
import tensorflow as tf
import json
import pylab as plt
import sonnet as snt

MODEL_MAP = {'disc_voxel_vae': DiscreteVoxelsVAE}

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

def train_discrete_voxel_vae(config, kwargs):
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
                          num_epochs=1,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          debug=False)


def main():

    config = dict(model_type='disc_voxel_vae',
                  model_parameters=dict(voxels_per_dimension=8*8,
                                        embedding_dim=64,  # 64
                                        num_embedding=128,  # 1024
                                        hidden_size=4,
                                        num_channels=1),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict(num_token_samples=4,
                  temperature=2.,
                  beta=1.)
    train_discrete_voxel_vae(config, kwargs)


if __name__ == '__main__':
    main()
