from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_example_dataset, grid_graphs, build_log_dir, build_checkpoint_dir
import os
import json
import pylab as plt
import sonnet as snt

MODEL_MAP = {'disc_image_vae': DiscreteImageVAE}

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

def train_discrete_image_vae(config, kwargs):
    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    dataset = build_example_dataset(10, batch_size=2, num_blobs=3, num_nodes=64**3, image_dim=256)

    # show example of image
    for graphs, image in iter(dataset):
        assert image.numpy().shape == (2, 256, 256, 1)
        plt.imshow(image[0].numpy())
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
                          num_epochs=1,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=checkpoint_dir,
                          variables=train_one_epoch.model.trainable_variables,
                          debug=False)


def main():

    config = dict(model_type='disc_image_vae',
                  model_parameters=dict(embedding_dim=64,  # 64
                                        num_embedding=128,  # 1024
                                        hidden_size=32,
                                        num_channels=1
                                        ),
                  optimizer_parameters=dict(learning_rate=1e-3, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict(num_token_samples=4,
                  temperature=1.,
                  beta=1.)
    train_discrete_image_vae(config, kwargs)


if __name__ == '__main__':
    main()
