import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')
from neural_deprojection.models.identify_medium_SCD.model_utils import build_dataset, build_training, AutoEncoder
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir, batch_dataset_set_graph_tuples
import glob, os
import tensorflow as tf
import json
import sonnet as snt


def train_identify_medium(data_dir, config):
    # Make strategy at the start of your main before any other tf code is run.
    strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=11000)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'))
    test_dataset = build_dataset(os.path.join(data_dir, 'test'))


    print('\nEXAMPLE FROM TEST DATASET:')
    for (graph, img, c) in iter(train_dataset):
        print(img)
        print('max: ', tf.math.reduce_max(img))
        break

    with strategy.scope():
        train_one_epoch = build_training(**config)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)

    if checkpoint_dir is not None:          # originally from vanilla_training_loop
        os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir,'config.json'), 'w') as f:        # checkpoint_dir not yet created
        json.dump(config, f)


    print('\nvanilla training loop...')
    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=10,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


def train_autoencoder(data_dir):
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=10000)

    # lists containing tfrecord files
    train_dataset = build_dataset(os.path.join(data_dir, 'train'))
    test_dataset = build_dataset(os.path.join(data_dir, 'test'))

    # print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    # print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    # print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')
    #
    # train_dataset = build_dataset(train_tfrecords)
    # test_dataset = build_dataset(test_tfrecords)

    train_dataset = train_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)
    test_dataset = test_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)

    # with strategy.scope():
    model = AutoEncoder()

    learning_rate = 1.0e-5

    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        (img,) = batch
        decoded_img = model_outputs
        # return tf.reduce_mean((gaussian_filter2d(img, filter_shape=[6, 6]) - decoded_img[:, :, :, :]) ** 2)
        return 100*tf.reduce_mean((img - decoded_img[:, :, :, :]) ** 2)

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = 'autoencoder_log_dir'
    checkpoint_dir = 'autoencoder_checkpointing'

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=1000,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


def main(data_dir):
    # train_autoencoder(data_dir)

    learning_rate = 1e-5
    kernel_size = 4
    # mlp_layers = 2
    image_feature_size = 32
    # conv_layers = 6
    # mlp_layer_nodes = 32
    mlp_size = 16
    cluster_encoded_size = 11
    image_encoded_size = 32
    core_steps = 20
    num_heads = 4

    config = dict(model_type='model1',
                  model_parameters=dict(mlp_size=mlp_size,
                                        cluster_encoded_size=cluster_encoded_size,
                                        image_encoded_size=image_encoded_size,
                                        kernel_size=kernel_size,
                                        image_feature_size=image_feature_size,
                                        core_steps=core_steps,
                                        num_heads=num_heads),
                  optimizer_parameters=dict(learning_rate=learning_rate, opt_type='adam'),
                  loss_parameters=dict())
    train_identify_medium(data_dir, config)


if __name__ == '__main__':
    test_train_dir = '/home/s1825216/data/train_data/ClaudeData/'
    # test_train_dir = '/home/s1825216/data/train_data/auto_encoder/'
    main(test_train_dir)
