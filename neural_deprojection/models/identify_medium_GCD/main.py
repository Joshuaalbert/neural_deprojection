import sys
sys.path.insert(1, '/home/s2675544/git/neural_deprojection/')

import glob, os
import tensorflow as tf
import sonnet as snt

from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_log_dir, \
    build_checkpoint_dir, batch_dataset_set_graph_tuples
from neural_deprojection.models.identify_medium_GCD.generate_data import decode_examples
from functools import partial
from tensorflow_addons.image import gaussian_filter2d
from graph_nets.graphs import GraphsTuple
from neural_deprojection.models.identify_medium_GCD.model_utils import Model, AutoEncoder


MODEL_MAP = dict(model1=Model)


def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None) -> TrainOneEpoch:
    model_cls = MODEL_MAP[model_type]

    model = model_cls(**model_parameters)

    def build_opt(**kwargs):
        opt_type = kwargs.get('opt_type')
        if opt_type == 'adam':
            learning_rate = kwargs.get('learning_rate', 1e-4)
            opt = snt.optimizers.Adam(learning_rate)
        else:
            raise ValueError('Opt {} invalid'.format(opt_type))
        return opt


    def build_loss(**loss_parameters):
        def loss(model_outputs, batch):
            (graph, img, c) = batch
            # loss =  mean(-sum_k^2 true[k] * log(pred[k]/true[k]))
            return tf.reduce_mean(tf.losses.binary_crossentropy(c[None, None], model_outputs, from_logits=True))# tf.math.sqrt(tf.reduce_mean(tf.math.square(rank - tf.nn.sigmoid(model_outputs[:, 0]))))
        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training


def build_dataset(tfrecords):
    # Extract the dataset (graph tuple, image, example_idx) from the tfrecords files
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(10,),
                                                             edge_shape=(2,),
                                                             image_shape=(1000, 1000, 1)))  # (graph, image, idx)
    # Take the graphs and their corresponding index and shuffle the order of these pairs
    # Do the same for the images
    _graphs = dataset.map(lambda graph_data_dict, img, cluster_idx, projection_idx, vprime:
                          (graph_data_dict, 26 * cluster_idx + projection_idx)).shuffle(buffer_size=260)  # .replace(globals=tf.zeros((1, 1)))
    _images = dataset.map(lambda graph_data_dict, img, cluster_idx, projection_idx, vprime:
                          (img, 26 * cluster_idx + projection_idx)).shuffle(buffer_size=260)

    # Zip the shuffled datasets back together so typically the index of the graph and image don't match.
    shuffled_dataset = tf.data.Dataset.zip((_graphs, _images))  # ((graph, idx1), (img, idx2))

    # Reshape the dataset to the graph, the image and a yes or no whether the indices are the same
    # So ((graph, idx1), (img, idx2)) --> (graph, img, True/False)
    shuffled_dataset = shuffled_dataset.map(lambda ds1, ds2: (ds1[0], ds2[0], ds1[1] == ds2[1]))  # (graph, img, yes/no)

    # Take the subset of the data where the graph and image don't correspond (which is most of the dataset, since it's shuffled)
    shuffled_dataset = shuffled_dataset.filter(lambda graph_data_dict, img, c: ~c)

    # Transform the True/False class into 1/0 integer
    shuffled_dataset = shuffled_dataset.map(lambda graph_data_dict, img, c:
                                            (GraphsTuple(**graph_data_dict), img, tf.cast(c, tf.int32)))

    # Use the original dataset where all indices correspond and give them class True and turn that into an integer
    # So every instance gets class 1
    nonshuffeled_dataset = dataset.map(lambda graph_data_dict, img, cluster_idx, projection_idx, vprime:
                                       (GraphsTuple(**graph_data_dict), img, tf.constant(1, dtype=tf.int32)))  # (graph, img, yes)

    # For the training data, take a sample either from the correct or incorrect combinations of graphs and images
    nn_dataset = tf.data.experimental.sample_from_datasets([shuffled_dataset, nonshuffeled_dataset])
    return nn_dataset


def train_identify_medium(data_dir, config):
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    train_dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=train_dataset, batch_size=32)
    test_dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=test_dataset, batch_size=32)

    train_one_epoch = build_training(**config)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=20,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)

def train_autoencoder(data_dir):
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    train_dataset = train_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)
    test_dataset = test_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)

    model = AutoEncoder(kernel_size=4)

    learning_rate = 1e-5
    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        (img,) = batch
        decoded_img = model_outputs
        return tf.reduce_mean((gaussian_filter2d(img, filter_shape=[6, 6]) - decoded_img[:, 12:-12, 12:-12, :]) ** 2)

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = 'autoencoder_log_dir'
    checkpoint_dir = 'autoencoder_checkpointing'

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=50,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)

def main(data_dir, config):
    # train_autoencoder(data_dir)
    train_identify_medium(data_dir, config)


if __name__ == '__main__':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    learning_rate = 1e-5
    kernel_size = 4
    # mlp_layers = 2
    image_feature_size = 64  # 64
    # conv_layers = 6
    # mlp_layer_nodes = 32
    mlp_size = 16  # 16
    cluster_encoded_size = 10  # 10
    image_encoded_size = 64  # 64
    core_steps = 28
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
    main(tfrec_dir, config)
