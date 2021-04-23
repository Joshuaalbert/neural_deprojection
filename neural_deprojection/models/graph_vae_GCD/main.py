import sys
sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from functools import partial
from graph_nets.graphs import GraphsTuple
from neural_deprojection.models.identify_medium_GCD.model_utils import decode_examples
from neural_deprojection.models.graph_vae_GCD.graph_VAE_utils import Model
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_log_dir, \
    build_checkpoint_dir, batch_dataset_set_graph_tuples
import glob, os
import tensorflow as tf
import json
import sonnet as snt

MODEL_MAP = {'model1': Model}

def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None, **kwargs) -> TrainOneEpoch:
    model_cls = MODEL_MAP[model_type]
    model = model_cls(**model_parameters, **kwargs)

    def build_opt(**kwargs):
        opt_type = kwargs.get('opt_type')
        if opt_type == 'adam':
            learning_rate = kwargs.get('learning_rate', 1e-4)
            opt = snt.optimizers.Adam(learning_rate, beta1=1 - 1 / 100, beta2=1 - 1 / 500)
        else:
            raise ValueError('Opt {} invalid'.format(opt_type))
        return opt

    def build_loss(**loss_parameters):
        def loss(model_outputs, batch):
            graph = batch
            decoded_graph = model_outputs
            return tf.reduce_mean((graph.nodes[:, 3:] - decoded_graph.nodes) ** 2, dtype=graph.nodes.dtype)
        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training

def build_dataset(data_dir):
    """
    Build data set from a directory of tfrecords. With graph batching

    Args:
        data_dir: str, path to *.tfrecords

    Returns: Dataset obj.
    """
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(11,),
                                                             image_shape=(256, 256, 1)))  # (graph, image, spsh, proj)

    dataset = dataset.map(lambda graph_data_dict, img, spsh, proj: graph_data_dict).shuffle(buffer_size=50)

    # _graphs = dataset.map(lambda graph_data_dict, img, spsh, proj: (graph_data_dict, spsh, proj)).shuffle(buffer_size=50)
    # _images = dataset.map(lambda graph_data_dict, img, spsh, proj: (img, spsh, proj)).shuffle(buffer_size=50)
    # shuffled_dataset = tf.data.Dataset.zip((_graphs, _images))  # ((graph_data_dict, idx1), (img, idx2))
    # shuffled_dataset = shuffled_dataset.map(lambda ds1, ds2: (ds1[0], ds2[0], (ds1[1] == ds2[1]) and
    #                                                           (ds1[2] == ds2[2])))  # (graph, img, yes/no)
    # shuffled_dataset = shuffled_dataset.filter(lambda graph_data_dict, img, c: ~c)
    # shuffled_dataset = shuffled_dataset.map(lambda graph_data_dict, img, c: (graph_data_dict, img, tf.cast(c, tf.int32)))
    # nonshuffeled_dataset = dataset.map(
    #     lambda graph_data_dict, img, spsh, proj : (graph_data_dict, img, tf.constant(1, dtype=tf.int32)))  # (graph, img, yes)
    # dataset = tf.data.experimental.sample_from_datasets([shuffled_dataset, nonshuffeled_dataset])
    dataset = dataset.map(lambda graph_data_dict: GraphsTuple(**graph_data_dict))

    dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=1)

    return dataset

def train_ae_3d(data_dir, config):
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    train_one_epoch = build_training(**config)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=20,
                          early_stop_patience=10,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=True)


def main(data_dir):
    config = dict(model_type='model1',
                  model_parameters=dict(mlp_size=16,
                                        cluster_encoded_size=10,
                                        num_heads=1,
                                        core_steps=15),
                  optimizer_parameters=dict(learning_rate=1e-5,
                                            opt_type='adam'),
                  loss_parameters=dict())
    train_ae_3d(data_dir, config)


if __name__ == '__main__':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    main(tfrec_dir)
