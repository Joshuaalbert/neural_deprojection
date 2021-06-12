"""
Input Graph:
    nodes: (positions, properties)
    senders/receivers: K-nearest neighbours
    edges: None
    globals: None

Latent Graph:
    nodes: Tokens
    senders/receivers: None
    edges: None
    globals: None

Output Graph:
    nodes: gaussian representation of 3d structure
    senders/receivers: None
    edges: None
    globals: None
"""

import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic
from graph_nets.blocks import NodeBlock, EdgeBlock, GlobalBlock, ReceivedEdgesToNodesAggregator
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir, \
    gaussian_loss_function, reconstruct_fields_from_gaussians
from neural_deprojection.models.ThreeD_to_3d_dVAE_SCD.graph_networks import DiscreteGraphVAE
from neural_deprojection.models.identify_medium_SCD.generate_data import decode_examples, decode_examples_old
from functools import partial
import glob, os, json
import tensorflow_probability as tfp

MODEL_MAP = dict(dis_graph_vae=DiscreteGraphVAE)


def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None,
                   **kwargs) -> TrainOneEpoch:
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
            (graph,) = batch
            # model_outputs = dict(loss=tf.reduce_mean(log_likelihood_samples - kl_term_samples),
            # var_exp=tf.reduce_mean(log_likelihood_samples),
            # kl_term=tf.reduce_mean(kl_term_samples),
            # mean_perplexity=mean_perplexity)
            return model_outputs['loss']

        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training


def build_dataset(data_dir, batch_size):
    """
    Build data set from a directory of tfrecords. With graph batching

    Args:
        data_dir: str, path to *.tfrecords

    Returns: Dataset obj.
    """
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples_old,
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
    dataset = dataset.map(lambda graph: (graph,))

    # dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=1)

    return dataset


def main(data_dir, config, kwargs):
    # Make strategy at the start of your main before any other tf code is run.
    strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=4)
    test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=4)

    # for (graph, positions) in iter(test_dataset):
    #     print(graph)
    #     break

    with strategy.scope():
        train_one_epoch = build_training(**config, **kwargs)
    train_one_epoch.model.set_beta(0.)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)


    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    save_model_dir = os.path.join('saved_models')

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=10,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=save_model_dir,
                          debug=False)


if __name__ == '__main__':
    config = dict(model_type='dis_graph_vae',
                  model_parameters=dict(embedding_dim=64,
                                        num_embedding=1024,
                                        num_gaussian_components=128,
                                        num_latent_tokens=64),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict(
        num_token_samples=1,
        num_properties=8,
        encoder_kwargs=dict(inter_graph_connect_prob=0.01,
                            reducer=tf.math.unsorted_segment_mean,
                            starting_global_size=4,
                            node_size=64,
                            edge_size=4,
                            crossing_steps=4, ),
        decode_kwargs=dict(inter_graph_connect_prob=0.01,
                           reducer=tf.math.unsorted_segment_mean,
                           starting_global_size=4,
                           node_size=64,
                           edge_size=4,
                           crossing_steps=4), )

    test_train_dir = '/home/s1825216/data/train_data/ClaudeData/'
    # test_train_dir = '/home/s1825216/data/train_data/auto_encoder/'

    main(test_train_dir, config, kwargs)


