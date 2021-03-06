import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from functools import partial

from graph_nets.graphs import GraphsTuple

from neural_deprojection.models.identify_medium_SCD.generate_data import decode_examples, decode_examples_old

from neural_deprojection.models.graph_VAE_SCD.graph_VAE_utils import Model, DiscreteGraphVAE, EncoderNetwork3D, \
    DecoderNetwork3D
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir, batch_dataset_set_graph_tuples
import glob, os
import tensorflow as tf
import json
import sonnet as snt
import numpy as np

MODEL_MAP = {'model1': Model,
             'discvae': DiscreteGraphVAE}


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
        if model_type == 'model2':
            def loss(model_outputs, batch):
                graph = batch
                decoded_graph, nn_index = model_outputs
                return tf.reduce_mean((tf.gather(graph.nodes[:, 3:], nn_index) - decoded_graph.nodes) ** 2 * tf.constant(
                    [0, 0, 0, 0, 1, 0, 0, 0], dtype=graph.nodes.dtype))
        else:
            def loss(model_outputs, batch):
                graph = batch
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


def build_dataset(data_dir):
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

    # dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=1)

    return dataset


def train_ae_3d(data_dir, config):
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=10000)

    # lists containing tfrecord files
    train_dataset = build_dataset(os.path.join(data_dir, 'train'))
    test_dataset = build_dataset(os.path.join(data_dir, 'test'))

    # train_dataset = train_dataset.map(lambda graph, img, c: (graph,))
    # test_dataset = test_dataset.map(lambda graph, img, c: (graph,))

    # for ds_item in iter(train_dataset):
    #     print(ds_item)
    #     break

    # for ds_item in iter(train_dataset):
    #     print(ds_item)
    #     br

    train_one_epoch = build_training(**config)

    log_dir = build_log_dir('new_test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('new_test_checkpointing', config)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    checkpoint = tf.train.Checkpoint(module=train_one_epoch)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3,
                                         checkpoint_name=train_one_epoch.model.__class__.__name__)
    if manager.latest_checkpoint is not None:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")

    # output_dir = './output_evaluations'
    # os.makedirs(output_dir, exist_ok=True)
    # names = ['vx', 'vy', 'vz', 'grav_potential', 'rho', 'temperature', 'cell_mass', 'cell_volume']
    # for i, test_graph in enumerate(iter(test_dataset)):
    #     positions = test_graph.nodes[:, :3].numpy()
    #     input_properties = test_graph.nodes[:, 3:].numpy()
    #
    #     decoded_graph = train_one_epoch.model(test_graph)
    #     decoded_properties = decoded_graph.nodes.numpy()
    #     save_dict = dict(positions=positions)
    #     for j in range(len(names)):
    #         save_dict[f"prop_{names[j]}_input"] = input_properties[:,j]
    #         save_dict[f"prop_{names[j]}_decoded"] = decoded_properties[:,j]
    #     np.savez(os.path.join(output_dir, 'test_evaluation_{:04d}.npz'.format(i)), **save_dict)
    #
    #     if i == 20:
    #         break

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=10,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


def train_disc_graph_vae(data_dir, config):
    # strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    # lists containing tfrecord files
    train_dataset = build_dataset(os.path.join(data_dir, 'train'))
    test_dataset = build_dataset(os.path.join(data_dir, 'test'))

    # train_dataset = train_dataset.map(lambda graph, img, c: (graph,))
    # test_dataset = test_dataset.map(lambda graph, img, c: (graph,))

    # for ds_item in iter(train_dataset):
    #     print(ds_item)
    #     break

    # for ds_item in iter(train_dataset):
    #     print(ds_item)
    #     br

    train_one_epoch = build_training(model_parameters=dict(encoder_fn=EncoderNetwork3D,
                                                           decode_fn=DecoderNetwork3D,
                                                           embedding_dim=4,  # 64
                                                           num_embedding=4,  # 64
                                                           num_gaussian_components=4,  # 128
                                                           num_token_samples=1,
                                                           num_properties=8,
                                                           temperature=50.,
                                                           beta=1.,
                                                           encoder_kwargs=dict(inter_graph_connect_prob=0.01,
                                                                               reducer=tf.math.unsorted_segment_mean,
                                                                               starting_global_size=4,
                                                                               node_size=4,  # 64
                                                                               edge_size=4,
                                                                               crossing_steps=1,
                                                                               name=None),
                                                           decode_kwargs=dict(inter_graph_connect_prob=0.01,
                                                                              reducer=tf.math.unsorted_segment_mean,
                                                                              starting_global_size=4,
                                                                              node_size=4,  # 64
                                                                              edge_size=4,
                                                                              crossing_steps=1,
                                                                              name=None),
                                                           name=None),
                                     **config)

    # log_dir = build_log_dir('test_log_dir', config)
    # checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)
    log_dir = 'dVAE_log_dir'
    checkpoint_dir = 'dVAE_checkpointing'
    model_dir = 'dVAE_model'

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=100,
                          early_stop_patience=10,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=model_dir,
                          debug=False)


def main(data_dir):
    # config = dict(model_type='model1',
    #               model_parameters=dict(activation='leaky_relu',
    #                   mlp_size=16,
    #                   cluster_encoded_size=11,
    #                   num_heads=1,
    #                   core_steps=15,
    #                   name='with_random_positions'),
    #               optimizer_parameters=dict(learning_rate=1e-5,
    #                                         opt_type='adam'),
    #               loss_parameters=dict())

    config = dict(model_type='discvae',
                  optimizer_parameters=dict(learning_rate=1e-5,
                                            opt_type='adam'),
                  loss_parameters=dict())

    train_disc_graph_vae(data_dir, config)


if __name__ == '__main__':
    test_train_dir = '/home/s1825216/data/train_data/ClaudeData/'
    # test_train_dir = '/home/s1825216/data/train_data/auto_encoder/'
    main(test_train_dir)
