import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

import tensorflow as tf
import sonnet as snt
from functools import partial
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, \
    build_log_dir, build_checkpoint_dir, get_distribution_strategy
from neural_deprojection.models.Simple_complete_model.model_utils import SimpleCompleteModel, VoxelisedModel
from neural_deprojection.models.Simple_complete_model.autoencoder_2d_old import DiscreteImageVAE
from neural_deprojection.models.identify_medium_SCD.generate_data import decode_examples_old, decode_examples
import glob, os, json
from graph_nets.graphs import GraphsTuple


MODEL_MAP = dict(simple_complete_model=SimpleCompleteModel,
                 voxelised_model=VoxelisedModel)


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
            (graph, img) = batch
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
    dataset = _build_dataset(data_dir)

    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def _build_dataset(data_dir):
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))

    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(11,),
                                                             image_shape=(256, 256, 1),
                                                             k=6))  # (graph, image, spsh, proj)

    dataset = dataset.map(lambda graph_data_dict, img, spsh, proj, e: (GraphsTuple(**graph_data_dict), img))

    return dataset

# def batch_dataset(dataset, batch_size):
#     dataset = dataset.batch(batch_size=batch_size)
#     dataset = dataset.map(lambda graph_data_dict, img: {nodes: graph_data_dict['nodes'],
#                                                         edges: graph_data_dict['edges'],})



# def graph_correct(graph_data_dict):
#     nodes = graph_data_dict['nodes']
#     edges = graph_data_dict['edges']
#     globals = graph_data_dict['globals']
#     senders = graph_data_dict['senders']
#     receivers = graph_data_dict['receivers']
#     n_node = graph_data_dict['n_node']
#     n_edge = graph_data_dict['n_edge']
#
#     globals = globals[:, 0, :]
#     n_edge = n_edge[:, 0]
#     n_node = n_node[:, 0]
#
#     graph = GraphsTuple(nodes=nodes,
#                         edges=edges,
#                         globals=globals,
#                         senders=senders,
#                         receivers=receivers,
#                         n_node=n_node,
#                         n_edge=n_edge)
#
#     # unbatch
#     # offsets (autoregressive example)

def build_distributed_dataset(data_dir, global_batch_size, strategy):
    def data_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        dataset = _build_dataset(data_dir)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE())

        return dataset

    distributed_dataset = strategy.distribute_datasets_from_function(data_fn)

    return distributed_dataset


def main(data_dir, batch_size, config, kwargs):
    # Make strategy at the start of your main before any other tf code is run.
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1,
    #                                      memory_limit=None)
    strategy = None

    if strategy is not None:
        train_dataset = build_distributed_dataset(os.path.join(data_dir, 'train'), global_batch_size=batch_size, strategy=strategy)
        test_dataset = build_distributed_dataset(os.path.join(data_dir, 'test'), global_batch_size=batch_size, strategy=strategy)
    else:
        train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=batch_size)
        test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=batch_size)

    # for (graph, positions) in iter(test_dataset):
    #     print(graph)
    #     break

    if strategy is not None:
        with strategy.scope():
            train_one_epoch = build_training(**config, **kwargs, strategy=strategy)
    else:
        train_one_epoch = build_training(**config, **kwargs, strategy=strategy)

    train_one_epoch.model.set_temperature(10.)
    train_one_epoch.model.set_beta(6.6)

    log_dir = build_log_dir('single_voxelised_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('single_voxelised_checkpointing', config)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    save_model_dir = os.path.join('single_voxelised_saved_models')

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=100,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=save_model_dir,
                          debug=False)


if __name__ == '__main__':
    data_dir = '/home/s1825216/data/dataset/'
    saved_model_dir = 'second_im_16_saved_models'

    batch_size = 1

    # no attributes or trainable variables so not trainable?
    discrete_image_vae = tf.saved_model.load(saved_model_dir)

    config = dict(model_type='voxelised_model',
                  model_parameters=dict(num_properties=1,
                                        # num_components=8,
                                        voxel_per_dimension=4,
                                        decoder_3d_hidden_size=4,
                                        component_size=16,
                                        num_embedding_3d=1024,
                                        edge_size=4,
                                        global_size=16,
                                        num_heads=2,
                                        multi_head_output_size=16,
                                        name='simple_complete_model'),
                  optimizer_parameters=dict(learning_rate=1e-4, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict(num_token_samples=2,
                  n_node_per_graph=256,
                  batch=batch_size,
                  discrete_image_vae=discrete_image_vae)
    main(data_dir, batch_size, config, kwargs)
