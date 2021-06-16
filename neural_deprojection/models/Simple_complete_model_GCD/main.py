import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir
from neural_deprojection.models.Simple_complete_model_SCD.graph_networks import DiscreteGraphVAE
from neural_deprojection.models.identify_medium.generate_data import decode_examples
import glob, os, json

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
    tfrecords = glob.glob(os.path.join(data_dir,'*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(
        lambda record_bytes: decode_examples(record_bytes, node_shape=[4]))
    dataset = dataset.map(
        lambda graph_data_dict, img, c: GraphsTuple(globals=tf.zeros([1]), edges=None, **graph_data_dict))
    dataset = dataset.map(
        lambda graph: graph._replace(nodes=tf.concat([graph.nodes[:, :3], tf.math.log(graph.nodes[:, 3:])], axis=1)))
    dataset = dataset.map(lambda graph: (graph, ))
    # dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=batch_size)
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

    save_model_dir = None#os.path.join('saved_models')

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
        num_properties=1,
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
    main('../identify_medium/data_remake', config, kwargs)
