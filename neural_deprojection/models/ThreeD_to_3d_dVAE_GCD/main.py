import sys
sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')

from functools import partial
from graph_nets.graphs import GraphsTuple
from neural_deprojection.models.identify_medium_GCD.model_utils import decode_examples
from neural_deprojection.models.ThreeD_to_3d_dVAE_GCD.graph_networks import DiscreteGraphVAE
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_log_dir, \
    build_checkpoint_dir, batch_dataset_set_graph_tuples, get_distribution_strategy
import glob, os
import tensorflow as tf
import json
import sonnet as snt

MODEL_MAP = {'discvae': DiscreteGraphVAE}

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

def build_dataset(tfrecords, batch_size):
    """
    Build data set from a directory of tfrecords. With graph batching

    Args:
        data_dir: str, path to *.tfrecords

    Returns: Dataset obj.
    """
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(10,),
                                                             edge_shape=(2,),
                                                             image_shape=(1024, 1024, 1)))  # (graph, image, idx)

    dataset = dataset.map(lambda graph_data_dict, img, cluster_idx, projection_idx, vprime:
                          graph_data_dict).shuffle(buffer_size=50)
    dataset = dataset.map(lambda graph_data_dict: GraphsTuple(**graph_data_dict))

    # dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=batch_size)

    return dataset


def train_disc_graph_vae(data_dir, config, kwargs):
    # strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords, batch_size=1)
    test_dataset = build_dataset(test_tfrecords, batch_size=1)

    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    # log_dir = build_log_dir('test_log_dir', config)
    # checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)
    log_dir = 'test_log_dir'
    checkpoint_dir = 'test_checkpointing'

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
                          debug=True)


def main(data_dir):
    config = dict(model_type='discvae',
                  model_parameters=dict(embedding_dim=64,
                                        num_embedding=512,
                                        num_latent_tokens=16,
                                        num_gaussian_components=4),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict(
        num_token_samples=4,
        num_properties=7,
        encoder_kwargs=dict(inter_graph_connect_prob=0.01,
                            reducer=tf.math.unsorted_segment_mean,
                            starting_global_size=16,
                            node_size=8,
                            edge_size=8,
                            crossing_steps=12,
                            name='encoder'),
        decode_kwargs=dict(inter_graph_connect_prob=0.1,
                           reducer=tf.math.unsorted_segment_mean,
                           starting_global_size=16,
                           node_size=8,
                           edge_size=8,
                           crossing_steps=4,
                           name='decoder'), )
    train_disc_graph_vae(data_dir, config, kwargs)


if __name__ == '__main__':
    # tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    main(tfrec_dir)
