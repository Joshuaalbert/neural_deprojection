import sys
sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')

from functools import partial
from graph_nets.graphs import GraphsTuple
from neural_deprojection.models.identify_medium_GCD.model_utils import decode_examples
from neural_deprojection.models.TwoD_to_2d_dVAE_GCD.graph_networks import DiscreteImageVAE
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, build_log_dir, \
    build_checkpoint_dir, batch_dataset_set_graph_tuples, get_distribution_strategy
import glob, os
import tensorflow as tf
import json
import sonnet as snt
from tensorflow_addons.image import gaussian_filter2d

MODEL_MAP = {'disc_img_vae': DiscreteImageVAE}

@tf.function
def double_downsample(image):
    filter = tf.ones((2, 2, 1, 1)) * 0.25
    img = tf.nn.conv2d(image[None, ...],
                        filters=filter, strides=2,
                        padding='SAME')
    return tf.nn.conv2d(img,
                        filters=filter, strides=2,
                        padding='SAME')[0, ...]

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
            img = batch
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
                          tf.concat([double_downsample(img), gaussian_filter2d(double_downsample(img), filter_shape=[6, 6])], axis=-1))

    dataset = dataset.shuffle(buffer_size=50).batch(batch_size=batch_size)

    # dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=batch_size)

    return dataset


def train_disc_img_vae(data_dir, config, kwargs):
    # strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords, batch_size=4)
    test_dataset = build_dataset(test_tfrecords, batch_size=4)

    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    # log_dir = build_log_dir('test_log_dir', config)
    # checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)
    log_dir = 'test_log_dir'
    checkpoint_dir = 'test_checkpointing'
    save_dir = 'saved_model'

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=10,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          save_model_dir=save_dir,
                          debug=True)


def main(data_dir):
    config = dict(model_type='disc_img_vae',
                  model_parameters=dict(embedding_dim=16,  # 64
                                        num_embedding=1024,  # 1024
                                        hidden_size=16,  # 64
                                        num_token_samples=4,  # 32
                                        num_channels=2),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict()
    train_disc_img_vae(data_dir, config, kwargs)


if __name__ == '__main__':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    # tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    main(tfrec_dir)
