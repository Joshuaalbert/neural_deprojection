import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from functools import partial
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.identify_medium_SCD.generate_data import decode_examples, decode_examples_old

import glob, os, json

MODEL_MAP = dict(dis_im_vae=DiscreteImageVAE)


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


def build_dataset(data_dir, batch_size):
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))

    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(11,),
                                                             image_shape=(256, 256, 1),
                                                             k=6))  # (graph, image, spsh, proj)

    dataset = dataset.map(lambda graph_data_dict, img, spsh, proj, e: img).batch(batch_size=batch_size)

    return dataset


def main(data_dir, config, kwargs):
    # Make strategy at the start of your main before any other tf code is run.
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1,
    #                                      memory_limit=None)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=4)
    test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=4)

    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)
    train_one_epoch.model.set_temperature(10.)

    log_dir = build_log_dir('im_OH_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('im_OH_checkpointing', config)
    save_model_dir = os.path.join('OH_saved_models')

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

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
    # data_dir = '/home/s1825216/data/train_data/ClaudeData/'
    data_dir = '/home/s1825216/data/dataset/'

    config = dict(model_type='dis_im_vae',
                  model_parameters=dict(hidden_size=64,
                                        embedding_dim=64,
                                        num_embedding=1024,
                                        num_channels=1,
                                        name='discreteImageVAE'),
                  optimizer_parameters=dict(learning_rate=1e-4, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict(
        num_token_samples=4,
    )
    main(data_dir, config, kwargs)
