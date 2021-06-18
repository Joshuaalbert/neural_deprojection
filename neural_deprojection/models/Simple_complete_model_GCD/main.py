import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.Simple_complete_model.model_utils import SimpleCompleteModel
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
from neural_deprojection.models.identify_medium_GCD.generate_data import decode_examples
import glob, os, json
from functools import partial
from tensorflow_addons.image import gaussian_filter2d

MODEL_MAP = dict(simple_complete_model=SimpleCompleteModel)


@tf.function
def double_downsample(image):
    filter = tf.ones((2, 2, 1, 1)) * 0.25
    img = tf.nn.conv2d(image[None, ...],
                        filters=filter, strides=2,
                        padding='SAME')
    return tf.nn.conv2d(img,
                        filters=filter, strides=2,
                        padding='SAME')[0, ...]


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
            (graphs, imgs) = batch
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

    dataset = dataset.map(lambda graph_data_dict,
                                 img,
                                 cluster_idx,
                                 projection_idx,
                                 vprime: (GraphsTuple(**graph_data_dict),
                                          tf.concat([double_downsample(img),
                                                     gaussian_filter2d(double_downsample(img),
                                                                       filter_shape=[6, 6])],
                                                    axis=-1))).shuffle(buffer_size=50).batch(batch_size=batch_size)
    return dataset


def main(data_dir, config, kwargs):
    # strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords, batch_size=4)
    test_dataset = build_dataset(test_tfrecords, batch_size=4)

    # print(next(iter(train_dataset)))

    # with strategy.scope():
    train_one_epoch = build_training(**config, **kwargs)

    # train_one_epoch.model.set_beta(0.)

    log_dir = build_log_dir('simple_complete_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('simple_complete_checkpointing', config)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    save_model_dir = os.path.join('simple_complete_saved_model')

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
    if os.getcwd().split('/')[2] == 's2675544':
        tfrec_base_dir = '/home/s2675544/data/tf_records'
        checkpoint_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/autoencoder_2d_checkpointing'
        print('Running on ALICE')
    else:
        tfrec_base_dir = '/home/matthijs/Documents/Studie/Master_Astronomy/1st_Research_Project/Data/tf_records'
        checkpoint_dir = '/home/matthijs/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/autoencoder_2d_checkpointing'
        print('Running at home')

    # Load the autoencoder model from checkpoint
    discrete_image_vae = DiscreteImageVAE(embedding_dim=64,  # 64
                                          num_embedding=1024,  # 1024
                                          hidden_size=64,  # 64
                                          num_token_samples=4,  # 4
                                          num_channels=2)

    encoder_cp = tf.train.Checkpoint(encoder=discrete_image_vae.encoder)
    model_cp = tf.train.Checkpoint(_model=encoder_cp)
    checkpoint = tf.train.Checkpoint(module=model_cp)
    status = tf.train.latest_checkpoint(checkpoint_dir)

    checkpoint.restore(status).expect_partial()

    config = dict(model_type='simple_complete_model',
                  model_parameters=dict(num_properties=7,
                                        num_components=64,  # 64
                                        component_size=16,  # 16
                                        num_embedding_3d=64,  # 64
                                        edge_size=16,  # 16
                                        global_size=16),  # 16
                  optimizer_parameters=dict(learning_rate=4e-5, opt_type='adam'),
                  loss_parameters=dict())
    kwargs = dict(discrete_image_vae=discrete_image_vae,
                  num_token_samples=4,
                  batch=4,
                  name='simple_complete_model')

    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    main(tfrec_dir, config, kwargs)
