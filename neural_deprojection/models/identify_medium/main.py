from neural_deprojection.models.identify_medium.generate_data import decode_examples
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir
import glob, os
import tensorflow as tf
from functools import partial
import pylab as plt

from graph_nets import blocks
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic
import json


class RelationNetwork(AbstractModule):
    """Implementation of a Relation Network.

    See https://arxiv.org/abs/1706.01427 for more details.

    The global and edges features of the input graph are not used, and are
    allowed to be `None` (the receivers and senders properties must be present).
    The output graph has updated, non-`None`, globals.
    """

    def __init__(self,
                 edge_model_fn,
                 global_model_fn,
                 reducer=tf.math.unsorted_segment_mean,
                 use_globals=False,
                 name="relation_network"):
        """Initializes the RelationNetwork module.

        Args:
          edge_model_fn: A callable that will be passed to EdgeBlock to perform
            per-edge computations. The callable must return a Sonnet module (or
            equivalent; see EdgeBlock for details).
          global_model_fn: A callable that will be passed to GlobalBlock to perform
            per-global computations. The callable must return a Sonnet module (or
            equivalent; see GlobalBlock for details).
          reducer: Reducer to be used by GlobalBlock to aggregate edges. Defaults to
            tf.math.unsorted_segment_sum.
          name: The module name.
        """
        super(RelationNetwork, self).__init__(name=name)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=edge_model_fn,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=use_globals)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=global_model_fn,
            use_edges=True,
            use_nodes=False,
            use_globals=use_globals,
            edges_reducer=reducer)

    def _build(self, graph:GraphsTuple) -> GraphsTuple:
        """Connects the RelationNetwork.

        Args:
          graph: A `graphs.GraphsTuple` containing `Tensor`s, except for the edges
            and global properties which may be `None`.

        Returns:
          A `graphs.GraphsTuple` with updated globals.

        Raises:
          ValueError: If any of `graph.nodes`, `graph.receivers` or `graph.senders`
            is `None`.
        """
        output_graph = self._global_block(self._edge_block(graph))
        return output_graph  # graph.replace(globals=output_graph.globals)


class Model(AbstractModule):
    def __init__(self, image_feature_size=16, kernel_size=3, name=None, **unused_kwargs):
        super(Model, self).__init__(name=name)
        self.encoder_graph = RelationNetwork(lambda: snt.nets.MLP([32,16], activate_final=True),
                                       lambda: snt.nets.MLP([32, 16], activate_final=True))
        self.encoder_image = RelationNetwork(lambda: snt.nets.MLP([32, 16], activate_final=True),
                                       lambda: snt.nets.MLP([32, 16], activate_final=True))
        self.image_cnn = snt.Sequential([snt.Conv2D(16,kernel_size, stride=2), tf.nn.relu,
                                         snt.Conv2D(image_feature_size, kernel_size, stride=2), tf.nn.relu])
        self.compare = snt.nets.MLP([32, 1])
        self.image_feature_size=image_feature_size

        self._step = None

    @property
    def step(self):
        if self._step is None:
            raise ValueError("Need to set step idx variable. model.step = epoch")
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _build(self, batch, *args, **kwargs):
        (graph, img, c) = batch
        del c
        encoded_graph = self.encoder_graph(graph)
        tf.summary.image(f'img_before_cnn', img[None,...], step=self.step)
        img = self.image_cnn(img[None,...])
        for channel in range(img.shape[-1]):
            tf.summary.image(f'img_after_cnn[{channel}]', img[...,channel:channel+1], step=self.step)
        #1, w,h,c -> w*h, c
        nodes = tf.reshape(img, (-1,self.image_feature_size))
        img_graph = GraphsTuple(nodes=nodes,
                            edges=None,
                            globals=None,
                            receivers=None,
                            senders=None,
                            n_node=tf.shape(nodes)[0:1],
                            n_edge=tf.constant([0]))
        connected_graph = fully_connect_graph_dynamic(img_graph)
        encoded_img = self.encoder_image(connected_graph)
        return self.compare(tf.concat([encoded_graph.globals, encoded_img.globals], axis=1))#[1]


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
            return tf.reduce_mean(tf.losses.binary_crossentropy(c[None,None],model_outputs, from_logits=True))
        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training

def build_dataset(data_dir):
    """
    Build data set from a directory of tfrecords.

    Args:
        data_dir: str, path to *.tfrecords

    Returns: Dataset obj.
    """
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(4,),
                                                             image_shape=(18, 18, 1)))  # (graph_data_dict, image, idx)
    _graphs = dataset.map(lambda graph_data_dict, img, idx: (graph_data_dict, idx)).shuffle(buffer_size=50)
    _images = dataset.map(lambda graph_data_dict, img, idx: (img, idx)).shuffle(buffer_size=50)
    shuffled_dataset = tf.data.Dataset.zip((_graphs, _images))  # ((graph_data_dict, idx1), (img, idx2))
    shuffled_dataset = shuffled_dataset.map(lambda ds1, ds2: (ds1[0], ds2[0], ds1[1] == ds2[1]))  # (graph, img, yes/no)
    shuffled_dataset = shuffled_dataset.filter(lambda graph, img, c: ~c)
    shuffled_dataset = shuffled_dataset.map(lambda graph_data_dict, img, c: (graph_data_dict, img, tf.cast(c, tf.int32)))
    nonshuffeled_dataset = dataset.map(
        lambda graph_data_dict, img, idx: (graph_data_dict, img, tf.constant(1, dtype=tf.int32)))  # (graph, img, yes)
    dataset = tf.data.experimental.sample_from_datasets([shuffled_dataset, nonshuffeled_dataset])
    dataset = dataset.map(lambda graph_data_dict, img, c: (GraphsTuple(globals=None, edges=None, **graph_data_dict), img, c))
    return dataset


def main(data_dir, config):
    # Make strategy at the start of your main before any other tf code is run.
    strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    train_dataset = build_dataset(os.path.join(data_dir,'train'))
    test_dataset = build_dataset(os.path.join(data_dir,'test'))

    for (graph, img, c) in iter(test_dataset):
        print(graph)
        break


    with strategy.scope():
        train_one_epoch = build_training(**config)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)
    with open(os.path.join(checkpoint_dir,'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=3,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


if __name__ == '__main__':
    import numpy as np
    for kernel_size in [3,4,5]:
        for learning_rate in 10**np.linspace(-5, -3,5):
            for image_feature_size in [4,8,16]:
                config = dict(model_type='model1',
                              model_parameters=dict(kernel_size=kernel_size, image_feature_size=image_feature_size),
                              optimizer_parameters=dict(learning_rate=learning_rate, opt_type='adam'),
                              loss_parameters=dict())
                main('data', config)