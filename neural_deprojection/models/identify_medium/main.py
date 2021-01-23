from neural_deprojection.models.identify_medium.generate_data import generate_data, decode_examples
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule
import glob, os
import tensorflow as tf
import numpy as np
from functools import partial

from graph_nets.utils_tf import set_zero_global_features
from graph_nets import blocks
from graph_nets.modules import GraphNetwork
from graph_nets._base import WrappedModelFnModule
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_np import graphs_tuple_to_networkxs
from graph_nets.utils_tf import fully_connect_graph_dynamic
from networkx.drawing import draw
from networkx.linalg.spectrum import normalized_laplacian_spectrum
from networkx import Graph
import pylab as plt


class RelationNetwork(AbstractModule):
    """Implementation of a Relation Network.

    See https://arxiv.org/abs/1706.01427 for more details.

    The global and edges features of the input graph are not used, and are
    allowed to be `None` (the receivers and senders properties must be present).
    The output graph has updated, non-`None`, globals.
    """

    def \
            __init__(self,
                 edge_model_fn,
                 global_model_fn,
                 reducer=tf.math.unsorted_segment_sum,
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

    def _build(self, graph):
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
    def __init__(self, image_feature_size=16, name=None):
        super(Model, self).__init__(name=name)
        self.encoder_graph = RelationNetwork(lambda: snt.nets.MLP([32, 32, 16], activate_final=True),
                                       lambda: snt.nets.MLP([32, 32, 16], activate_final=True))
        self.encoder_image = RelationNetwork(lambda: snt.nets.MLP([32, 32, 16], activate_final=True),
                                       lambda: snt.nets.MLP([32, 32, 16], activate_final=True))
        self.image_cnn = snt.Sequential([snt.Conv2D(16,5), tf.nn.relu, snt.Conv2D(image_feature_size, 5), tf.nn.relu])
        self.compare = snt.nets.MLP([32, 1])
        self.image_feature_size=image_feature_size

    def _build(self, batch, *args, **kwargs):
        (graph, img, c) = batch
        del c
        encoded_graph = self.encoder_graph(graph)
        img = self.image_cnn(img[None,...])#1, w,h,c -> w*h, c
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


def main(data_dir):
    tfrecords = glob.glob(os.path.join(data_dir,'*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(5,),
                                                             edge_shape=(2,),
                                                             image_shape=(24,24,1)))# (graph, image, idx)
    _graphs = dataset.map(lambda graph, img, idx: (graph, idx)).shuffle(buffer_size=50)
    _images = dataset.map(lambda graph, img, idx: (img, idx)).shuffle(buffer_size=50)
    shuffled_dataset = tf.data.Dataset.zip((_graphs, _images)) #((graph, idx1), (img, idx2))
    shuffled_dataset = shuffled_dataset.map(lambda ds1, ds2: (ds1[0], ds2[0], ds1[1]==ds2[1])) #(graph, img, yes/no)
    shuffled_dataset = shuffled_dataset.filter(lambda graph, img, c: ~c)
    shuffled_dataset = shuffled_dataset.map(lambda graph, img, c: (graph, img, tf.cast(c, tf.int32)))
    nonshuffeled_dataset = dataset.map(lambda graph, img, idx: (graph, img, tf.constant(1, dtype=tf.int32)))#(graph, img, yes)
    train_dataset = tf.data.experimental.sample_from_datasets([shuffled_dataset,nonshuffeled_dataset])

    train_dataset = dataset.shard(2,0)
    test_dataset = dataset.shard(2,1)

    model = Model()

    def loss(model_outputs, batch):
        (graph, img, c) = batch
        return tf.reduce_mean(tf.losses.binary_crossentropy(c[None,None],model_outputs, from_logits=True))# tf.math.sqrt(tf.reduce_mean(tf.math.square(rank - tf.nn.sigmoid(model_outputs[:, 0]))))

    opt = snt.optimizers.Adam(0.001)
    training = TrainOneEpoch(model, loss, opt)

    vanilla_training_loop(train_dataset, training, 3, False)

    # for (target_graph, graph, rank) in iter(test_dataset):
    #     predict_rank = tf.sigmoid(model((target_graph, graph, rank)))
    #     tg = graphs_tuple_to_networkxs(target_graph)[0]
    #     g = graphs_tuple_to_networkxs(graph)[0]
    #     dist, e1,e2 = graph_distance(Graph(tg), Graph(g))
    #     print("True rank={}, predicted rank={}".format(rank, predict_rank))
    #     fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    #     draw(tg, pos={n: tg.nodes[n]['features'] for n in tg.nodes}, ax=axs[0])
    #     draw(g, pos={n: g.nodes[n]['features'] for n in g.nodes}, ax=axs[1])
    #     axs[0].set_title(
    #         "True rank={:.3f}, predicted rank={:.3f}".format(rank.numpy().item(0), predict_rank.numpy().item(0)))
    #     axs[2].plot(e1,c='black',label='Target spectrum')
    #     axs[2].plot(e2,c='red',label='spectrum')
    #     axs[2].set_title("Normalised Laplacian spectral distance: {:.3f}".format(dist))
    #     axs[2].legend()
    #     plt.show()


if __name__ == '__main__':
    main('test_train_data')