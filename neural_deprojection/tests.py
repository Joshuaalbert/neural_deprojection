import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple

from neural_deprojection.data.geometric_graph import find_screen_length, generate_example
from neural_deprojection.graph_net_utils import AbstractModule, TrainOneEpoch, vanilla_training_loop, histogramdd, \
    efficient_nn_index, graph_batch_reshape, graph_unbatch_reshape, map_coordinates


class TestClass(object):
    def __init__(self, a):
        self.a = a

    def __add__(self, b):
        """
        This overrides addition on this class.

        Args:
            b: some number

        Returns: self.a + b

        """
        return self.a + b

def test_test_class():
    tc = TestClass(1.)
    assert tc + 5. == 6.


def test_vanillia_training_loop():
    import sonnet as snt

    class Model(AbstractModule):
        def __init__(self, name=None):
            super(Model, self).__init__(name=name)
            self.net = snt.nets.MLP([10, 1] , activate_final=False)
        def _build(self, batch):
            (inputs, _) = batch
            return self.net(inputs)

    def loss(model_output, batch):
        (_, target) = batch
        return tf.reduce_mean((target - model_output)**2)


    dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal((100,5)), tf.random.normal((100,1)))).batch(10)

    training = TrainOneEpoch(Model(), loss, snt.optimizers.Adam(1e-4))
    vanilla_training_loop(dataset, training, 100, debug = False)


def test_find_screen_length():
    n_nodes = 100
    k_mean = 10.
    dim = 2
    R = np.sqrt(k_mean / (np.pi * n_nodes))
    nodes = np.random.uniform(size=(n_nodes, dim))
    # n_nodes, n_nodes
    dist = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    assert np.abs(R -find_screen_length(dist, k_mean))< 0.05


def test_generate_example():
    positions = np.random.uniform(0., 1., size=(50, 3))
    properties = np.random.uniform(0., 1., size=(50, 16, 16, 16, 5))  # 16^3 images of 5 properties.
    graph = generate_example(positions, properties, k_mean=3)
    assert graph.nodes.shape[1:] == (16,16,16,5)
    assert graph.edges.shape[1:] == (2,)


def test_histogramdd():
    sample = np.random.normal(size=(1000,2))
    hist, edges = histogramdd(sample, bins=30, weights=None, density=None)
    # plt.imshow(hist)
    # plt.show()
    hist_np, _, _ = np.histogram2d(sample[:,0], sample[:,1], bins=30)
    assert np.all(hist.numpy() == hist_np)


def test_efficient_nn_index():
    query_positions = tf.range(10)[:,None]
    positions = tf.range(20)[::-1,None]

    assert (efficient_nn_index(query_positions, positions).numpy() == tf.range(10,20)[::-1].numpy()).all()


def test_batch_reshape():
    data = dict(
        nodes=tf.reshape(tf.range(3*2),(3,2)),
        edges=tf.reshape(tf.range(40*5),(40,5)),
        senders=tf.random.uniform((40,),minval=0, maxval=3, dtype=tf.int32),
        receivers=tf.random.uniform((40,),minval=0, maxval=3, dtype=tf.int32),
        n_node=tf.constant([3]),
        n_edge=tf.constant([40]),
        globals=None)
    graph = GraphsTuple(**data)
    graphs = utils_tf.concat([graph]*4,axis=0)
    batched_graphs = graph_batch_reshape(graphs)
    assert tf.reduce_all(batched_graphs.nodes[0]==batched_graphs.nodes[1]).numpy()
    assert tf.reduce_all(batched_graphs.edges[0]==batched_graphs.edges[1]).numpy()
    assert tf.reduce_all(batched_graphs.senders[0]+graphs.n_node[0]==batched_graphs.senders[1]).numpy()
    assert tf.reduce_all(batched_graphs.receivers[0]+graphs.n_node[0]==batched_graphs.receivers[1]).numpy()

    # print(batched_graphs)
    unbatched_graphs = graph_unbatch_reshape(batched_graphs)
    for (t1, t2) in zip(graphs, unbatched_graphs):
        if t1 is not None:
            assert tf.reduce_all(t1==t2).numpy()


def test_map_coordinates():
    # print(tf.reshape(tf.cast(tf.range(4*5), tf.float32), (4,5)))
    # print(map_coordinates(tf.reshape(tf.cast(tf.range(4*5), tf.float32), (4,5)), ((0., 1., 2.), (0., 1., 3.)), 1).numpy())
    assert map_coordinates(tf.reshape(tf.cast(tf.range(4*5), tf.float32), (4,5)), (1., 1.), 1).numpy() == 6.0
    assert map_coordinates(tf.reshape(tf.cast(tf.range(4*5), tf.float32), (4,5)), ((0., 1., 2.), (0., 1., 3.)), 1).numpy().tolist() == [0., 6., 13.]
    assert tf.function(lambda coords: map_coordinates(tf.reshape(tf.cast(tf.range(4*5), tf.float32), (4,5)), coords, 1))((1., 1.)).numpy() == 6.
    assert tf.function(lambda coords: map_coordinates(tf.reshape(tf.cast(tf.range(4*5), tf.float32), (4,5)), coords, 1))(((0., 1., 2.), (0., 1., 3.))).numpy().tolist() == [0., 6., 13.]