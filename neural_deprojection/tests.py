import numpy as np
import tensorflow as tf
from graph_nets.graphs import GraphsTuple

from neural_deprojection.geometric_graph import find_screen_length
from neural_deprojection.graph_net_utils import AbstractModule, TrainOneEpoch, vanilla_training_loop, save_graph_examples, \
    decode_graph_examples, save_graph_and_image_examples, decode_graph_and_image_examples


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


def test_graph_encode_decode():
    import numpy as np
    graph = GraphsTuple(nodes=np.random.normal(size=(100,2)),
                        edges=np.random.normal(size=(30,3)),
                        senders=np.random.randint(low=0, high=100,size=(30,)),
                        receivers=np.random.randint(low=0, high=100,size=(30,)),
                        globals=None,
                        n_node=100,
                        n_edge=30)

    tfrecords = save_graph_examples([graph])
    #loading the data into tf.data.Dataset object (same as TFDS makes)
    dataset = tf.data.TFRecordDataset(tfrecords).map(decode_graph_examples)
    loaded_graph = next(iter(dataset))
    for a,b in zip(graph, loaded_graph):
        if b is None:
            continue
        assert np.allclose(a,b.numpy())


def test_graph_image_encode_decode():
    import numpy as np
    graph = GraphsTuple(nodes=np.random.normal(size=(100,2)),
                        edges=np.random.normal(size=(30,3)),
                        senders=np.random.randint(low=0, high=100,size=(30,)),
                        receivers=np.random.randint(low=0, high=100,size=(30,)),
                        globals=None,
                        n_node=100,
                        n_edge=30)

    image = np.random.normal(size=(100,100, 3))

    tfrecords = save_graph_and_image_examples([graph], [image])
    dataset = tf.data.TFRecordDataset(tfrecords).map(decode_graph_and_image_examples)
    loaded_graph, loaded_image = next(iter(dataset))
    assert np.allclose(image, loaded_image.numpy())
    for a,b in zip(graph, loaded_graph):
        if b is None:
            continue
        assert np.allclose(a,b.numpy())


def test_find_screen_length():
    n_nodes = 100
    k_mean = 10.
    dim = 2
    R = np.sqrt(k_mean / (np.pi * n_nodes))
    nodes = np.random.uniform(size=(n_nodes, dim))
    # n_nodes, n_nodes
    dist = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    assert np.abs(R -find_screen_length(dist, k_mean))< 0.05