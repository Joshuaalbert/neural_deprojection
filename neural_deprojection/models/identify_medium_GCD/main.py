import sys
sys.path.append('/home/s2675544/git/neural_deprojection')

from neural_deprojection.models.identify_medium_GCD.generate_data import decode_examples
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, get_distribution_strategy
import glob, os
import tensorflow as tf
from functools import partial
from graph_nets import blocks
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic
from typing import Callable, Iterable, Optional, Text
from sonnet.src import initializers
from sonnet.src import linear


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
                 reducer=tf.math.unsorted_segment_mean, #try with mean instead of sum
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
        edge_block = self._edge_block(graph)
        # print(f'This is the maximum value of the edge block :{np.max(np.array(edge_block.edges))}')
        # print(f'This is the minimum value of the edge block :{np.min(np.array(edge_block.edges))}')
        # print(f'Mean of all edges : {np.mean(np.array(edge_block.edges), axis=0)}')
        output_graph = self._global_block(edge_block)
        # print(f'This is the global block :{np.array(output_graph.globals)}')
        return output_graph  # graph.replace(globals=output_graph.globals)


class MLP_with_bn(snt.Module):
    """A multi-layer perceptron module."""

    def __init__(self,
                 output_sizes: Iterable[int],
                 w_init: Optional[initializers.Initializer] = None,
                 b_init: Optional[initializers.Initializer] = None,
                 with_bias: bool = True,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 dropout_rate=None,
                 activate_final: bool = False,
                 name: Optional[Text] = None,
                 with_bn=False):
        """Constructs an MLP.

        Args:
          output_sizes: Sequence of layer sizes.
          w_init: Initializer for Linear weights.
          b_init: Initializer for Linear bias. Must be `None` if `with_bias` is
            `False`.
          with_bias: Whether or not to apply a bias in each layer.
          activation: Activation function to apply between linear layers. Defaults
            to ReLU.
          dropout_rate: Dropout rate to apply, a rate of `None` (the default) or `0`
            means no dropout will be applied.
          activate_final: Whether or not to activate the final layer of the MLP.
          name: Optional name for this module.

        Raises:
          ValueError: If with_bias is False and b_init is not None.
        """
        if not with_bias and b_init is not None:
            raise ValueError("When with_bias=False b_init must not be set.")

        super(MLP_with_bn, self).__init__(name=name)
        self._with_bias = with_bias
        self._w_init = w_init
        self._b_init = b_init
        self._activation = activation
        self._activate_final = activate_final
        self._dropout_rate = dropout_rate
        self._layers = []
        self._with_bn = with_bn
        self._bn = []
        for index, output_size in enumerate(output_sizes):
            # Besides a layer for every output_size in output_sizes (which are e.g [32, 32, 16])
            # also make a batch_normalization object except for the last layer
            # Create scale determines the scale of the normalization, which is 1 by default
            # Create offset determines the center of the normalization, which is 0 by default
            if self._with_bn:
                if index < len(output_sizes) - 1:
                    self._bn.append(
                        snt.BatchNorm(
                            create_scale=False,
                            create_offset=False))
            self._layers.append(
                linear.Linear(
                    output_size=output_size,
                    w_init=w_init,
                    b_init=b_init,
                    with_bias=with_bias,
                    name="linear_%d" % index))
        # print(f'Number of batch normalization objects : {len(self._bn)}')
        # print(f'Number of layer objects : {len(self._layers)}')

    def __call__(self, inputs: tf.Tensor, is_training=True) -> tf.Tensor:
        """Connects the module to some inputs.

        Args:
          inputs: A Tensor of shape `[batch_size, input_size]`.
          is_training: A bool indicating if we are currently training. Defaults to
            `None`. Required if using dropout.

        Returns:
          output: The output of the model of size `[batch_size, output_size]`.
        """

        num_layers = len(self._layers)

        for i, (layer, bn) in enumerate(zip(self._layers, self._bn)):
            # print(f'These are the inputs: {inputs}')
            inputs = layer(inputs)
            print(f'Shape : {inputs.shape}')
            print(f'These are the inputs after the layer: {inputs}')


            # Activation for all but the last layer, unless specified otherwise.
            if i < (num_layers - 1) or self._activate_final:
                inputs = self._activation(inputs)
                print(f'These are the inputs after activation: {inputs}')

            if self._with_bn:
                # Apply batch normalization for all but the last layer.
                if i < (num_layers - 1):
                    inputs = bn(inputs, is_training=is_training)
                print(f'These are the inputs after batch normalization: {inputs}')

        return inputs


class Model(AbstractModule):
    """Model inherits from AbstractModule, which contains a __call__ function which executes a _build function
    that is to be specified in the child class. So for example:
    model = Model(), then model() returns the output of _build()

    AbstractModule inherits from snt.Module, which has useful functions that can return the (trainable) variables,
    so the Model class has this functionality as well
    An instance of the RelationNetwork class also inherits from AbstractModule,
    so it also executes its _build() function when called and it can return its (trainable) variables


    A RelationNetwork contains an edge block and a global block:

    The edge block generally uses the edge, receiver, sender and global attributes of the input graph
    to calculate the new edges.
    In our case we currently only use the receiver and sender attributes to calculate the edges.

    The global block generally uses the aggregated edge, aggregated node and the global attributes of the input graph
    to calculate the new globals.
    In our case we currently only use the aggregated edge attributes to calculate the new globals.


    As input the RelationNetwork needs two (neural network) functions:
    one to calculate the new edges from receiver and sender nodes
    and one to calculate the globals from the aggregated edges.

    The new edges will be a vector with size 16 (i.e. the output of the first function in the RelationNetwork)
    The new globals will also be a vector with size 16 (i.e. the output of the second function in the RelationNetwork)

    The image_cnn downscales the image (currently from 4880x4880 to 35x35) and encodes the image in 16 channels.
    So we (currently) go from (4880,4880,1) to (35,35,16)
    """

    def __init__(self, image_feature_size=16, name=None):
        super(Model, self).__init__(name=name)
        self.encoder_graph = RelationNetwork(lambda: MLP_with_bn([32, 32, 16], activate_final=True),
                                       lambda: MLP_with_bn([32, 32, 16], activate_final=True))
        self.encoder_image = RelationNetwork(lambda: MLP_with_bn([32, 32, 16], activate_final=True),
                                       lambda: MLP_with_bn([32, 32, 16], activate_final=True))
        self.image_cnn = snt.Sequential([snt.Conv2D(16, 4, stride=2, padding='valid'), tf.nn.relu,
                                         snt.Conv2D(16, 4, stride=2, padding='valid'), tf.nn.relu,
                                         snt.Conv2D(16, 4, stride=2, padding='valid'), tf.nn.relu,
                                         snt.Conv2D(16, 4, stride=2, padding='valid'), tf.nn.relu,
                                         snt.Conv2D(image_feature_size, 4, stride=2, padding='valid'), tf.nn.relu])
        self.compare = snt.nets.MLP([32, 1])
        self.image_feature_size = image_feature_size

    def _build(self, batch, *args, **kwargs):
        (graph, img, c) = batch
        # print(f'Node example : {graph.nodes[0]}')
        # print(f'Virtual particles in cluster : {graph.nodes.shape}')
        # print(f'Original image shape : {img.shape}')
        # print(f'Image : {tf.reduce_mean(img)}')
        # print(f'St dev : {(tf.reduce_mean(img**2) - tf.reduce_mean(img)**2)**(1/2)}')
        # print(f'Min : {tf.reduce_min(img)}')
        # print(f'Max : {tf.reduce_max(img)}')
        del c
        encoded_graph = self.encoder_graph(graph)
        img = self.image_cnn(img[None,...])#1, w,h,c -> w*h, c
        # print(f'Convolutional network output shape : {img.shape}')
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
        # print(f'Encoded particle graph nodes shape : {encoded_graph.nodes.shape}')
        # print(f'Encoded particle graph edges shape : {encoded_graph.edges.shape}')
        # print(f'Encoded image graph nodes shape : {encoded_img.nodes.shape}')
        # print(f'Encoded image graph edges shape : {encoded_img.edges.shape}')
        # print(f'Encoded particle graph globals : {encoded_graph.globals}')
        # print(f'Encoded image graph globals : {encoded_img.globals}')
        distance = self.compare(tf.concat([encoded_graph.globals, encoded_img.globals], axis=1)) + self.compare(tf.concat([encoded_img.globals, encoded_graph.globals], axis=1))
        return distance

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
            return tf.reduce_mean(tf.losses.binary_crossentropy(c[None,None],model_outputs, from_logits=True))# tf.math.sqrt(tf.reduce_mean(tf.math.square(rank - tf.nn.sigmoid(model_outputs[:, 0]))))
        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training

def tfrecords_to_dataset(tfrecords):
    # Extract the dataset (graph tuple, image, example_idx) from the tfrecords files
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(10,),
                                                             edge_shape=(2,),
                                                             image_shape=(1000, 1000, 1)))  # (graph, image, idx)
    # Take the graphs and their corresponding index and shuffle the order of these pairs
    # Do the same for the images
    _graphs = dataset.map(
        lambda graph, img, cluster_idx, projection_idx, vprime: (graph, 26 * cluster_idx + projection_idx)).shuffle(
        buffer_size=260)
    _images = dataset.map(
        lambda graph, img, cluster_idx, projection_idx, vprime: (img, 26 * cluster_idx + projection_idx)).shuffle(
        buffer_size=260)

    # Zip the shuffled datasets back together so typically the index of the graph and image don't match.
    shuffled_dataset = tf.data.Dataset.zip((_graphs, _images))  # ((graph, idx1), (img, idx2))

    # Reshape the dataset to the graph, the image and a yes or no whether the indices are the same
    # So ((graph, idx1), (img, idx2)) --> (graph, img, True/False)
    shuffled_dataset = shuffled_dataset.map(lambda ds1, ds2: (ds1[0], ds2[0], ds1[1] == ds2[1]))  # (graph, img, yes/no)

    # Take the subset of the data where the graph and image don't correspond (which is most of the dataset, since it's shuffled)
    shuffled_dataset = shuffled_dataset.filter(lambda graph, img, c: ~c)

    # Transform the True/False class into 1/0 integer
    shuffled_dataset = shuffled_dataset.map(lambda graph, img, c: (graph, img, tf.cast(c, tf.int32)))

    # Use the original dataset where all indices correspond and give them class True and turn that into an integer
    # So every instance gets class 1
    nonshuffeled_dataset = dataset.map(lambda graph, img, cluster_idx, projection_idx, vprime: (
    graph, img, tf.constant(1, dtype=tf.int32)))  # (graph, img, yes)

    # For the training data, take a sample either from the correct or incorrect combinations of graphs and images
    nn_dataset = tf.data.experimental.sample_from_datasets([shuffled_dataset, nonshuffeled_dataset])
    return nn_dataset

def main(data_dir):
    tfrecords = glob.glob(os.path.join(data_dir,'*.tfrecords'))  # list containing tfrecord files
    print(f'Number of tfrecord files : {len(tfrecords)}')

    splitpoint = int(0.60 * len(tfrecords))
    train_tfrecords = tfrecords[:splitpoint]
    test_tfrecords = tfrecords[splitpoint:]

    train_dataset = tfrecords_to_dataset(train_tfrecords)
    test_dataset = tfrecords_to_dataset(test_tfrecords)

    strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=2, memory_limit=2000)

    config = dict(model_type='model1',
                  model_parameters=dict(num_layers=3),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict())

    with strategy.scope():
        train_one_epoch = build_training(**config)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=10,
                          early_stop_patience=3,
                          checkpoint_dir='test_checkpointing',
                          debug=False)

    # # Use one half as train dataset and the other half as test dataset
    # train_dataset = train_dataset.shard(2,0)
    # test_dataset = train_dataset.shard(2,1)

    # min_val = 0
    # max_val = 0
    #
    # for (graph, img, c) in iter(train_dataset):
    #     max_val = tf.maximum(max_val, tf.reduce_max(img))
    #     print(max_val, c)
    #
    # # Instantiate a model (based on a relational network, see also graph_net_utils)
    # model = Model()

    # # Loss function that uses the batch (which contains the desired classification c)
    # # and the model_outputs (classification determined by the model).
    # def loss(model_outputs, batch):
    #     (graph, img, c) = batch
    #     # print(f'Model outputs = {model_outputs}')
    #     # print(f'Desired outputs = {c[None,None]}')
    #     return tf.reduce_mean(tf.losses.binary_crossentropy(c[None,None],model_outputs, from_logits=True))# tf.math.sqrt(tf.reduce_mean(tf.math.square(rank - tf.nn.sigmoid(model_outputs[:, 0]))))
    #
    # opt = snt.optimizers.Adam(0.0001)
    # # loss to evaluate the model, opt to determine how to improve the model
    # training = TrainOneEpoch(model, loss, opt)
    #
    # # Train the model for 10 epochs
    # vanilla_training_loop(train_dataset, training, 30, True)

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
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')
    main(tfrec_dir)
