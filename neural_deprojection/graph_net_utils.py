import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from graph_nets.modules import SelfAttention
import tensorflow_probability as tfp
from graph_nets import utils_tf, blocks
import tqdm
import sonnet as snt
from sonnet.src.base import Optimizer, Module
from sonnet.src import utils, once
import numpy as np
import six
import abc
import contextlib
import os
from collections import namedtuple
from functools import partial, reduce
import itertools

def sort_graph(graphs:GraphsTuple, node_ids, edge_ids)->GraphsTuple:
    """
    Sorts the nodes and edges of a batch of graphs such that they are ordered in blocks.

    Args:
        graphs: GraphsTuple
        node_ids: int32 1D array giving graph index of each node
        edge_ids: int32 1D array giving graph index of each edge

    Returns:
        GraphsTuple
    """
    #sort the nodes into blocks
    node_sort = tf.argsort(node_ids)
    nodes = graphs.nodes[node_sort]
    senders = node_sort[graphs.senders]
    receivers = node_sort[graphs.receivers]
    #sort edges into blocks
    edge_sort = tf.argsort(edge_ids)
    senders = senders[edge_sort]
    receivers = receivers[edge_sort]
    edges = graphs.edges[edge_sort]
    return graphs.replace(nodes=nodes,
                          edges=edges,
                          senders=senders,
                          receivers=receivers)

def replicate_graph(graph, num_repeats):
    if isinstance(num_repeats, int):
        return utils_tf.concat([graph]*num_repeats, axis=0)
    def _repeat(tensor):
        if tensor is None:
            return None
        shape = get_shape(tensor)
        return tf.tile(tensor,[num_repeats]+[1]*(len(shape) - 1))
    graph = graph.map(_repeat, ('nodes', 'edges', 'senders', 'receivers', 'globals', 'n_node', 'n_edge'))
    offsets = utils_tf._compute_stacked_offsets(graph.n_node, graph.n_edge)
    if graph.senders is not None:
        graph = graph.replace(senders = graph.senders + offsets)
    if graph.receivers is not None:
        graph = graph.replace(receivers = graph.receivers + offsets)
    return graph

def get_shape(tensor):
  """Returns the tensor's shape.

   Each shape element is either:
   - an `int`, when static shape values are available, or
   - a `tf.Tensor`, when the shape is dynamic.

  Args:
    tensor: A `tf.Tensor` to get the shape of.

  Returns:
    The `list` which contains the tensor's shape.
  """

  shape_list = tensor.shape.as_list()
  if all(s is not None for s in shape_list):
    return shape_list
  shape_tensor = tf.shape(tensor)
  return [shape_tensor[i] if s is None else s for i, s in enumerate(shape_list)]

def reconstruct_fields_from_gaussians(tokens, positions):
    """
    Computes the reconstruction of fields as a sum of spatial Gaussian basis functions.

        rho(x) = sum_i w_i * e^{-0.5 * (x - mu_i)^T @ R_i^T @ W_i^{-1} @ R_i (x - mu_i)}

        M = (R^T @ W^{-1} @ R)
        = L^T @ L

        (x - mu_i)^T @ M (x - mu_i)
        ((x - mu_i)^T @ L^T) @ (L @ (x - mu_i))

        dx = (L @ (x - mu))

        dx^T @ dx

        L = [a, 0, 0],
            [b, c, 0],
            [d, e, f]


    Args:
        tokens: [batch, num_gaussian_components, num_properties * 10]
        positions: [batch, n_nodes_per_graph, 3]

    Returns:
        [batch, n_nodes_per_graph, num_properties]
    """
    def _single_gaussian_property(arg):
        """

        Args:
            positions: [N, 3]
            weight: [C]
            mu: [C, 3]
            L: [C, 3, 3]

        Returns: [N]

        """
        positions, weight, mu, L = arg
        dx = (positions - mu[:, None, :])#C, N, 3
        dx = tf.einsum("cij,cnj->cni", L, dx )#C, N, 3
        maha = tf.einsum("cni,cni->cn",dx,dx)#C,N
        return tf.reduce_sum(weight[:, None] * tf.math.exp(-0.5 * maha), axis=0)#N

    def _single_batch_evaluation(arg):
        """
        Evaluation Gaussian for single graph
        Args:
            positions: [N, 3]
            tokens: [C, P*10]

        Returns:
            [N,P]
        """
        # vectorized_map only takes single inputs
        positions, tokens = arg
        P = tokens.shape[1]//10
        weights = tf.transpose(tokens[:, 0:P], (1, 0))#P, C
        mu = tf.transpose(tf.reshape(tokens[:, P:P*3+P], (-1, P, 3)), (1, 0, 2))#P,C,3
        L_flat = tf.transpose(tf.reshape(tokens[:, P*3+P:], (-1, P, 6)), (1,0,2))#P,C,6
        L = tfp.math.fill_triangular(L_flat)#P,C,3,3
        # tf.stack(P*[positions]) = P, N, 3
        properties = tf.vectorized_map(_single_gaussian_property, (tf.stack(P*[positions]), weights, mu, L))#P,N
        properties = tf.transpose(properties, (1,0)) #N, P
        return properties

    return tf.vectorized_map(_single_batch_evaluation, (positions, tokens))  # [batch, N, P]

def graph_batch_reshape(graphs:GraphsTuple)->GraphsTuple:
    """
    If each graph is exactly the same size, i.e. has the same number of nodes and edges,
    then you can reshape into batch form.

    Args:
        graph: GraphsTuple

    Returns:
        GraphsTuple with
            nodes: [n_graphs, n_node[0]//n_graphs,...]
            edges: [n_graphs, n_edge[0]//n_graphs,...]
            senders: [n_graphs, n_edge[0]//n_graphs]
            receivers: [n_graphs, n_edge[0]//n_graphs]
    """
    n_graphs = utils_tf.get_num_graphs(graphs)

    def _to_batched(tensor):
        if tensor is None:
            return tensor
        in_shape = get_shape(tensor)
        to_shape = [n_graphs, in_shape[0]//n_graphs] + in_shape[1:]
        new_tensor = tf.reshape(tensor, to_shape)
        return new_tensor

    return graphs.map(_to_batched,fields=('nodes','edges','senders','receivers'))


def graph_unbatch_reshape(graphs: GraphsTuple)->GraphsTuple:
    """
    Undoes `graph_batch_reshape`.

    Args:
        graph: GraphsTuple with
            nodes: [n_graphs, n_node[0]//n_graphs,...]
            edges: [n_graphs, n_edge[0]//n_graphs,...]
            senders: [n_graphs, n_edge[0]//n_graphs]
            receivers: [n_graphs, n_edge[0]//n_graphs]

    Returns:
        GraphsTuple with normal shaping of elements.
    """
    n_graphs = utils_tf.get_num_graphs(graphs)
    def _to_unbatched(tensor):
        if tensor is None:
            return tensor
        from_shape = get_shape(tensor)
        to_shape = [from_shape[1] * n_graphs] + from_shape[2:]
        new_tensor = tf.reshape(tensor, to_shape)
        return new_tensor

    return graphs.map(_to_unbatched, fields=('nodes', 'edges', 'senders', 'receivers'))


def gaussian_loss_function(gaussian_tokens, graphs:GraphsTuple):
    """
    Args:
        gaussian tokens: [batch, n_tokens, num_properties*10]

        graph: GraphsTuple
            graph.nodes: [n_node, num_positions + num_properties]

    Returns:
        scalar
    """

    graphs = graph_batch_reshape(graphs)
    positions = graphs.nodes[:,:,:3]#batch,n_node, 3
    input_properties = graphs.nodes[:,:,3:]#batch, n_node, n_prop

    with tf.GradientTape() as tape:
        field_properties = reconstruct_fields_from_gaussians(gaussian_tokens, positions)#N,P

    diff_properties = (input_properties - field_properties)

    return field_properties, tf.reduce_mean(tf.reduce_sum(tf.math.square(diff_properties),axis=1),axis=0)



def efficient_nn_index(query_positions, positions):
    """
    For each point in query_positions, find the index of the closest point in positions.

    :param query_positions: [N, D]
    :param positions: [M, D]
    :return: int64, indices of shape [N]
    """
    def _nearest_neighbour_index(state, point):
        return tf.argmin(tf.reduce_sum(tf.math.square(point - positions),axis=1))

    results = tf.scan(_nearest_neighbour_index,query_positions,initializer=tf.zeros((),dtype=tf.int64))
    return results


@six.add_metaclass(abc.ABCMeta)
class AbstractModule(snt.Module):
    """Makes Sonnet1-style childs from this look like a Sonnet2 module."""

    def __init__(self, *args, **kwargs):
        super(AbstractModule, self).__init__(*args, **kwargs)
        self.__call__.__func__.__doc__ = self._build.__doc__  # pytype: disable=attribute-error

    # In snt2 calls to `_enter_variable_scope` are ignored.
    @contextlib.contextmanager
    def _enter_variable_scope(self, *args, **kwargs):
        yield None

    def __call__(self, *args, **kwargs):
        return self._build(*args, **kwargs)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        """Similar to Sonnet 1 ._build method."""


class TrainOneEpoch(Module):
    _model: AbstractModule
    _opt: Optimizer

    def __init__(self, model: AbstractModule, loss, opt: Optimizer, strategy: tf.distribute.MirroredStrategy = None,
                 name=None):
        super(TrainOneEpoch, self).__init__(name=name)
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.minibatch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._model = model
        self._learn_variables = None
        self._model.epoch = self.epoch
        self._model.step = self.minibatch
        self._opt = opt
        self._loss = loss
        self._strategy = strategy
        self._checkpoint = tf.train.Checkpoint(module=model)

    @property
    def strategy(self) -> tf.distribute.MirroredStrategy:
        return self._strategy

    @property
    def model(self):
        return self._model

    @property
    def opt(self):
        return self._opt

    def loss(self, model_output, batch):
        return self._loss(model_output, batch)

    # def summarize(self, model_output, batch):
    #     summaries = model_output['summaries']
    #     for key in summaries:
    #         summary_func = summaries[key][0]
    #         summary_input = summaries[key][1]


    def train_step(self, batch):
        """
        Trains on a single batch.

        Args:
            batch: user defined batch from a dataset.

        Returns:
            loss
        """
        with tf.GradientTape() as tape:
            if not isinstance(batch, (list, tuple)):
                batch = (batch,)
            model_output = self.model(*batch)
            loss = self.loss(model_output, batch)
            # summaries = self.summarize(model_output, batch)
        if self._learn_variables is None:
            params = self.model.trainable_variables
        else:
            params = self._learn_variables
        grads = tape.gradient(loss, params)

        if self.strategy is not None:
            replica_ctx = tf.distribute.get_replica_context()
            grads = replica_ctx.all_reduce("mean", grads)
        # for (param, grad) in zip(params, grads):
        #     if grad is not None:
        #         tf.summary.histogram(param.name + "_grad", grad, step=self.minibatch)
        self.opt.apply(grads, params)
        return loss

    def one_epoch_step(self, train_dataset):
        """
        Updates a model with one epoch of train_one_epoch, and returns a dictionary of values to monitor, i.e. metrics.

        Returns:
            average loss
        """
        self.epoch.assign_add(1)
        # metrics = None
        loss = 0.
        num_batches = 0.
        for train_batch in train_dataset:
            self.minibatch.assign_add(1)
            if self.strategy is not None:
                _loss = self.strategy.run(self.train_step, args=(train_batch,))
                _loss = self.strategy.reduce("sum", _loss, axis=None)
            else:
                _loss = self.train_step(train_batch)
            tf.summary.scalar('mini_batch_loss', _loss, step=self.minibatch)
            loss += _loss
            num_batches += 1.
        tf.summary.scalar('epoch_loss', loss / num_batches, step=self.epoch)
        return loss / num_batches

    def evaluate(self, test_dataset):
        loss = 0.
        num_batches = 0.
        for test_batch in test_dataset:
            if not isinstance(test_batch, (list, tuple)):
                test_batch = (test_batch,)
            if self.strategy is not None:
                model_output = self.strategy.run(self.model, args=(test_batch,))
                _loss = self.strategy.run(self.loss, args=(model_output, test_batch))
                loss += self.strategy.reduce("sum", _loss, axis=0)
            else:
                model_output = self.model(*test_batch)
                loss += self.loss(model_output, test_batch)
            num_batches += 1.
        tf.summary.scalar('loss', loss / num_batches, step=self.epoch)
        return loss / num_batches


def get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1,
                              memory_limit=2000) -> tf.distribute.MirroredStrategy:
    # trying to set GPU distribution
    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    physical_cpus = tf.config.experimental.list_physical_devices("CPU")
    if len(physical_gpus) > 0 and not use_cpus:
        print("Physical GPUS: {}".format(physical_gpus))
        if logical_per_physical_factor > 1:
            for dev in physical_gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    dev,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)] * logical_per_physical_factor
                )

        gpus = tf.config.experimental.list_logical_devices("GPU")

        print("Logical GPUs: {}".format(gpus))

        strategy = snt.distribute.Replicator(
            ["/device:GPU:{}".format(i) for i in range(len(gpus))],
            tf.distribute.ReductionToOneDevice("GPU:0"))
    else:
        print("Physical CPUS: {}".format(physical_cpus))
        if logical_per_physical_factor > 1:
            for dev in physical_cpus:
                tf.config.experimental.set_virtual_device_configuration(
                    dev,
                    [tf.config.experimental.VirtualDeviceConfiguration()] * logical_per_physical_factor
                )

        cpus = tf.config.experimental.list_logical_devices("CPU")
        print("Logical CPUs: {}".format(cpus))

        strategy = snt.distribute.Replicator(
            ["/device:CPU:{}".format(i) for i in range(len(cpus))],
            tf.distribute.ReductionToOneDevice("CPU:0"))

    return strategy

def test_checkpoint_restore():

    class TestClassA(AbstractModule):
        def __init__(self, name=None):
            super(TestClassA, self).__init__(name=name)
            self.mlp = snt.nets.MLP([1], name='mlp_a')
        def _build(self, input):
            return self.mlp(input)

    class TestClassB(AbstractModule):
        def __init__(self, name=None):
            super(TestClassB, self).__init__(name=name)
            self.a = TestClassA()

        def _build(self, input):
            return self.a(input) + input

    b = TestClassB()

    input = tf.ones((5,1))
    output = b(input)

    print(b.trainable_variables)

    checkpoint_dir = 'test_ckpt_2'

    ### save b
    checkpoint = tf.train.Checkpoint(module=b.a)#saving b.a means that we can't load b but only a later
    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=3,
                                         checkpoint_name=b.__class__.__name__)
    manager.save()

    ### restore into b
    b = TestClassB()
    input = tf.ones((5, 1))
    output = b(input)
    print("Before restore", b.trainable_variables)
    checkpoint = tf.train.Checkpoint(module=b)#won't work because we saved from a
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print(f"Restored from {manager.latest_checkpoint}")
    print("After restore", b.trainable_variables)

    ### restore into a
    a = TestClassA()
    input = tf.ones((5, 1))
    output = a(input)
    print("Before restore", a.trainable_variables)
    checkpoint = tf.train.Checkpoint(module=a)#will work because we saved from a
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    # print(restore_checkpoint_from_other_model(manager.latest_checkpoint, a.trainable_variables))
    print(f"Restored from {manager.latest_checkpoint}")
    print("After restore", a.trainable_variables)
    # Note, no re.match happens.


def vanilla_training_loop(train_one_epoch: TrainOneEpoch, training_dataset, test_dataset=None, num_epochs=1,
                          early_stop_patience=None, checkpoint_dir=None, log_dir=None, save_model_dir=None, variables=None, debug=False):
    """
    A simple training loop.

    Args:
        train_one_epoch: TrainOneEpoch object
        training_dataset: training dataset, elements are expected to be tuples of model input
        test_dataset: test dataset, elements are expected to be tuples of model input
        num_epochs: int
        early_stop_patience: int, how many epochs with non-decreasing loss before stopping.
        checkpoint_dir: where to save checkpoints
        log_dir: where to log to tensorboard
        save_model_dir: where to save the model
        variables: optional, if not None then which variables to train on, defaults to model.trainable_variables.
        debug: bool, whether to not compile to faster code but allow debugging.

    Returns:

    """
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if save_model_dir is not None:
        os.makedirs(save_model_dir, exist_ok=True)

    if train_one_epoch.strategy is not None:
        training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
        if test_dataset is not None:
            test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

    if variables is not None:
        train_one_epoch._learn_variables = variables
    else:
        train_one_epoch._learn_variables = None

    # We'll turn the one_epoch_step function which updates our models into a tf.function using
    # autograph. This makes train_one_epoch much faster. If debugging, you can turn this
    # off by setting `debug = True`.
    step = train_one_epoch.one_epoch_step
    evaluate = train_one_epoch.evaluate
    if not debug:
        step = tf.function(step)
        evaluate = tf.function(evaluate)

    fancy_progress_bar = tqdm.tqdm(range(num_epochs),
                                   unit='epochs',
                                   position=0)
    early_stop_min_loss = np.inf
    early_stop_interval = 0

    train_log_dir = os.path.join(log_dir, "train")
    test_log_dir = os.path.join(log_dir, "test")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    checkpoint = tf.train.Checkpoint(module=train_one_epoch.model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3,
                                         checkpoint_name=train_one_epoch.model.__class__.__name__)
    if manager.latest_checkpoint is not None:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Restored from {manager.latest_checkpoint}")
    for step_num in fancy_progress_bar:
        with train_summary_writer.as_default():
            loss = step(iter(training_dataset))
            if save_model_dir is not None:
                tf.saved_model.save(train_one_epoch.model, save_model_dir)
        tqdm.tqdm.write(
            '\nEpoch = {}/{} (loss = {:.02f})'.format(
                train_one_epoch.epoch.numpy(), num_epochs, loss))
        if test_dataset is not None:
            with test_summary_writer.as_default():
                test_loss = evaluate(iter(test_dataset))
            tqdm.tqdm.write(
                '\n\tTest loss = {:.02f})'.format(test_loss))
            if early_stop_patience is not None:
                if test_loss <= early_stop_min_loss:
                    early_stop_min_loss = test_loss
                    early_stop_interval = 0
                    manager.save()
                else:
                    early_stop_interval += 1
                if early_stop_interval == early_stop_patience:
                    tqdm.tqdm.write(
                        '\n\tStopping Early')
                    break
            else:
                manager.save()
        else:
            manager.save()
    train_summary_writer.close()
    test_summary_writer.close()


def batch_dataset_set_graph_tuples(*, all_graphs_same_size=False, dataset: tf.data.Dataset,
                                   batch_size) -> tf.data.Dataset:
    """

    Args:
        dataset: dataset of GraphTuple containing only a single graph.
        batch_size:
        all_graphs_same_size:

    Returns:

    """
    if not all_graphs_same_size:
        raise ValueError("Only able to batch graphs with the same number of nodes and edges.")

    TempGraphTuple = namedtuple('TempGraphTuple',
                                ['nodes', 'edges', 'senders', 'receivers', 'globals', 'n_node', 'n_edge'])

    def _concat_graph_from_batched(*args):
        _output_args = []
        for arg in args:
            if isinstance(arg, TempGraphTuple):
                graph: TempGraphTuple = arg
                # # nodes: [batch_size,nnodes_max,Fnodes]
                # graph.nodes.set_shape([batch_size, None, None])
                # # edges: [batch_size,nedges_max,Fedges]
                # graph.edges.set_shape([batch_size, None, None])
                # # senders: [batch_size, nedges_max]
                # graph.senders.set_shape([batch_size, None])
                # # receivers: [batch_size, nedges_max]
                # graph.receivers.set_shape([batch_size, None])
                # # globals: [batch_size, 1, Fglobals]
                # graph.globals.set_shape([batch_size, None, None])
                # # nnodes: [batch_size, 1]
                # graph.n_node.set_shape([batch_size, None])
                # # nedges: [batch_size, 1]
                # graph.n_edge.set_shape([batch_size, None])

                nodes = tf.unstack(graph.nodes, num=batch_size, name='nodes')
                edges = tf.unstack(graph.edges, num=batch_size, name='edges')
                senders = tf.unstack(graph.senders, num=batch_size, name='senders')
                receivers = tf.unstack(graph.receivers, num=batch_size, name='receivers')
                _globals = tf.unstack(graph.globals, num=batch_size, name='globals')
                n_node = tf.unstack(graph.n_node, num=batch_size, name='n_node')
                n_edge = tf.unstack(graph.n_edge, num=batch_size, name='n_edge')
                graphs = []
                for _nodes, _edges, _senders, _receivers, _n_node, _n_edge, __globals in zip(nodes, edges, senders,
                                                                                             receivers, n_node, n_edge,
                                                                                             _globals):
                    graphs.append(GraphsTuple(nodes=_nodes,
                                              edges=_edges,
                                              globals=__globals,
                                              receivers=_receivers,
                                              senders=_senders,
                                              n_node=_n_node,
                                              n_edge=_n_edge))
                    # print(graphs[-1])
                graphs = utils_tf.concat(graphs, axis=0)
                _output_args.append(graphs)
            else:
                _output_args.append(arg)
        if len(_output_args) == 1:
            return _output_args[0]
        return tuple(_output_args)

    def _to_temp_graph_tuple(*args):
        _output_args = []
        for arg in args:
            if isinstance(arg, GraphsTuple):
                _output_args.append(TempGraphTuple(**arg._asdict()))
            else:
                _output_args.append(arg)
        return tuple(_output_args)

    return dataset.map(_to_temp_graph_tuple).padded_batch(batch_size=batch_size, drop_remainder=True).map(
        _concat_graph_from_batched)


def test_batch_dataset_set_graph_tuples():
    graphs = []
    images = []
    n_node = 5
    n_edge = n_node * 2
    for i in range(5, 11):
        graph = GraphsTuple(nodes=np.random.normal(size=(n_node, 2)).astype(np.float32),
                            edges=np.random.normal(size=(n_edge, 3)).astype(np.float32),
                            senders=np.random.randint(n_node, size=(n_edge,)).astype(np.int32),
                            receivers=np.random.randint(n_node, size=(n_edge,)).astype(np.int32),
                            globals=np.random.normal(size=(1, 4)).astype(np.float32),
                            n_node=[n_node],
                            n_edge=[n_edge]
                            )
        graphs.append(graph)
        images.append(np.random.normal(size=(24, 24, 1)))
    dataset = tf.data.Dataset.from_generator(lambda: iter(zip(graphs, images)),
                                             output_types=
                                             (GraphsTuple(nodes=tf.float32,
                                                          edges=tf.float32,
                                                          senders=tf.int32,
                                                          receivers=tf.int32,
                                                          globals=tf.float32,
                                                          n_node=tf.int32,
                                                          n_edge=tf.int32
                                                          ),
                                              tf.float32),
                                             output_shapes=(GraphsTuple(nodes=tf.TensorShape([None, None]),
                                                                        edges=tf.TensorShape([None, None]),
                                                                        senders=tf.TensorShape([None]),
                                                                        receivers=tf.TensorShape([None]),
                                                                        globals=tf.TensorShape([None, None]),
                                                                        n_node=tf.TensorShape([None]),
                                                                        n_edge=tf.TensorShape([None])
                                                                        ),
                                                            tf.TensorShape([None, None, None])))

    # for graph in iter(dataset):
    #     print(graph.receivers.dtype)
    dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=2)
    for graph, image in iter(dataset):
        assert graph.nodes.shape == (n_node * 2, 2)
        assert graph.edges.shape == (n_edge * 2, 3)
        assert graph.globals.shape == (2, 4)
        assert image.shape == (2, 24, 24, 1)


def batched_tensor_to_fully_connected_graph_tuple_dynamic(nodes_tensor, pos=None, globals=None):
    """
    Convert tensor with batch dim to batch of GraphTuples.
    :param nodes_tensor: [B, num_nodes, F] Tensor to turn into nodes. F must be statically known.
    :param pos: [B, num_nodes, D] Tensor to calculate edge distance using difference. D must be statically known.
    :param globals: [B, G] Tensor to use as global. G must be statically known.
    :return: GraphTuple with batch of fully connected graphs
    """
    shape = tf.shape(nodes_tensor)
    batch_size, num_nodes = shape[0], shape[1]
    F = nodes_tensor.shape.as_list()[-1]
    graphs_with_nodes = GraphsTuple(n_node=tf.fill([batch_size], num_nodes),
                                    n_edge=tf.fill([batch_size], 0),
                                    nodes=tf.reshape(nodes_tensor, [batch_size * num_nodes, F]),
                                    edges=None, globals=None, receivers=None, senders=None)
    graphs_tuple_with_nodes_connectivity = utils_tf.fully_connect_graph_dynamic(
        graphs_with_nodes, exclude_self_edges=False)

    if pos is not None:
        D = pos.shape.as_list()[-1]
        graphs_with_position = graphs_tuple_with_nodes_connectivity.replace(
            nodes=tf.reshape(pos, [batch_size * num_nodes, D]))
        edge_distances = (
                blocks.broadcast_receiver_nodes_to_edges(graphs_with_position) -
                blocks.broadcast_sender_nodes_to_edges(graphs_with_position))
        graphs_with_nodes_edges = graphs_tuple_with_nodes_connectivity.replace(edges=edge_distances)
    else:
        graphs_with_nodes_edges = utils_tf.set_zero_edge_features(graphs_tuple_with_nodes_connectivity, 1,
                                                                  dtype=nodes_tensor.dtype)

    if globals is not None:
        graphs_with_nodes_edges_globals = graphs_with_nodes_edges.replace(globals=globals)
    else:
        graphs_with_nodes_edges_globals = utils_tf.set_zero_global_features(
            graphs_with_nodes_edges, global_size=1)

    return graphs_with_nodes_edges_globals


def build_log_dir(base_log_dir, config):
    """
    Builds log dir.

    Args:
        base_log_dir: where all logs should be based from.
        config: dict with following structure

            Example config:

            config = dict(model_type='model1',
                  model_parameters=dict(num_layers=3),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict(loss_type='cross_entropy'))

    Returns:
        log_dir representing this config
    """
    log_dir_subdir = stringify_config(config)
    log_dir = os.path.join(base_log_dir, log_dir_subdir)
    return log_dir


def build_checkpoint_dir(base_checkpoint_dir, config):
    """
    Builds log dir.

    Args:
        base_checkpoint_dir: where all logs should be based from.
        config: dict with following structure

            Example config:

            config = dict(model_type='model1',
                  model_parameters=dict(num_layers=3),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict(loss_type='cross_entropy'))

    Returns:
        checkpoint_dir representing this config
    """
    checkpoint_dir_subdir = stringify_config(config)
    checkpoint_dir = os.path.join(base_checkpoint_dir, checkpoint_dir_subdir)
    return checkpoint_dir


def stringify_config(config):
    def transform_key(key: str):
        # use every other letter of key as name
        keys = key.split("_")
        parts = []
        for key in keys:
            vowels = 'aeiou'
            for v in vowels:
                key = key[0] + key[1:].replace(v, '')
            parts.append(key)
        return "".join(parts)

    def transform_value(value):
        if isinstance(value, int):
            return str(value)
        if isinstance(value, (float)):
            return "{:.1e}".format(value)
        else:
            return value

    def stringify_dict(d):
        return "|{}|".format(",".join(["{}={}".format(transform_key(k), transform_value(d[k]))
                                       for k in sorted(d.keys())]))

    model_type = f"|{config['model_type']}|"
    model_parameters = stringify_dict(config['model_parameters'])
    optimizer_parameters = stringify_dict(config['optimizer_parameters'])
    loss_parameters = stringify_dict(config['loss_parameters'])
    subdir = "".join([model_type, model_parameters, optimizer_parameters, loss_parameters])
    return subdir


def tf_ravel_multi_index(multi_index, dims):
    """
    Equivalent of np.ravel_multi_index.

    Args:
        multi_index: [N, D]
        dims: [D]

    Returns: [N]
    """
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)  # D
    return tf.reduce_sum(multi_index * strides[:, None], axis=0)  # D,N -> N


def histogramdd(sample, bins=10, weights=None, density=None):
    """
    Compute histogram over D-dimensional samples, potentially summing weights.

    Args:
        sample: [N, D] or tuple of array[N]
        bins: int
        weights: [N, P], optionally [N]
        density: bool

    Returns:
        [bins-1]*D + [P], optionally missing [P] if weights is 1-D

    """
    if isinstance(sample, (tuple, list)):
        sample = tf.stack(sample, axis=-1)
    N, D = get_shape(sample)
    if weights is None:
        weights = tf.ones((N,))

    if not isinstance(bins, int):
        raise ValueError("Only support integer bins")

    bin_idx_by_dim = D * [None]
    nbins = [None]*D
    bin_edges_by_dim = D * [None]
    dedges = D * [None]

    vmin = tf.reduce_min(sample, axis=0)
    vmax = tf.reduce_max(sample, axis=0)
    bin_edges = tf.cast(tf.linspace(0., 1., bins + 1), sample.dtype)[:, None] * (vmax - vmin) + vmin
    for i in range(D):
        dim_bin_edges = bin_edges[:, i]
        bin_idx = tf.searchsorted(dim_bin_edges, sample[:, i], side='right')
        bin_idx = tf.where(sample[:, i] == dim_bin_edges[-1], bin_idx - 1, bin_idx)
        bin_idx_by_dim[i] = bin_idx
        nbins[i] = dim_bin_edges.shape[0] + 1
        bin_edges_by_dim[i] = dim_bin_edges
        dedges[i] = bin_edges_by_dim[i][1:] - bin_edges_by_dim[i][:-1]

    minlength = maxlength = tf.constant(np.prod(nbins), dtype=tf.int32)
    nbins = tf.constant(nbins, dtype=tf.int32)
    xy = tf_ravel_multi_index(bin_idx_by_dim, nbins)
    def _sum_weights(weights):
        hist = tf.math.bincount(tf.cast(xy, tf.int32), weights,
                                minlength=minlength, maxlength=maxlength)
        hist = tf.reshape(hist, nbins)
        core = D * (slice(1, -1),)
        hist = hist[core]
        return hist

    if len(get_shape(weights)) == 2:
        hist = tf.vectorized_map(_sum_weights, tf.transpose(weights, (1,0)), weights) #[P] + [bins]*D
        perm = list(range(len(hist.shape)))
        perm.append(perm[0])
        del perm[0]
        hist = tf.transpose(hist, perm)#[bins]*D + [P]
    else:
        hist = _sum_weights(weights)

    if density:
        raise ValueError('density=True not supported.')
        # s = sum(hist)
        # for i in range(D):
        #     _shape = np.ones(D, int)
        #     _shape[i] = nbins[i] - 2
        #     hist = hist / tf.maximum(1, tf.cast(tf.reshape(dedges[i], _shape), hist.dtype))
        # hist /= tf.cast(s, hist.dtype)

    return hist, bin_edges_by_dim

class GraphDecoder(AbstractModule):
    def __init__(self, output_property_size, name=None):
        super(GraphDecoder, self).__init__(name=name)
        self.output_property_size = output_property_size


    def _build(self, encoded_graph, positions, **kwargs):
        mean_position = tf.reduce_mean(positions, axis=0)
        centroid_index = tf.argmin(tf.reduce_sum(tf.math.square(positions - mean_position), axis=1))

class MultiHeadLinear(AbstractModule):
    """Linear module, optionally including bias."""

    def __init__(self,
                 output_size: int,
                 num_heads: int = 1,
                 with_bias: bool = True,
                 w_init=None,
                 b_init=None,
                 name=None):
        """Constructs a `Linear` module.

        Args:
          output_size: Output dimensionality.
          with_bias: Whether to include bias parameters. Default `True`.
          w_init: Optional initializer for the weights. By default the weights are
            initialized truncated random normal values with a standard deviation of
            `1 / sqrt(input_feature_size)`, which is commonly used when the inputs
            are zero centered (see https://arxiv.org/abs/1502.03167v3).
          b_init: Optional initializer for the bias. By default the bias is
            initialized to zero.
          name: Name of the module.
        """
        super(MultiHeadLinear, self).__init__(name=name)
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.num_heads = num_heads
        if with_bias:
            self.b_init = b_init if b_init is not None else snt.initializers.Zeros()
        elif b_init is not None:
            raise ValueError("When not using a bias the b_init must be None.")

    @once.once
    def _initialize(self, inputs: tf.Tensor):
        """Constructs parameters used by this module."""
        utils.assert_minimum_rank(inputs, 2)

        input_size = inputs.shape[-1]
        if input_size is None:  # Can happen inside an @tf.function.
            raise ValueError("Input size must be specified at module build time.")

        self.input_size = input_size

        if self.w_init is None:
            # See https://arxiv.org/abs/1502.03167v3.
            stddev = 1 / tf.math.sqrt(self.input_size * 1.0)
            self.w_init = snt.initializers.TruncatedNormal(stddev=stddev)

        self.w = tf.Variable(
            self.w_init([self.num_heads, self.input_size, self.output_size], inputs.dtype),
            name="w")

        if self.with_bias:
            self.b = tf.Variable(
                self.b_init([self.num_heads, self.output_size], inputs.dtype), name="b")

    def _build(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Parallel computation of multi-head linear.

        Args:
            inputs: [n_nodes, node_size]

        Returns:
            [n_nodes, num_heads, output_size]
        """
        self._initialize(inputs)

        # [num_nodes, node_size].[num_heads, node_size, output_size] -> [num_nodes, num_heads, output_size]
        outputs = tf.einsum('ns,hso->nho', inputs, self.w, optimize='optimal')
        if self.with_bias:
            outputs = tf.add(outputs, self.b)
        return outputs


class CoreNetwork(AbstractModule):
    """
    Core network which can be used in the EncodeProcessDecode network. Consists of a (full) graph network block
    and a self attention block.
    """

    def __init__(self,
                 num_heads,
                 multi_head_output_size,
                 input_node_size,
                 name=None):
        super(CoreNetwork, self).__init__(name=name)
        self.num_heads = num_heads
        self.multi_head_output_size = multi_head_output_size

        self.output_linear = snt.Linear(output_size=input_node_size)
        self.FFN = snt.nets.MLP([32, input_node_size], activate_final=False)  # Feed forward network
        self.normalization = lambda x: (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
        self.ln1 = snt.LayerNorm(axis=1, eps=1e-6, create_scale=True, create_offset=True)
        self.ln2 = snt.LayerNorm(axis=1, eps=1e-6, create_scale=True, create_offset=True)

        self.v_linear = MultiHeadLinear(output_size=multi_head_output_size, num_heads=num_heads)  # values
        self.k_linear = MultiHeadLinear(output_size=multi_head_output_size, num_heads=num_heads)  # keys
        self.q_linear = MultiHeadLinear(output_size=multi_head_output_size, num_heads=num_heads)  # queries
        self.self_attention = SelfAttention()

    def _build(self, latent, positions=None):
        node_values = self.v_linear(latent.nodes)
        node_keys = self.k_linear(latent.nodes)
        node_queries = self.q_linear(latent.nodes)
        attended_latent = self.self_attention(node_values=node_values,
                                              node_keys=node_keys,
                                              node_queries=node_queries,
                                              attention_graph=latent)
        output_nodes = tf.reshape(attended_latent.nodes, (-1, self.num_heads * self.multi_head_output_size))
        output_nodes = self.ln1(self.output_linear(output_nodes) + latent.nodes)
        output_nodes = self.ln2(self.FFN(output_nodes))
        output_graph = latent.replace(nodes=output_nodes)
        if positions is not None:
            prepend_nodes = tf.concat([positions, output_graph.nodes[:, 3:]], axis=1)
            output_graph = output_graph.replace(nodes=prepend_nodes)
        return output_graph

# interpolate on nd-array


_nonempty_prod = partial(reduce, tf.multiply)
_nonempty_sum = partial(reduce, tf.add)

_INDEX_FIXERS = {
    'constant': lambda index, size: index,
    'nearest': lambda index, size: tf.clip_by_value(index, 0, size - 1),
    'wrap': lambda index, size: index % size,
}


def _round_half_away_from_zero(a):
    return tf.round(a)


def _nearest_indices_and_weights(coordinate):
    index = tf.cast(_round_half_away_from_zero(coordinate), tf.int32)
    weight = coordinate.dtype.type(1)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate):
    lower = tf.math.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = tf.cast(lower, tf.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _map_coordinates(input, coordinates, order, mode, cval):
    input = tf.convert_to_tensor(input)
    coordinates = [tf.convert_to_tensor(c) for c in coordinates]
    cval = tf.constant(cval, input.dtype)

    if len(coordinates) != len(get_shape(input)):
        raise ValueError('coordinates must be a sequence of length input.ndim, but '
                         '{} != {}'.format(len(coordinates), len(get_shape(input))))

    index_fixer = _INDEX_FIXERS.get(mode)

    if mode == 'constant':
        is_valid = lambda index, size: (0 <= index) & (index < size)
    else:
        is_valid = lambda index, size: True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError(
            'map_coordinates currently requires order<=1')

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinates, get_shape(input)):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        indices = tf.stack(indices, axis=-1)
        if all(valid is True for valid in validities):
            # fast path
            contribution = tf.gather_nd(input, indices)
        else:
            all_valid = reduce(tf.logical_and, validities)
            contribution = tf.where(all_valid, tf.gather_nd(input, indices), cval)
        outputs.append(_nonempty_prod(weights) * contribution)
    result = _nonempty_sum(outputs)
    return tf.cast(result, input.dtype)


def map_coordinates(input, coordinates, order, mode='constant', cval=0.0):
    return _map_coordinates(input, coordinates, order, mode, cval)

### grid graph onto voxel grid

def grid_properties(positions, properties, voxels_per_dimension):
    """
    We construct a meshgrid over the min/max range of positions, which act as the bin boundaries.
    Args:
        positions: [n_node_per_graph, 3]
        properties: [n_node_per_graph, num_properties]
        voxels_per_dimension: int

    Returns:
        [voxels_per_dimension, voxels_per_dimension, voxels_per_dimension, num_properties]
    """
    binned_properties, bin_edges = histogramdd(positions,
                                               bins=voxels_per_dimension,
                                               weights=properties)  # n_node_per_graph, num_properties

    bin_count, _ = histogramdd(positions,
                               bins=voxels_per_dimension) # n_node_per_graph

    # binned_properties /= bin_count[:, None]# n_node_per_graph, num_properties
    binned_properties = tf.where(bin_count[..., None] > 0, binned_properties/bin_count[..., None], 0.)
    return binned_properties

def grid_graphs(graphs, voxels_per_dimension):
    """
    Grid the nodes onto a voxel 3D meshgrid.

    Args:
        graphs: GraphTuples a batch of graphs

    Returns:
        [batch, voxels_per_dimension, voxels_per_dimension, voxels_per_dimension, num_properties]
    """
    batched_graphs = graph_batch_reshape(graphs)
    positions = batched_graphs.nodes[..., :3]#num_graphs, n_node_per_graph, 3
    properties = batched_graphs.nodes[..., 3:]#num_graphs, n_node_per_graph, num_properties

    gridded_graphs = tf.vectorized_map(lambda args: grid_properties(*args, voxels_per_dimension), (positions, properties))#[batch, voxels_per_dimension, voxels_per_dimension, voxels_per_dimension, num_properties]
    return gridded_graphs


def build_example_dataset(num_examples, batch_size, num_blobs=3, num_nodes=64**3, image_dim=256):
    """
    Creates an example dataset
    Args:
        num_examples: int, number of examples in an epoch
        batch_size: int, ideally should divide num_examples
        num_blobs: number of components in the 3D medium
        n_voxels_per_dimension: size of one cube dimension

    Returns:
        Dataset (GraphsTuple,
        image [batch, n_voxels_per_dimension, n_voxels_per_dimension, 1]

    """
    def _single_blob(positions):
        # all same weight
        weight = tf.random.uniform(shape=(), minval=1., maxval=1.)
        shift = tf.random.uniform(shape=(3,))
        lengthscale = tf.random.uniform(shape=(), minval=0.05, maxval=0.15)
        density = weight * tf.math.exp(-0.5 * tf.linalg.norm(positions - shift, axis=-1) ** 2 / lengthscale ** 2)
        return density

    def _map(i):
        positions = tf.random.uniform(shape=(num_nodes, 3))
        density = _single_blob(positions)
        for _ in range(num_blobs-1):
            density += _single_blob(positions)

        image = grid_properties(positions[:,:2], density[:, None], image_dim)#[image_dim, image_dim, 1]
        image += tfp.stats.percentile(image, 5) * tf.random.normal(shape=image.shape)
        image = tf.math.log(tf.math.maximum(image, 1e-5))

        log_properties = tf.math.log(tf.math.maximum(density, 1e-10))
        nodes = tf.concat([positions, log_properties[:, None]], axis=-1)
        n_node = tf.shape(nodes)[:1]
        data_dict = dict(nodes=nodes,n_node=n_node, n_edge=tf.zeros_like(n_node))
        return (data_dict, image)

    dataset = tf.data.Dataset.range(num_examples)
    dataset = dataset.map(_map).batch(batch_size)
    #batch fixing mechanism
    dataset = dataset.map(lambda data_dict, image: (batch_graph_data_dict(data_dict), image))
    dataset = dataset.map(lambda data_dict, image: (GraphsTuple(**data_dict,
                                                                edges=None, receivers=None, senders=None, globals=None), image))
    dataset = dataset.map(lambda batched_graphs, image: (graph_unbatch_reshape(batched_graphs), image))
    # dataset = dataset.cache()
    return dataset

def batch_graph_data_dict(batched_data_dict):
    """
    After running dataset.batch() on data_dict representation of GraphTuple, correct the batch dimensions.

    Args:
        batched_data_dict: dict(
            nodes[num_graphs, n_node_per_graph, F_nodes],
            edges[num_graphs, n_edge_per_graph, F_edges],
            senders[num_graphs, n_edge_per_graph],
            receivers[num_graphs, n_edge_per_graph],
            globals[num_graphs, 1, F_globals],
            n_node[num_graphs, 1],
            n_edge[num_graphs, 1])

    Returns:
        batched_data_dict representing a batched GraphTuple:
        dict(
            nodes[num_graphs, n_node_per_graph, F_nodes],
            edges[num_graphs, n_edge_per_graph, F_edges],
            senders[num_graphs, n_edge_per_graph],
            receivers[num_graphs, n_edge_per_graph],
            globals[num_graphs, F_globals],
            n_node[num_graphs],
            n_edge[num_graphs])
    """
    if "globals" in batched_data_dict.keys():
        batched_data_dict["globals"] = batched_data_dict["globals"][:,0,:]
    if "n_node" in batched_data_dict.keys():
        batched_data_dict['n_node'] = batched_data_dict['n_node'][:,0]
    if "n_edge" in batched_data_dict.keys():
        batched_data_dict['n_edge'] = batched_data_dict['n_edge'][:,0]
    return batched_data_dict


def temperature_schedule(num_embedding, num_epochs, S=100, t0=1., thresh=0.95):
    """
    Returns callable for temperature schedule.
    Assumes logits will be normalised, such that std(logits) = 1

    then the schedule will be,

    temp = max( final_temp, t0 * exp(alpha * i))
    where
    alpha = log(final_temp / t0) / num_epochs
    and
    final_temp is determined through a quick MC search.

    Args:
        num_embedding:
        num_epochs: int number of epochs after which to be at final temp.
        S: int number of samples to use in search
        t0: float, initial temperature
        thresh: float, mean maximum value when to consider one-hot

    Returns:
        callable(epoch: tf.int32) -> temperature:tf.float32
        Callable that takes epoch to (tf.int32) and get the temperature (tf.float32)
    """

    temp_array = np.exp(np.linspace(np.log(0.001), np.log(1.), 1000))

    def softmax(r, temp):
        return np.exp(r/temp)/np.sum(np.exp(r/temp), axis=-1, keepdims=True)

    _temp = np.min(temp_array)

    final_temp = t0
    while True:
        r = np.random.normal(size=(S, num_embedding))
        x = softmax(r, final_temp)
        _max = np.max(x, axis=-1)
        if np.mean(_max) > thresh:
            break
        final_temp *= 0.95

    t0 = tf.constant(t0, dtype=tf.float32)
    final_temp = tf.constant(final_temp, dtype=tf.float32)
    alpha = tf.constant(np.log(final_temp / t0)/num_epochs, dtype=tf.float32)
    def _get_temperature(step):
        step = tf.cast(tf.convert_to_tensor(step), dtype=tf.float32)
        return tf.math.maximum(final_temp, t0 * tf.math.exp(alpha * step))

    return _get_temperature


