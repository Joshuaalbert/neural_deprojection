import tensorflow as tf
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf, blocks
import tqdm
import sonnet as snt
from sonnet.src.base import Optimizer, Module
import numpy as np
import six
import abc
import contextlib
import os
from collections import namedtuple


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
        self.epoch = tf.Variable(0, dtype=tf.int64)
        self.minibatch = tf.Variable(0, dtype=tf.int64)
        self._model = model
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

    def train_step(self, batch):
        """
        Trains on a single batch.

        Args:
            batch: user defined batch from a dataset.

        Returns:
            loss
        """
        with tf.GradientTape() as tape:
            model_output = self.model(batch)
            loss = self.loss(model_output, batch)
        params = self.model.trainable_variables
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
        if self.strategy is not None:
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
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
        if self.strategy is not None:
            test_dataset = self.strategy.experimental_distribute_dataset(test_dataset)
        for test_batch in test_dataset:
            if self.strategy is not None:
                model_output = self.strategy.run(self.model, args=(test_batch,))
                _loss = self.strategy.run(self.loss, args=(model_output, test_batch))
                loss += self.strategy.reduce("sum", _loss, axis=0)
            else:
                model_output = self.model(test_batch)
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


def vanilla_training_loop(train_one_epoch: TrainOneEpoch, training_dataset, test_dataset=None, num_epochs=1,
                          early_stop_patience=None, checkpoint_dir=None, log_dir=None, debug=False):
    """
    Does simple training.

    Args:
        training_dataset: Dataset for training
        train_one_epoch: TrainOneEpoch
        num_epochs: how many epochs to train
        test_dataset: Dataset for testing
        early_stop_patience: Stops training after this many epochs where test dataset loss doesn't improve
        checkpoint_dir: where to save epoch results.
        debug: bool, whether to use debug mode.

    Returns:

    """
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
    if test_dataset is not None:
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

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

    checkpoint = tf.train.Checkpoint(module=train_one_epoch)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3,
                                         checkpoint_name=train_one_epoch.model.__class__.__name__)
    if manager.latest_checkpoint is not None:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    for step_num in fancy_progress_bar:
        with train_summary_writer.as_default():
            loss = step(iter(training_dataset))
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
    N, D = sample.shape

    if not isinstance(bins, int):
        raise ValueError("Only support integer bins")

    bin_idx_by_dim = D * [None]
    nbins = np.empty(D, int)
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

    xy = tf_ravel_multi_index(bin_idx_by_dim, nbins)
    hist = tf.math.bincount(tf.cast(xy, tf.int32), weights, minlength=nbins.prod(), maxlength=nbins.prod())
    hist = tf.reshape(hist, nbins)
    core = D * (slice(1, -1),)
    hist = hist[core]

    if density:
        raise ValueError('Density doesnt work')
        # s = sum(hist)
        # for i in range(D):
        #     _shape = np.ones(D, int)
        #     _shape[i] = nbins[i] - 2
        #     hist = hist / tf.maximum(1, tf.cast(tf.reshape(dedges[i], _shape), hist.dtype))
        # hist /= tf.cast(s, hist.dtype)

    return hist, bin_edges_by_dim
