import tensorflow as tf
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple


def connect_graph_dynamic(graph: GraphsTuple, is_edge_func, name="connect_graph_dynamic"):
    """
    Connects a graph using a boolean edge mask to create edges.

    Args:
        graph: GraphsTuple
        is_edge_func: callable(sender: int, receiver: int) -> bool, should broadcast
        name:

    Returns:
        connected GraphsTuple
    """
    utils_tf._validate_edge_fields_are_all_none(graph)

    with tf.name_scope(name):
        def body(i, senders, receivers, n_edge):
            edges = _create_functional_connect_edges_dynamic(graph.n_node[i], is_edge_func)
            # edges = create_edges_func(graph.n_node[i])
            return (i + 1, senders.write(i, edges['senders']),
                    receivers.write(i, edges['receivers']),
                    n_edge.write(i, edges['n_edge']))

        num_graphs = utils_tf.get_num_graphs(graph)
        loop_condition = lambda i, *_: tf.less(i, num_graphs)
        initial_loop_vars = [0] + [
            tf.TensorArray(dtype=tf.int32, size=num_graphs, infer_shape=False)
            for _ in range(3)  # senders, receivers, n_edge
        ]
        _, senders_array, receivers_array, n_edge_array = tf.while_loop(loop_condition, body, initial_loop_vars)

        n_edge = n_edge_array.concat()
        offsets = utils_tf._compute_stacked_offsets(graph.n_node, n_edge)
        senders = senders_array.concat() + offsets
        receivers = receivers_array.concat() + offsets
        senders.set_shape(offsets.shape)
        receivers.set_shape(offsets.shape)

        receivers.set_shape([None])
        senders.set_shape([None])

        num_graphs = graph.n_node.get_shape().as_list()[0]
        n_edge.set_shape([num_graphs])

        return graph.replace(senders=tf.stop_gradient(senders),
                             receivers=tf.stop_gradient(receivers),
                             n_edge=tf.stop_gradient(n_edge))


def _create_functional_connect_edges_dynamic(n_node, is_edge_func):
    """Creates complete edges for a graph with `n_node`.

    Args:
    n_node: (integer scalar `Tensor`) The number of nodes.
    is_edge_func: (bool) callable(sender, receiver) that returns tf.bool if connected. Must broadcast.

    Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
    """

    rng = tf.range(n_node)
    ind = is_edge_func(rng, rng[:, None])
    n_edge = tf.reduce_sum(tf.cast(ind, tf.int32))
    indicies = tf.where(ind)
    receivers = indicies[:, 0]
    senders = indicies[:, 1]

    receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
    senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
    n_edge = tf.reshape(n_edge, [1])

    return {'receivers': receivers, 'senders': senders, 'n_edge': n_edge}


def _create_autogressive_edges_from_nodes_dynamic(n_node, exclude_self_edges):
    """Creates complete edges for a graph with `n_node`.

    Args:
    n_node: (integer scalar `Tensor`) The number of nodes.
    exclude_self_edges: (bool) Excludes self-connected edges.

    Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
    """

    if exclude_self_edges:
        is_edge_func = lambda sender, receiver: receiver > sender
        output = _create_functional_connect_edges_dynamic(n_node, is_edge_func)
        output['n_edge'] = tf.fill([1], n_node * (n_node - 1) // 2)
    else:
        is_edge_func = lambda sender, receiver: receiver >= sender
        output = _create_functional_connect_edges_dynamic(n_node, is_edge_func)
        output['n_edge'] = tf.fill([1], n_node * (n_node - 1) // 2 + n_node)

    return output


def test_autoregressive_connect_graph_dynamic():
    graphs = GraphsTuple(nodes=tf.range(12)[:, None], n_node=tf.constant([12]), n_edge=tf.constant([0]),
                         edges=None, receivers=None, senders=None, globals=None)
    graphs = autoregressive_connect_graph_dynamic(graphs, exclude_self_edges=False)
    for s, r in zip(graphs.senders.numpy(), graphs.receivers.numpy()):
        print(f"{s} -> {r}")
    import networkx as nx
    G = nx.MultiDiGraph()
    for sender, receiver in zip(graphs.senders.numpy(), graphs.receivers.numpy()):
        G.add_edge(sender, receiver)
    nx.drawing.draw_circular(G, with_labels=True, node_color=(0, 0, 0), font_color=(1, 1, 1), font_size=25,
                             node_size=1000, arrowsize=30, )
    import pylab as plt
    plt.show()


def autoregressive_connect_graph_dynamic(graph, exclude_self_edges=False):
    """Adds edges to a graph by auto-regressively the nodes.

    This method does not require the number of nodes per graph to be constant,
    or to be known at graph building time.

    Args:
    graph: A `graphs.GraphsTuple` with `None` values for the edges, senders and
      receivers.
    exclude_self_edges (default=False): Excludes self-connected edges.
    name: (string, optional) A name for the operation.

    Returns:
    A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

    Raises:
    ValueError: if any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
      `None` in `graph`.
    """

    if exclude_self_edges:
        is_edge_func = lambda sender, receiver: receiver > sender
        # output['n_edge'] = tf.fill([1], n_node * (n_node - 1) // 2)
    else:
        is_edge_func = lambda sender, receiver: receiver >= sender
        # output['n_edge'] = tf.fill([1], n_node * (n_node - 1) // 2 + n_node)

    return connect_graph_dynamic(graph, is_edge_func, name="autoregressive_connect_graph_dynamic")
