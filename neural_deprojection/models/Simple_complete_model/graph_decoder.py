from graph_nets import blocks
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, graph_batch_reshape, graph_unbatch_reshape
import tensorflow_probability as tfp


def _create_autogressive_edges_from_nodes_dynamic(n_node, exclude_self_edges):
    """Creates complete edges for a graph with `n_node`.

    Args:
    n_node: (integer scalar `Tensor`) The number of nodes.
    exclude_self_edges: (bool) Excludes self-connected edges.

    Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
    """

    rng = tf.range(n_node)

    if exclude_self_edges:
        ind = rng[:, None] > rng
        n_edge = n_node * (n_node - 1) // 2
    else:
        ind = rng[:, None] >= rng
        n_edge = n_node * (n_node - 1) // 2 + n_node

    indicies = tf.where(ind)
    receivers = indicies[:, 0]
    senders = indicies[:, 1]

    receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
    senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
    n_edge = tf.reshape(n_edge, [1])

    return {'receivers': receivers, 'senders': senders, 'n_edge': n_edge}


def test_autoregressive_connect_graph_dynamic():
    graphs = GraphsTuple(nodes=tf.range(20), n_node=tf.constant([12, 8]), n_edge=tf.constant([0, 0]),
                         edges=None, receivers=None, senders=None, globals=None)
    graphs = autoregressive_connect_graph_dynamic(graphs, exclude_self_edges=False)
    import networkx as nx
    G = nx.MultiDiGraph()
    for sender, receiver in zip(graphs.senders.numpy(), graphs.receivers.numpy()):
        G.add_edge(sender, receiver)
    nx.drawing.draw_circular(G)
    import pylab as plt
    plt.show()


def autoregressive_connect_graph_dynamic(graph,
                                         exclude_self_edges=False,
                                         name="autoregressive_connect_graph_dynamic"):
    """Adds edges to a graph by fully-connecting the nodes.

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
    utils_tf._validate_edge_fields_are_all_none(graph)

    with tf.name_scope(name):
        def body(i, senders, receivers, n_edge):
            edges = _create_autogressive_edges_from_nodes_dynamic(graph.n_node[i],
                                                                  exclude_self_edges)
            return (i + 1, senders.write(i, edges['senders']),
                    receivers.write(i, edges['receivers']),
                    n_edge.write(i, edges['n_edge']))

        num_graphs = utils_tf.get_num_graphs(graph)
        loop_condition = lambda i, *_: tf.less(i, num_graphs)
        initial_loop_vars = [0] + [
            tf.TensorArray(dtype=tf.int32, size=num_graphs, infer_shape=False)
            for _ in range(3)  # senders, receivers, n_edge
        ]
        _, senders_array, receivers_array, n_edge_array = tf.while_loop(
            loop_condition, body, initial_loop_vars, back_prop=False)

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

        return graph.replace(senders=senders, receivers=receivers, n_edge=n_edge)


### Use this one###
class GraphMappingNetwork(AbstractModule):
    """
    This autoregressive models the creation of a set of nodes based on some pre-existing set of tokens.
    This is done by adding a set of nodes to the set of tokens for each graph, and incrementally populating them.
    To do this, an artificial ordering must be imposed on the nodes.

    Args:
        num_output: number of new nodes to map to for each graph.
        node_size: size of the nodes during computation. Input nodes will be projected to this size.
        edge_size: size of edges during computation.
        global_size: size of global during computation.

    """

    def __init__(self,
                 num_output: int,
                 num_embedding: int,
                 embedding_size: int = 4,
                 edge_size: int = 4,
                 global_size: int = 10,
                 name=None):
        super(GraphMappingNetwork, self).__init__(name=name)
        self.num_output = num_output
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding

        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_size)))

        # starting value of nodes to be computed.
        self.starting_node_variable = tf.Variable(initial_value=tf.random.truncated_normal((num_output, embedding_size)),
                                                  name='starting_token_node')

        # starting value of globals
        self.starting_global_variable = tf.Variable(initial_value=tf.random.truncated_normal((global_size,)),
                                                    name='starting_global_var')

        self.projection_node_block = blocks.NodeBlock(lambda: snt.Linear(embedding_size, name='project'),
                                                      use_received_edges=False,
                                                      use_sent_edges=False,
                                                      use_nodes=True,
                                                      use_globals=False)

        node_model_fn = lambda: snt.nets.MLP([num_embedding, num_embedding], activate_final=True, activation=tf.nn.leaky_relu)
        edge_model_fn = lambda: snt.nets.MLP([edge_size, edge_size], activate_final=True, activation=tf.nn.leaky_relu)

        self.edge_block = blocks.EdgeBlock(edge_model_fn,
                                           use_edges=False,
                                           use_receiver_nodes=True,
                                           use_sender_nodes=True,
                                           use_globals=True)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=True,
                                           use_sent_edges=True,
                                           use_nodes=True,
                                           use_globals=True)

        self.global_block = blocks.GlobalBlock(global_model_fn,
                                               use_edges=True,
                                               use_nodes=True,
                                               use_globals=True,
                                               edges_reducer=reducer)

        self.output_projection_node_block = blocks.NodeBlock(lambda: snt.Linear(self.output_size, name='project'),
                                                             use_received_edges=False,
                                                             use_sent_edges=False,
                                                             use_nodes=True,
                                                             use_globals=False)

    def _build(self, graphs):
        """
        Adds another set of nodes to each graph. Autoregressively links all nodes in a graph.

        Args:
            graphs: batched GraphsTuple

        Returns:

        """
        # give graphs edges and new node dimension (linear transformation)
        graphs = self.projection_node_block(graphs)  # [n_node, embedding_size]
        batched_graphs = graph_batch_reshape(graphs)  # nodes = [n_graphs, n_node_per_graph, dim]
        [n_graphs, n_node_per_graph_before_concat, _] = get_shape(batched_graphs.nodes)
        # n_graphs, n_node+num_output, embedding_size
        concat_nodes = tf.concat(
            [batched_graphs.nodes, tf.tile(self.starting_node_variable[None, :], [n_graphs, 1, 1])], axis=0)
        batched_graphs = batched_graphs.replace(nodes=concat_nodes,
                                                globals=tf.tile(self.starting_global_variable[None, :], [n_graphs, 1]),
                                                n_node=tf.fill([n_graphs],
                                                               n_node_per_graph_before_concat + self.num_output))
        concat_graphs = graph_unbatch_reshape(batched_graphs)  # [n_node+num_output, embedding_size]

        # nodes, senders, recievers, globals
        concat_graphs = self.autoregressive_connect_graph_dynamic(
            concat_graphs)  # exclude self edges because 3d tokens orginally placeholder?

        latent_graphs = concat_graphs

        temperature = tf.maximum(0.1, tf.cast(10. - 0.1 / (self.step / 1000), tf.float32))

        def _core(output_token_idx, latent_graphs, prev_kl_term):
            batched_latent_graphs = graph_batch_reshape(latent_graphs)
            batched_input_nodes = batched_latent_graphs.nodes  # [n_graphs, n_node+num_output, embedding_size]
            latent_graphs = self.edge_block(latent_graphs)
            latent_graphs = self.node_block(latent_graphs)
            batched_latent_graphs = graph_batch_reshape(latent_graphs)
            token_3d_logits = batched_latent_graphs.nodes[:, -self.num_output,
                              :]  # n_graphs, num_output, num_embedding
            token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=token_3d_logits)
            token_3d_samples_onehot = token_distribution.sample((1,),
                                                                name='token_samples')  # [1, n_graphs, num_output, num_embedding]
            token_3d_samples_onehot = token_3d_samples_onehot[0]  # [n_graphs, num_output, num_embedding]
            token_3d_samples = tf.einsum('goe,ed->god', token_3d_samples_onehot,
                                         self.embeddings)  # n_graphs,num_ouput, embedding_dim
            _mask = tf.range(self.num_output) == output_token_idx  # num_output
            mask = tf.concat([tf.zeros(n_node_per_graph_before_concat, dtype=tf.bool), _mask])  # n_node+num_output
            mask = tf.tile(mask, [n_graphs])  #

            kl_term = tf.reduce_sum(token_3d_samples_onehot * token_3d_logits, axis=-1)  # [n_graphs, num_output]
            kl_term = tf.reduce_sum(tf.cast(_mask, tf.float32) * kl_term, axis=-1)  # [n_graphs]
            kl_term += prev_kl_term

            # n_graphs, n_node+num_output, embedding_size
            output_nodes = tf.where(mask[None, :, None],
                                    tf.concat([tf.zeros([n_graphs, n_node_per_graph_before_concat, self.embedding_size]),
                                               token_3d_samples], axis=1),
                                    batched_input_nodes)
            batched_latent_graphs = batched_latent_graphs.replace(nodes=output_nodes)
            latent_graphs = graph_unbatch_reshape(batched_latent_graphs)

            return (output_token_idx + 1, latent_graphs, kl_term)

        _, latent_graphs, kl_div = tf.while_loop(
            cond=lambda output_token_idx, state, _: output_token_idx < self.num_output,
            body=lambda output_token_idx, state, prev_kl_term: _core(output_token_idx + 1, state, prev_kl_term),
            loop_vars=(tf.constant(0), latent_graphs, tf.zeros((n_graphs,), dtype=tf.float32)))

        latent_graphs = graph_batch_reshape(latent_graphs)
        token_nodes = latent_graphs.nodes[:, -self.num_output:, :]
        # todo:scalar(temp)
        return token_nodes, kl_div