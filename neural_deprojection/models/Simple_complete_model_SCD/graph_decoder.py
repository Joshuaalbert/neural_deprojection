from graph_nets import blocks
from graph_nets.utils_tf import concat

import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf
from neural_deprojection.graph_net_utils import AbstractModule, gaussian_loss_function, \
    get_shape, graph_batch_reshape, graph_unbatch_reshape, replicate_graph, sort_graph
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

    receivers, senders = tf.where(ind)


    receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
    senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
    n_edge = tf.reshape(n_edge, [1])

    return {'receivers': receivers, 'senders': senders, 'n_edge': n_edge}


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

        return graph._replace(senders=senders, receivers=receivers, n_edge=n_edge)


class GraphMappingNetwork(AbstractModule):
    """
    Encoder network that updates the graph to viable input for the DiscreteGraphVAE network.
    """

    def __init__(self,
                 num_output: int,
                 output_size: int,
                 node_size: int = 4,
                 edge_size: int = 4,
                 starting_global_size: int = 10,
                 inter_graph_connect_prob: float = 0.01,
                 crossing_steps: int = 4,
                 reducer=tf.math.unsorted_segment_mean,
                 properties_size=10,
                 name=None):
        super(GraphMappingNetwork, self).__init__(name=name)
        self.num_output = num_output
        self.output_size = output_size
        self.crossing_steps = crossing_steps
        self.empty_node_variable = tf.Variable(initial_value=tf.random.truncated_normal((node_size,)),
                                               name='empty_token_node')

        # values for different kinds of edges in the graph, which will be learned
        self.intra_graph_edge_variable = tf.Variable(initial_value=tf.random.truncated_normal((edge_size,)),
                                                     name='intra_graph_edge_var')
        self.intra_token_graph_edge_variable = tf.Variable(initial_value=tf.random.truncated_normal((edge_size,)),
                                                           name='intra_token_graph_edge_var')
        self.inter_graph_edge_variable = tf.Variable(initial_value=tf.random.truncated_normal((edge_size,)),
                                                     name='inter_graph_edge_var')
        self.starting_global_variable = tf.Variable(initial_value=tf.random.truncated_normal((starting_global_size,)),
                                                    name='starting_global_var')

        self.inter_graph_connect_prob = inter_graph_connect_prob

        self.projection_node_block = blocks.NodeBlock(lambda: snt.Linear(node_size, name='project'),
                                                      use_received_edges=False,
                                                      use_sent_edges=False,
                                                      use_nodes=True,
                                                      use_globals=False)

        node_model_fn = lambda: snt.nets.MLP([node_size, node_size], activate_final=True, activation=tf.nn.leaky_relu)
        edge_model_fn = lambda: snt.nets.MLP([edge_size, edge_size], activate_final=True, activation=tf.nn.leaky_relu)
        global_model_fn = lambda: snt.nets.MLP([starting_global_size, starting_global_size], activate_final=True,
                                               activation=tf.nn.leaky_relu)

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
        graphs = self.projection_node_block(graphs)  # [n_node, node_size]
        batched_graphs = graph_batch_reshape(graphs) #nodes = [n_graphs, n_node_per_graph, dim]
        [n_graphs, n_node_per_graph, _] = get_shape(batched_graphs.nodes)
        concat_nodes = tf.concat([batched_graphs.nodes, tf.tile(self.empty_node_variable[None, :], [n_graphs, self.num_output, 1])], axis=0)
        batched_graphs = batched_graphs.replace(nodes=concat_nodes,
                                                globals=tf.tile(self.global_block[None,:],[n_graphs]))
        concat_graphs = graph_unbatch_reshape(batched_graphs) #[n_node+num_output, node_size]

        #nodes, senders, recievers, globals
        concat_graphs = self.autoregressive_connect_graph_dynamic(concat_graphs)    # exclude self edges because 3d tokens orginally placeholder?

        latent_graphs = concat_graphs
        node_idx = -self.num_output

        def _core(latent_graphs):
            input_nodes = latent_graphs.nodes
            latent_graphs = self.edge_block(latent_graphs)
            latent_graphs = self.node_block(latent_graphs)
            latent_graphs = self.global_block(latent_graphs)
            masked_nodes = tf.concat([latent_graphs.nodes[:-self.num_output + node_idx], tf.tile(self.masked_node[None, :], [n_graphs, self.num_output - node_idx, 1])], axis=0)
            latent_graphs = latent_graphs.replace(masked_nodes)
            #todo: mask
            latent_graphs = latent_graphs.replace(nodes=latent_graphs.nodes + input_nodes)  # residual connections   # why residual when 3d tokens are placeholder orginally?

            node_idx += 1

            return latent_graphs

        node_idx = 0
        def _alt_core(latent_graphs, node_idx):
            input_nodes = latent_graphs.nodes
            latent_graphs = self.node_block(latent_graphs)    # node block uses only sender nodes

            new_nodes = latent_graphs.nodes
            new_nodes[-self.num_output:-self.num_output+node_idx] = input_nodes[-self.num_output:-self.num_output+node_idx]
            latent_graphs.replace(nodes=new_nodes)
            node_idx += 1

            return latent_graphs, node_idx

        _, latent_graphs = tf.while_loop(cond=lambda const, state: const < self.num_output,
                                        body=lambda const, state: (const + 1, _core(state)),
                                        loop_vars=(tf.constant(0), latent_graphs))

        latent_graphs = graph_batch_reshape(latent_graphs)
        token_nodes = latent_graphs.nodes[:,-self.num_output:,:]

        # latent_graphs = latent_graphs.replace(nodes=token_nodes,
        #                                     edges=None,
        #                                     receivers=None,
        #                                     senders=None,
        #                                     globals=None,
        #                                     n_node=self.num_output*tf.ones(n_graphs, dtype=tf.int32),
        #                                     n_edge=tf.constant(0, dtype=tf.int32))
        # latent_graphs = graph_unbatch_reshape(latent_graphs)
        # output_graphs = self.output_projection_node_block(latent_graphs)

        return token_nodes