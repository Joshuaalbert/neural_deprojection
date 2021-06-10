import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from neural_deprojection.graph_net_utils import AbstractModule
import tensorflow as tf
# import tensorflow_addons as tfa

from graph_nets import blocks
import sonnet as snt
import numpy as np
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_static, concat


class GraphMappingNetwork(AbstractModule):
    """
    Encoder network that updates the graph to viable input for the DiscreteGraphVAE network.
    """

    def __init__(self,
                 num_output: int,
                 output_size: int,
                 node_size: int,
                 edge_size: int,
                 starting_global_size: int,
                 inter_graph_connect_prob: float = 0.01,
                 reducer=tf.math.unsorted_segment_mean,
                 properties_size=11,
                 name=None):
        super(GraphMappingNetwork, self).__init__(name=name)
        self.num_output = num_output
        self.output_size = output_size
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
                                                      use_nodes=False,
                                                      use_globals=False)

        node_model_fn = lambda: snt.nets.MLP([node_size, node_size], activate_final=True, activation=tf.nn.leaky_relu)
        edge_model_fn = lambda: snt.nets.MLP([edge_size, edge_size], activate_final=True, activation=tf.nn.leaky_relu)
        global_model_fn = lambda: snt.nets.MLP([starting_global_size, starting_global_size], activate_final=True,
                                               activation=tf.nn.leaky_relu)

        self.edge_block = blocks.EdgeBlock(edge_model_fn,
                                           use_edges=True,
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
                                                             use_nodes=False,
                                                             use_globals=False)

    def _build(self, graph, crossing_steps):
        n_edge = graph.n_edge[0]
        graph = graph.replace(edges=tf.tile(self.intra_graph_edge_variable[None, :], [n_edge, 1]))
        graph = self.projection_node_block(graph)  # [n_nodes, node_size]
        n_node = tf.shape(graph.nodes)[0]
        # create fully connected output token nodes
        token_start_nodes = tf.tile(self.empty_node_variable[None, :], [self.num_output, 1])
        token_graph = GraphsTuple(nodes=token_start_nodes,
                                  edges=None,
                                  globals=None,
                                  senders=None,
                                  receivers=None,
                                  n_node=[self.num_output],
                                  n_edge=None)
        token_graph = fully_connect_graph_static(token_graph)
        n_edge = token_graph.n_edge[0]
        token_graph = token_graph.replace(edges=tf.tile(self.intra_token_graph_edge_variable[None, :], [n_edge, 1]))

        concat_graph = concat([graph, token_graph], axis=0)  # n_node = [n_nodes, n_tokes]
        concat_graph = concat_graph.replace(
            n_node=tf.reduce_sum(concat_graph, keepdims=True))  # n_node=[n_nodes+n_tokens]
        # add random edges between
        # choose random unique set of nodes in graph, choose random set of nodes in token_graph
        gumbel = -tf.log(-tf.log(tf.random_uniform((n_node,))))
        n_connect_edges = tf.cast(self.inter_graph_connect_prob * n_node, tf.int32)
        _, graph_senders = tf.nn.top_k(gumbel, n_connect_edges)
        token_graph_receivers = n_node + tf.random.uniform((n_connect_edges,), minval=0, maxval=self.num_output,
                                                           dtype=tf.int32)
        senders = tf.concat([concat_graph.senders, graph_senders, token_graph_receivers],
                            axis=0)  # add bi-directional senders + receivers
        receivers = tf.concat([concat_graph.receivers, token_graph_receivers, graph_senders], axis=0)
        inter_edges = tf.tile(self.inter_graph_edge_variable[None, :], [2 * n_connect_edges, 1])
        edges = tf.concat([concat_graph.edges, inter_edges], axis=0)
        concat_graph = concat_graph.replace(senders=senders, receivers=receivers, edges=edges,
                                            n_edge=[concat_graph.n_edge[0] + 2 * n_connect_edges],
                                            globals=self.starting_global_variable[None, :])

        latent_graph = concat_graph
        for _ in range(
                self.crossing_steps):  # this would be that theoretical crossing time for information through the graph
            input_nodes = latent_graph.nodes
            latent_graph = self.edge_block(latent_graph)
            latent_graph = self.node_block(latent_graph)
            latent_graph = self.global_block(latent_graph)
            latent_graph = latent_graph.replace(nodes=latent_graph.nodes + input_nodes)  # residual connections

        output_graph = self.output_projection_node_block(latent_graph)

        return output_graph


class EncoderNetwork(GraphMappingNetwork):
    def __init__(self, num_output: int,
                 output_size: int,
                 inter_graph_connect_prob: float = 0.01,
                 reducer=tf.math.unsorted_segment_mean,
                 starting_global_size=16,
                 node_size=64,
                 edge_size=3,
                 name=None):
        super(EncoderNetwork, self).__init__(num_output=num_output,
                                             output_size=output_size,
                                             inter_graph_connect_prob=inter_graph_connect_prob,
                                             reducer=reducer,
                                             starting_global_size=starting_global_size,
                                             node_size=node_size,
                                             edge_size=edge_size,
                                             name=name)

class DecoderNetwork(GraphMappingNetwork):
    def __init__(self, num_output: int,
                 output_size: int,
                 inter_graph_connect_prob: float = 0.01,
                 reducer=tf.math.unsorted_segment_mean,
                 starting_global_size=16,
                 node_size=64,
                 edge_size=3,
                 name=None):
        super(DecoderNetwork, self).__init__(num_output=num_output,
                                             output_size=output_size,
                                             inter_graph_connect_prob=inter_graph_connect_prob,
                                             reducer=reducer,
                                             starting_global_size=starting_global_size,
                                             node_size=node_size,
                                             edge_size=edge_size,
                                             name=name)


class DecoderNetwork(AbstractModule):
    """
    Decoder network that updates the latent graph to gaussian components.
    """

    def __init__(self,
                 num_components,
                 reducer=tf.math.unsorted_segment_mean,
                 properties_size=11,
                 name=None):
        super(DecoderNetwork, self).__init__(name=name)

        self.num_components = num_components
        self.components_dim = properties_size * 10

        edge_enc_size = 32

        node_model_fn = lambda: snt.nets.MLP([self.components_dim], activate_final=True, activation=tf.nn.sigmoid)
        edge_model_fn = lambda: snt.nets.MLP([edge_enc_size], activate_final=True, activation=tf.nn.sigmoid)
        global_model_fn = lambda: snt.nets.MLP([self.components_dim], activate_final=True, activation=tf.nn.sigmoid)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=False,
                                           use_sent_edges=False,
                                           use_nodes=True,
                                           use_globals=False)

        self.edge_block = blocks.EdgeBlock(edge_model_fn,
                                           use_edges=False,
                                           use_receiver_nodes=True,
                                           use_sender_nodes=True,
                                           use_globals=False)

        self.global_block = blocks.GlobalBlock(global_model_fn,
                                               use_edges=True,
                                               use_nodes=False,
                                               use_globals=False,
                                               edges_reducer=reducer)

    def _build(self, graph, crossing_steps):
        n_non_components = graph.n_node
        latent = self.node_block(graph)

        for _ in range(self.num_components):
            latent = self.edge_block(latent)
            latent = self.global_block(latent)

            component = latent.globals
            new_nodes = tf.concat([latent.nodes, component], axis=0)
            n_nodes = latent.n_node
            new_senders = tf.concat([tf.range(n_nodes + 1), tf.fill(dims=(n_nodes + 1), value=n_nodes)])
            new_receivers = tf.reverse(new_senders, axis=0)
            new_edges = tf.fill(dims=((n_nodes + 1) ** 2), value=0.)
            n_edges = (n_nodes + 1) ** 2

            latent = GraphsTuple(nodes=new_nodes,
                                 edges=new_edges,
                                 globals=None,
                                 senders=new_senders,
                                 receivers=new_receivers,
                                 n_node=n_nodes + 1,
                                 n_edge=n_edges)

        gaussian_components = latent.nodes[n_non_components:]

        return gaussian_components
