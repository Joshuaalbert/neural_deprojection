sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from neural_deprojection.graph_net_utils import AbstractModule
import tensorflow as tf
# import tensorflow_addons as tfa

from graph_nets import blocks
import sonnet as snt
import numpy as np
from graph_nets.graphs import GraphsTuple


class EncoderNetwork(AbstractModule):
    """
    Encoder network that updates the graph to viable input for the DiscreteGraphVAE network.
    """

    def __init__(self,
                 num_output: int,
                 output_size: int,
                 reducer=tf.math.unsorted_segment_mean,
                 properties_size=11,
                 name=None):
        super(EncoderNetwork, self).__init__(name=name)
        self.num_output = num_output
        self.output_size = output_size

        node_enc_size = 64
        edge_enc_size = 64

        node_model_fn = lambda: snt.nets.MLP([node_enc_size], activate_final=True, activation=tf.nn.sigmoid)
        edge_model_fn = lambda: snt.nets.MLP([edge_enc_size], activate_final=True, activation=tf.nn.sigmoid)
        global_model_fn = lambda: snt.nets.MLP([num_output * output_size], activate_final=True,
                                               activation=tf.nn.sigmoid)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=False,
                                           use_sent_edges=False,
                                           use_nodes=False,
                                           use_globals=True)

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
        latent = self.edge_block(graph)
        latent = self.global_block(latent)

        for _ in range(
                self.crossing_steps):  # this would be that theoretical crossing time for information through the graph
            latent = self.node_block(latent)
            latent = self.edge_block(latent)
            latent = self.global_block(latent)

        nodes = tf.reshape(latent.globals, shape=(self.num_output, self.output_size))

        output_graph = GraphsTuple(nodes=nodes,
                                   edges=None,
                                   globals=None,
                                   senders=None,
                                   receivers=None,
                                   n_node=self.num_output,
                                   n_edge=None)  # [n_node, embedding_dim]

        return output_graph


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
