import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

from graph_nets import blocks
from graph_nets.utils_tf import concat

import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic
from neural_deprojection.graph_net_utils import AbstractModule, gaussian_loss_function, \
    reconstruct_fields_from_gaussians, histogramdd
import tensorflow_probability as tfp


class DiscreteGraphVAE(AbstractModule):
    def __init__(self, embedding_dim: int = 64,
                 num_embedding: int = 1024, num_gaussian_components: int = 128, num_latent_tokens: int = 64,
                 num_token_samples: int = 1, num_properties: int = 10, encoder_kwargs: dict = None,
                 decode_kwargs: dict = None, name=None):
        super(DiscreteGraphVAE, self).__init__(name=name)
        # (num_embedding, embedding_dim)
        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_dim)),
                                      name='embeddings')
        self.encoder = EncoderNetwork3D(num_output=num_latent_tokens, output_size=num_embedding,
                                        **encoder_kwargs)
        self.decoder = DecoderNetwork3D(num_output=num_gaussian_components,
                                        output_size=num_properties * 10,
                                        **decode_kwargs)
        self.num_token_samples = num_token_samples
        self.num_properties = num_properties
        self.num_embedding = num_embedding
        self.temperature = tf.Variable(initial_value=tf.constant(1.), name='temperature', trainable=False)
        self.beta = tf.Variable(initial_value=tf.constant(6.6), name='beta', trainable=False)

    # @tf.function(input_signature=tf.TensorSpec(shape=[None], dtype=tf.float32))  # what is the shape ???
    # def sample_encoder(self, graph):
    #     return self.encoder(graph)

    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)

    @tf.function(input_signature=[tf.TensorSpec([None, 3], dtype=tf.float32),
                                  tf.TensorSpec([None, None], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32)])
    def sample_decoder(self, positions, logits, temperature):
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((1,),
                                                         name='token_samples')
        token_sample_onehot = token_samples_onehot[0]  # [n_node, num_embedding]
        token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [n_node, embedding_dim]
        n_node = tf.shape(token_sample)[0]
        latent_graph = GraphsTuple(nodes=token_sample,
                                   edges=None,
                                   globals=tf.constant([0.], dtype=tf.float32),
                                   senders=None,
                                   receivers=None,
                                   n_node=tf.constant([n_node], dtype=tf.int32),
                                   n_edge=tf.constant([0], dtype=tf.int32))  # [n_node, embedding_dim]
        latent_graph = fully_connect_graph_dynamic(latent_graph)
        gaussian_tokens = self.decoder(latent_graph)  # nodes=[num_gaussian_components, component_dim]
        reconstructed_fields = reconstruct_fields_from_gaussians(gaussian_tokens, positions)
        return reconstructed_fields

    def _build(self, graph, **kwargs) -> dict:
        encoded_graph = self.encoder(graph)
        # print('\n encoded_graph.nodes', encoded_graph.nodes, '\n')
        # nodes = [n_node, num_embeddings]
        # node = [num_embeddings] -> log(p_i) = logits
        # -> [S, n_node, embedding_dim]
        logits = encoded_graph.nodes  # [n_node, num_embeddings]
        print('\n logits', logits, '\n')
        log_norm = tf.math.reduce_logsumexp(logits, axis=1)  # [n_node]
        # print('log_norm', log_norm)
        self.set_temperature(40 * tf.math.exp(-0.01 * tf.cast(self.step, tf.float32)))
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((self.num_token_samples,),
                                                         name='token_samples')  # [S, n_node, num_embeddings]
        print('temperature', self.temperature)
        print('token_samples_onehot', token_samples_onehot[0])
        print('max index', tf.argmax(token_samples_onehot[0], axis=1))

        def _single_decode(token_sample_onehot):
            """

            Args:
                token_sample: [n_node, embedding_dim]

            Returns:
                log_likelihood: scalar
                kl_term: scalar
            """
            token_sample = tf.matmul(token_sample_onehot, self.embeddings) # / self.num_embedding  # [n_node, embedding_dim]  # = z ~ q(z|x)
            latent_graph = GraphsTuple(nodes=token_sample,
                                       edges=None,
                                       globals=tf.constant([0.], dtype=tf.float32),
                                       senders=None,
                                       receivers=None,
                                       n_node=encoded_graph.n_node,
                                       n_edge=tf.constant([0], dtype=tf.int32))  # [n_node, embedding_dim]
            latent_graph = fully_connect_graph_dynamic(latent_graph)
            print('\n latent_graph.nodes', latent_graph.nodes, '\n')
            gaussian_tokens = self.decoder(latent_graph)  # nodes=[num_gaussian_components, component_dim]
            print('\n gaussian_tokens_nodes', gaussian_tokens.nodes, '\n')
            field_properties, log_likelihood = gaussian_loss_function(gaussian_tokens.nodes, graph)
            # [n_node, num_embeddings].[n_node, num_embeddings]
            sum_selected_logits = tf.math.reduce_sum(token_sample_onehot * logits, axis=1)  # [n_node]
            print('sum', sum_selected_logits)
            print('norm', log_norm)
            print('num_embed', tf.cast(self.num_embedding, tf.float32))
            print('embed', tf.math.log(tf.cast(self.num_embedding, tf.float32)))
            kl_term = sum_selected_logits - self.num_embedding * log_norm + \
                      self.num_embedding * tf.math.log(tf.cast(self.num_embedding, tf.float32))
            print('kl_term 0', kl_term)
            print('kl_term', tf.reduce_mean(kl_term))
            kl_term = self.beta * tf.reduce_mean(kl_term)
            return log_likelihood, kl_term, field_properties

        log_likelihood_samples, kl_term_samples, properties = _single_decode(
            token_samples_onehot[0])  # tf.vectorized_map(_single_decode, token_samples_onehot)  # [S],[S]

        # good metric = average entropy of embedding usage! The more precisely embeddings are selected the lower the entropy.

        log_prob_tokens = logits - log_norm[:, None]  # num_tokens, num_embeddings
        entropy = -tf.reduce_sum(log_prob_tokens * tf.math.exp(log_prob_tokens), axis=1)  # num_tokens
        perplexity = 2. ** (-entropy / tf.math.log(2.))
        mean_perplexity = tf.reduce_mean(perplexity)

        var_exp = tf.reduce_mean(log_likelihood_samples)
        kl_term = tf.reduce_mean(kl_term_samples)

        elbo_samples = log_likelihood_samples - kl_term_samples
        elbo = tf.reduce_mean(elbo_samples)
        loss = - elbo  # maximize ELBO so minimize -ELBO

        print('properties', properties)

        if self.step % 100 == 0:
            for i in range(7):
                image_before, _ = histogramdd(graph.nodes[:, :2], bins=50, weights=graph.nodes[:, 3+i])
                image_before -= tf.reduce_min(image_before)
                image_before /= tf.reduce_max(image_before)
                tf.summary.image(f"{3+i}_xy_image_before", image_before[None, :, :, None], step=self.step)
                tf.summary.scalar(f"properties{3+i}_std_before", tf.math.reduce_std(graph.nodes[:, 3+i]), step=self.step)

                image_after, _ = histogramdd(graph.nodes[:, :2], bins=50, weights=properties[:, i])
                image_after -= tf.reduce_min(image_after)
                image_after /= tf.reduce_max(image_after)
                tf.summary.image(f"{3+i}_xy_image_after", image_after[None, :, :, None], step=self.step)
                tf.summary.scalar(f"properties{3+i}_std_after", tf.math.reduce_std(properties[:, i]), step=self.step)

        if self.step % 10 == 0:
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', var_exp, step=self.step)
            tf.summary.scalar('kl_term', kl_term, step=self.step)

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_term=kl_term,
                                 mean_perplexity=mean_perplexity))


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
                                           use_edges=True,
                                           use_receiver_nodes=True,
                                           use_sender_nodes=True,
                                           use_globals=True)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=True,
                                           use_sent_edges=True,
                                           use_nodes=True,
                                           use_globals=True,
                                           received_edges_reducer=reducer,
                                           sent_edges_reducer=reducer)

        self.global_block = blocks.GlobalBlock(global_model_fn,
                                               use_edges=True,
                                               use_nodes=True,
                                               use_globals=True,
                                               edges_reducer=reducer,
                                               nodes_reducer=reducer)

        self.output_projection_node_block = blocks.NodeBlock(lambda: snt.Linear(self.output_size, name='project'),
                                                             use_received_edges=False,
                                                             use_sent_edges=False,
                                                             use_nodes=True,
                                                             use_globals=False)

    def _build(self, graph):
        # give graph edges and new node dimension (linear transformation)
        graph = graph.replace(edges=tf.tile(self.intra_graph_edge_variable[None, :], [graph.n_edge[0], 1]))
        graph = self.projection_node_block(graph)  # [n_nodes, node_size]
        # print('graph 1', graph)
        n_node = tf.shape(graph.nodes)[0]
        graph.replace(n_node=n_node)
        # create fully connected output token nodes
        token_start_nodes = tf.tile(self.empty_node_variable[None, :], [self.num_output, 1])
        token_graph = GraphsTuple(nodes=token_start_nodes,
                                  edges=None,
                                  globals=tf.constant([0.], dtype=tf.float32),
                                  senders=None,
                                  receivers=None,
                                  n_node=tf.constant([self.num_output], dtype=tf.int32),
                                  n_edge=tf.constant([0], dtype=tf.int32))
        token_graph = fully_connect_graph_dynamic(token_graph)
        # print('\n token graph', token_graph, '\n')
        token_graph = token_graph.replace(
            edges=tf.tile(self.intra_token_graph_edge_variable[None, :], [token_graph.n_edge[0], 1]))
        concat_graph = concat([graph, token_graph], axis=0)  # n_node = [n_nodes, n_tokes]
        concat_graph = concat_graph.replace(n_node=tf.reduce_sum(concat_graph.n_node, keepdims=True),
                                            n_edge=tf.reduce_sum(concat_graph.n_edge,
                                                                 keepdims=True))  # n_node=[n_nodes+n_tokens]

        # add random edges between
        # choose random unique set of nodes in graph, choose random set of nodes in token_graph
        gumbel = -tf.math.log(-tf.math.log(tf.random.uniform((n_node,))))
        n_connect_edges = tf.cast(
            tf.multiply(tf.constant([self.inter_graph_connect_prob]), tf.cast(n_node, tf.float32)), tf.int32)
        _, graph_senders = tf.nn.top_k(gumbel, n_connect_edges[0])
        # print('graph_senders', graph_senders)
        token_graph_receivers = n_node + tf.random.uniform(shape=n_connect_edges, minval=0, maxval=self.num_output,
                                                           dtype=tf.int32)
        # print('token_graph_receivers', token_graph_receivers)
        senders = tf.concat([concat_graph.senders, graph_senders, token_graph_receivers],
                            axis=0)  # add bi-directional senders + receivers
        receivers = tf.concat([concat_graph.receivers, token_graph_receivers, graph_senders], axis=0)
        inter_edges = tf.tile(self.inter_graph_edge_variable[None, :],
                              tf.concat([2 * n_connect_edges, tf.constant([1], dtype=tf.int32)],
                                        axis=0))  # 200 = 10000(n_nodes) * 0.01 * 2
        edges = tf.concat([concat_graph.edges, inter_edges], axis=0)
        concat_graph = concat_graph.replace(senders=senders, receivers=receivers, edges=edges,
                                            n_edge=concat_graph.n_edge[0] + 2 * n_connect_edges[0],
                                            # concat_graph.n_edge[0] + 2 * n_connect_edges
                                            globals=self.starting_global_variable[None, :])
        # print('starting global', self.starting_global_variable[None, :])
        latent_graph = concat_graph

        print('concat_graph_nodes', self.name, concat_graph.nodes)
        for step in range(self.crossing_steps):  # this would be that theoretical crossing time for information through the graph
            input_nodes = latent_graph.nodes
            latent_graph = self.edge_block(latent_graph)
            latent_graph = self.node_block(latent_graph)
            latent_graph = self.global_block(latent_graph)
            latent_graph = latent_graph.replace(nodes=latent_graph.nodes + input_nodes)  # residual connections
            if step % 3 == 0:
                print('latent_graph_nodes', self.name, latent_graph.nodes)
                print('latent_graph_edges', self.name, latent_graph.edges)
                print('latent_graph_globals', self.name, latent_graph.globals)

        latent_graph = latent_graph.replace(nodes=latent_graph.nodes[n_node:],
                                            edges=None,
                                            receivers=None,
                                            senders=None,
                                            globals=None,
                                            n_node=tf.constant([self.num_output], dtype=tf.int32),
                                            n_edge=tf.constant(0, dtype=tf.int32))
        output_graph = self.output_projection_node_block(latent_graph)
        print('output_graph_nodes', self.name, output_graph.nodes)

        return output_graph


class EncoderNetwork3D(GraphMappingNetwork):
    def __init__(self, num_output: int,
                 output_size: int,
                 inter_graph_connect_prob: float = 0.01,
                 reducer=tf.math.unsorted_segment_mean,
                 starting_global_size=4,
                 node_size=64,
                 edge_size=4,
                 crossing_steps=4,
                 name=None):
        super(EncoderNetwork3D, self).__init__(num_output=num_output,
                                               output_size=output_size,
                                               inter_graph_connect_prob=inter_graph_connect_prob,
                                               reducer=reducer,
                                               starting_global_size=starting_global_size,
                                               node_size=node_size,
                                               edge_size=edge_size,
                                               crossing_steps=crossing_steps,
                                               name=name)


class DecoderNetwork3D(GraphMappingNetwork):
    def __init__(self, num_output: int,
                 output_size: int,
                 inter_graph_connect_prob: float = 0.01,
                 reducer=tf.math.unsorted_segment_mean,
                 starting_global_size=4,
                 node_size=64,
                 edge_size=4,
                 crossing_steps=4,
                 name=None):
        super(DecoderNetwork3D, self).__init__(num_output=num_output,
                                               output_size=output_size,
                                               inter_graph_connect_prob=inter_graph_connect_prob,
                                               reducer=reducer,
                                               starting_global_size=starting_global_size,
                                               node_size=node_size,
                                               edge_size=edge_size,
                                               crossing_steps=crossing_steps,
                                               name=name)

