"""
Input Graph:
    nodes: (positions, properties)
    senders/receivers: K-nearest neighbours
    edges: None
    globals: None

Latent Graph:
    nodes: Tokens
    senders/receivers: None
    edges: None
    globals: None

Output Graph:
    nodes: gaussian representation of 3d structure
    senders/receivers: None
    edges: None
    globals: None
"""
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic
from graph_nets.blocks import NodeBlock, EdgeBlock, GlobalBlock, ReceivedEdgesToNodesAggregator
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir, \
    gaussian_loss_function, reconstruct_fields_from_gaussians
from neural_deprojection.models.identify_medium.generate_data import decode_examples
import glob, os, json
import tensorflow_probability as tfp


class DiscreteGraphVAE(AbstractModule):
    def __init__(self, encoder_fn: AbstractModule, decode_fn: AbstractModule, embedding_dim: int = 64,
                 num_embedding: int = 1024, num_gaussian_components:int=128,
                 num_token_samples: int = 1, num_properties: int = None, encoder_kwargs: dict = None,
                 decode_kwargs: dict = None, name=None):
        super(DiscreteGraphVAE, self).__init__(name=name)
        # (num_embedding, embedding_dim)
        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_dim)),
                                      name='embeddings')
        self.encoder = encoder_fn(num_output=num_embedding, output_size=embedding_dim,
                                  **encoder_kwargs)
        self.decoder = decode_fn(num_gaussian_components=num_gaussian_components, component_dim=num_properties*10,
                                 **decode_kwargs)
        self.num_token_samples = num_token_samples
        self.num_properties = num_properties
        self.num_embedding = num_embedding

    @tf.function(input_signature=tf.TensorSpec(shape=[None], dtype=tf.float32))  # what is the shape ???
    def sample_encoder(self, graph):
        return self.encoder(graph)

    @tf.function(input_signature=[tf.TensorSpec([None,3], dtype=tf.float32),
                                  tf.TensorSpec([None,None], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32)])
    def sample_decoder(self, positions, logits, temperature):
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((1,),
                                                         name='token_samples')
        token_sample_onehot = token_samples_onehot[0]#[n_node, num_embedding]
        token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [n_node, embedding_dim]
        n_node = tf.shape(token_sample)[0]
        latent_graph = GraphsTuple(nodes=token_sample,
                                   edges=None,
                                   globals=None,
                                   senders=None,
                                   receivers=None,
                                   n_node=[n_node],
                                   n_edge=None)  # [n_node, embedding_dim]
        latent_graph = fully_connect_graph_dynamic(latent_graph)
        gaussian_tokens = self.decoder(latent_graph)  # nodes=[num_gaussian_components, component_dim]
        reconstructed_fields = reconstruct_fields_from_gaussians(gaussian_tokens, positions)
        return reconstructed_fields

    def _build(self, batch, **kwargs) -> dict:
        graph, temperature, beta = batch
        encoded_graph = self.encoder(graph)
        n_node = encoded_graph.n_node
        # nodes = [n_node, num_embeddings]
        # node = [num_embeddings] -> log(p_i) = logits
        # -> [S, n_node, embedding_dim]
        logits = encoded_graph.nodes  # [n_node, num_embeddings]
        log_norm = tf.math.reduce_logsumexp(logits, axis=1)  # [n_node]
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((self.num_token_samples,),
                                                         name='token_samples')  # [S, n_node, num_embeddings]

        def _single_decode(token_sample_onehot):
            """

            Args:
                token_sample: [n_node, embedding_dim]

            Returns:
                log_likelihood: scalar
                kl_term: scalar
            """
            token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [n_node, embedding_dim]  # = z ~ q(z|x)
            latent_graph = GraphsTuple(nodes=token_sample,
                                       edges=None,
                                       globals=None,
                                       senders=None,
                                       receivers=None,
                                       n_node=n_node,
                                       n_edge=None)  # [n_node, embedding_dim]
            latent_graph = fully_connect_graph_dynamic(latent_graph)
            gaussian_tokens = self.decoder(latent_graph)  # nodes=[num_gaussian_components, component_dim]
            _, log_likelihood = gaussian_loss_function(gaussian_tokens, graph)
            # [n_node, num_embeddings].[n_node, num_embeddings]
            sum_selected_logits = tf.math.reduce_sum(token_sample_onehot * logits, axis=1)  # [n_node]
            kl_term = sum_selected_logits - self.num_embedding * log_norm + self.num_embedding * tf.math.log(
                self.num_embedding)  # [n_node]
            kl_term = beta * tf.reduce_mean(kl_term)
            return log_likelihood, kl_term

        log_likelihood_samples, kl_term_samples = tf.vectorized_map(_single_decode, token_samples_onehot)  # [S],[S]

        # good metric = average entropy of embedding usage! The more precisely embeddings are selected the lower the entropy.

        log_prob_tokens = logits - log_norm[:, None]#num_tokens, num_embeddings
        entropy = -tf.reduce_sum(log_prob_tokens * tf.math.exp(log_prob_tokens), axis=1)#num_tokens
        perplexity = 2.**(-entropy/tf.math.log(2.))
        mean_perplexity = tf.reduce_mean(perplexity)

        return dict(loss=tf.reduce_mean(log_likelihood_samples - kl_term_samples),
                    var_exp=tf.reduce_mean(log_likelihood_samples),
                    kl_term=tf.reduce_mean(kl_term_samples),
                    mean_perplexity=mean_perplexity)


MODEL_MAP = dict(dis_graph_vae=DiscreteGraphVAE)


def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None,
                   **kwargs) -> TrainOneEpoch:
    model_cls = MODEL_MAP[model_type]
    model = model_cls(**model_parameters, **kwargs)

    def build_opt(**kwargs):
        opt_type = kwargs.get('opt_type')
        if opt_type == 'adam':
            learning_rate = kwargs.get('learning_rate', 1e-4)
            opt = snt.optimizers.Adam(learning_rate, beta1=1 - 1 / 100, beta2=1 - 1 / 500)
        else:
            raise ValueError('Opt {} invalid'.format(opt_type))
        return opt

    def build_loss(**loss_parameters):
        def loss(model_outputs, batch):
            graph, temperature, beta = batch
            #model_outputs = dict(loss=tf.reduce_mean(log_likelihood_samples - kl_term_samples),
                    # var_exp=tf.reduce_mean(log_likelihood_samples),
                    # kl_term=tf.reduce_mean(kl_term_samples),
                    # mean_perplexity=mean_perplexity)
            return model_outputs['loss']

        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training


def build_dataset(data_dir, batch_size):
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(
        lambda record_bytes: decode_examples(record_bytes, node_shape=[4]))
    dataset = dataset.map(
        lambda graph_data_dict, img, c: GraphsTuple(globals=None, edges=None, **graph_data_dict))
    dataset = dataset.map(
        lambda graph: graph._replace(nodes=tf.concat([graph.nodes[:, :3], tf.math.log(graph.nodes[:, 3:])], axis=1)))
    dataset = dataset.map(lambda graph: (graph, graph._replace(nodes=graph.nodes[:, :3])))
    # dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=batch_size)
    return dataset


def main(data_dir, config):
    # Make strategy at the start of your main before any other tf code is run.
    strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=4)
    test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=4)

    # for (graph, positions) in iter(test_dataset):
    #     print(graph)
    #     break

    with strategy.scope():
        train_one_epoch = build_training(num_processing_steps=4, **config)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=3,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


if __name__ == '__main__':
    config = dict(model_type='dis_graph_vae',
                  model_parameters=dict(message_size=8, latent_size=16, input_size=1),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict())
    main('../identify_medium/data', config)
