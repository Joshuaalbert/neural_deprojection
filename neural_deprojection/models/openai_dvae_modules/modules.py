import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, get_distribution_strategy, build_log_dir, build_checkpoint_dir, histogramdd, get_shape
import glob, os, json

def same_padding(filter_size):
    return (filter_size - 1)//2

class EncoderResBlock(AbstractModule):
    def __init__(self, out_size, post_gain, name=None):
        super(EncoderResBlock, self).__init__(name=name)
        assert out_size % 4 == 0
        self.out_size = out_size
        hidden_size = out_size // 4
        self.id_path = snt.Conv2D(self.out_size, 1, name='id_path')
        self.conv_block = snt.Sequential([tf.nn.relu, snt.Conv2D(hidden_size, 3, padding=same_padding(3),name='conv_1'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3,  padding=same_padding(3),name='conv_2'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3, padding=same_padding(3), name='conv_3'),
                                          tf.nn.relu, snt.Conv2D(out_size, 1,  padding=same_padding(1),name='conv_4')])

        self.post_gain = post_gain

    def _build(self, img, **kwargs):
        # self.initialise(img)
        return self.id_path(img) + self.post_gain * self.conv_block(img)


class Encoder(AbstractModule):
    def __init__(self, hidden_size, num_embeddings, name=None):
        super(Encoder, self).__init__(name=name)
        num_groups = 4
        num_blk_per_group = 1
        num_layers = num_groups * num_blk_per_group
        post_gain = 1. / num_layers ** 2
        def _single_group(group_idx):
            blk_hidden_size = 2**group_idx * hidden_size
            res_blocks = [EncoderResBlock(blk_hidden_size, post_gain, name=f'blk_{res_blk}')
                          for res_blk in range(num_blk_per_group)]
            if group_idx < num_groups - 1:
                res_blocks.append(lambda x: tf.nn.max_pool2d(x, 2, strides=2, padding='SAME'))
            return snt.Sequential(res_blocks, name=f'group_{group_idx}')

        groups = [snt.Conv2D(hidden_size, 7,  padding=same_padding(7), name='input_group')]
        for groud_idx in range(num_groups):
            groups.append(_single_group(groud_idx))
        groups.append(snt.Sequential([tf.nn.relu, snt.Conv2D(num_embeddings, 1,  padding=same_padding(1),name='logits_conv')], name='output_group'))

        self.blocks = snt.Sequential(groups, name='groups')

    def _build(self, img, **kwargs):
        return self.blocks(img)

class DecoderResBlock(AbstractModule):
    def __init__(self, out_size, post_gain, name=None):
        super(DecoderResBlock, self).__init__(name=name)
        assert out_size % 4 == 0
        self.out_size = out_size
        hidden_size = out_size // 4
        self.id_path = snt.Conv2D(self.out_size, 1, name='id_path')
        self.conv_block = snt.Sequential([tf.nn.relu, snt.Conv2D(hidden_size, 1,  padding=same_padding(1),name='conv_1'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3,  padding=same_padding(3),name='conv_2'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3,  padding=same_padding(3),name='conv_3'),
                                          tf.nn.relu, snt.Conv2D(out_size, 3, padding=same_padding(3), name='conv_4')])

        self.post_gain = post_gain

    def _build(self, img, **kwargs):
        # self.initialise(img)
        return self.id_path(img) + self.post_gain * self.conv_block(img)

def upsample(x):
    """
    Doubles resolution.

    Args:
        x: [batch, W, H, C]

    Returns:
        [batch, 2*W, 2*H, C]
    """
    # shape = x.shape[1:3]
    # return tf.image.resize(x,[shape[0]*2, shape[1]*2], method = 'nearest')
    return tf.repeat(tf.repeat(x,2,axis=1),2,axis=2)


class Decoder(AbstractModule):
    def __init__(self, hidden_size, num_channels=1, name=None):
        super(Decoder, self).__init__(name=name)
        num_groups = 4
        num_blk_per_group = 1
        num_layers = num_groups * num_blk_per_group
        post_gain = 1. / num_layers ** 2

        def _single_group(group_idx):
            blk_hidden_size = 2**(num_groups - group_idx-1) * hidden_size
            res_blocks = [DecoderResBlock(blk_hidden_size, post_gain, name=f'blk_{res_blk}')
                          for res_blk in range(num_blk_per_group)]
            if group_idx < num_groups - 1:
                res_blocks.append(upsample)
            return snt.Sequential(res_blocks, name=f'group_{group_idx}')

        groups = [snt.Conv2D(hidden_size//2, 1,  padding=same_padding(1),name='input_group')]
        for groud_idx in range(num_groups):
            groups.append(_single_group(groud_idx))
        groups.append(
            snt.Sequential([tf.nn.relu, snt.Conv2D(num_channels*2, 1,  padding=same_padding(1),name='likelihood_params_conv')], name='output_group'))

        self.blocks = snt.Sequential(groups, name='groups')

    def _build(self, img, **kwargs):
        return self.blocks(img)