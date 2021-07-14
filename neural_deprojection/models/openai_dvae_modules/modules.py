import tensorflow as tf
import sonnet as snt
from neural_deprojection.graph_net_utils import AbstractModule


def same_padding(filter_size):
    return tuple([(filter_size - 1) // 2] * 2)


class EncoderResBlock2D(AbstractModule):
    def __init__(self, out_size, post_gain, name=None):
        super(EncoderResBlock2D, self).__init__(name=name)
        assert out_size % 4 == 0
        self.out_size = out_size
        hidden_size = out_size // 4
        self.id_path = snt.Conv2D(self.out_size, 1, name='id_path')
        self.conv_block = snt.Sequential([snt.LayerNorm(-1, True, True, name='layer_norm'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3, padding="SAME", name='conv_1'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3, padding="SAME", name='conv_2'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3, padding="SAME", name='conv_3'),
                                          tf.nn.relu, snt.Conv2D(out_size, 1, padding="SAME", name='conv_4')])
        self.post_gain = post_gain

    def _build(self, img, **kwargs):
        output = self.id_path(img) + self.post_gain * self.conv_block(img)
        return output


class Encoder2D(AbstractModule):
    def __init__(self, hidden_size, num_embeddings, name=None):
        super(Encoder2D, self).__init__(name=name)
        num_groups = 4
        num_blk_per_group = 1
        num_layers = num_groups * num_blk_per_group
        post_gain = 1. / num_layers ** 2

        def _single_group(group_idx):
            blk_hidden_size = 2 ** group_idx * hidden_size
            res_blocks = [EncoderResBlock2D(blk_hidden_size, post_gain, name=f'blk_{res_blk}')
                          for res_blk in range(num_blk_per_group)]
            if group_idx < num_groups - 1:
                res_blocks.append(lambda x: tf.nn.max_pool2d(x, 2, strides=2, padding='SAME'))
            return snt.Sequential(res_blocks, name=f'group_{group_idx}')

        groups = [snt.Conv2D(hidden_size, 7, padding="SAME", name='input_group')]
        for groud_idx in range(num_groups):
            groups.append(_single_group(groud_idx))
        groups.append(
            snt.Sequential([tf.nn.relu, snt.Conv2D(num_embeddings, 1, padding="SAME", name='logits_conv')],
                           name='output_group'))

        self.blocks = snt.Sequential(groups, name='groups')

    def _build(self, img, **kwargs):
        return self.blocks(img)


class DecoderResBlock2D(AbstractModule):
    def __init__(self, out_size, post_gain, name=None):
        super(DecoderResBlock2D, self).__init__(name=name)
        assert out_size % 4 == 0
        self.out_size = out_size
        hidden_size = out_size // 4
        self.id_path = snt.Conv2D(self.out_size, 1, name='id_path')
        self.conv_block = snt.Sequential([snt.LayerNorm(-1,True, True, name='layer_norm'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 1, padding="SAME", name='conv_1'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3, padding="SAME", name='conv_2'),
                                          tf.nn.relu, snt.Conv2D(hidden_size, 3, padding="SAME", name='conv_3'),
                                          tf.nn.relu, snt.Conv2D(out_size, 3, padding="SAME", name='conv_4')])
        self.post_gain = post_gain

    def _build(self, img, **kwargs):
        output = self.id_path(img) + self.post_gain * self.conv_block(img)
        return output


def upsample(x):
    """
    Doubles resolution.

    Args:
        x: [batch, W, H, ..., C]

    Returns:
        [batch, 2*W, 2*H, 2*..., C]
    """
    out = x
    for i in range(len(x.shape)-2):
        out = tf.repeat(out, 2, axis=1+i)
    return out


class Decoder2D(AbstractModule):
    def __init__(self, hidden_size, num_channels=1, name=None):
        super(Decoder2D, self).__init__(name=name)
        num_groups = 4
        num_blk_per_group = 1
        num_layers = num_groups * num_blk_per_group
        post_gain = 1. / num_layers ** 2

        def _single_group(group_idx):
            blk_hidden_size = 2 ** (num_groups - group_idx - 1) * hidden_size
            res_blocks = [DecoderResBlock2D(blk_hidden_size, post_gain, name=f'blk_{res_blk}')
                          for res_blk in range(num_blk_per_group)]
            if group_idx < num_groups - 1:
                res_blocks.append(upsample)
            return snt.Sequential(res_blocks, name=f'group_{group_idx}')

        groups = [snt.Conv2D(hidden_size // 2, 1, padding="SAME", name='input_group')]
        for groud_idx in range(num_groups):
            groups.append(_single_group(groud_idx))
        groups.append(
            snt.Sequential(
                [tf.nn.relu, snt.Conv2D(num_channels * 2, 1, padding="SAME", name='likelihood_params_conv')],
                name='output_group'))

        self.blocks = snt.Sequential(groups, name='groups')

    def _build(self, img, **kwargs):
        return self.blocks(img)


class DecoderResBlock3D(AbstractModule):
    def __init__(self, out_size, post_gain, name=None):
        super(DecoderResBlock3D, self).__init__(name=name)
        assert out_size % 4 == 0
        self.out_size = out_size
        hidden_size = out_size // 4
        self.id_path = snt.Conv3D(self.out_size, 1, name='id_path')
        self.conv_block = snt.Sequential([snt.LayerNorm(-1, True, True, name='layer_norm'),
                                          tf.nn.relu, snt.Conv3D(hidden_size, 1, padding="SAME", name='conv_1'),
                                          tf.nn.relu, snt.Conv3D(hidden_size, 3, padding="SAME", name='conv_2'),
                                          tf.nn.relu, snt.Conv3D(hidden_size, 3, padding="SAME", name='conv_3'),
                                          tf.nn.relu, snt.Conv3D(out_size, 3, padding="SAME", name='conv_4')])
        self.post_gain = post_gain

    def _build(self, img, **kwargs):
        # self.initialise(img)
        output = self.id_path(img) + self.post_gain * self.conv_block(img)
        return output

class Decoder3D(AbstractModule):
    def __init__(self, hidden_size, num_channels=1, name=None):
        super(Decoder3D, self).__init__(name=name)
        num_groups = 4
        num_blk_per_group = 1
        num_layers = num_groups * num_blk_per_group
        post_gain = 1. / num_layers ** 2

        def _single_group(group_idx):
            blk_hidden_size = 2 ** (num_groups - group_idx - 1) * hidden_size
            res_blocks = [DecoderResBlock3D(blk_hidden_size, post_gain, name=f'blk_{res_blk}')
                          for res_blk in range(num_blk_per_group)]
            if group_idx < num_groups - 1:
                res_blocks.append(upsample)
            return snt.Sequential(res_blocks, name=f'group_{group_idx}')

        groups = [snt.Conv3D(hidden_size // 2, 1, padding="SAME", name='input_group')]
        for groud_idx in range(num_groups):
            groups.append(_single_group(groud_idx))
        groups.append(
            snt.Sequential(
                [tf.nn.relu, snt.Conv3D(num_channels * 2, 1, padding="SAME", name='likelihood_params_conv')],
                name='output_group'))

        self.blocks = snt.Sequential(groups, name='groups')

    def _build(self, img, **kwargs):
        return self.blocks(img)


class EncoderResBlock3D(AbstractModule):
    """
    O[out] = sum_{i,j,k,in} W[i,j,k,in,out]*I[i,j,k,in]
    count = kernel^3 * N_in
    """
    def __init__(self, out_size, post_gain, name=None):
        super(EncoderResBlock3D, self).__init__(name=name)
        assert out_size % 4 == 0
        self.out_size = out_size
        hidden_size = out_size // 4
        self.id_path = snt.Conv3D(self.out_size, 1, name='id_path')
        self.conv_block = snt.Sequential([snt.LayerNorm(-1,True, True,name='layer_norm'),
                                          tf.nn.relu, snt.Conv3D(hidden_size, 3, padding="SAME", name='conv_1'),
                                          tf.nn.relu, snt.Conv3D(hidden_size, 3, padding="SAME", name='conv_2'),
                                          tf.nn.relu, snt.Conv3D(hidden_size, 3, padding="SAME", name='conv_3'),
                                          tf.nn.relu, snt.Conv3D(out_size, 1, padding="SAME", name='conv_4')])

        self.post_gain = post_gain

    def _build(self, img, **kwargs):
        # self.initialise(img)
        output = self.id_path(img) + self.post_gain * self.conv_block(img)
        return output


class Encoder3D(AbstractModule):
    def __init__(self, hidden_size, num_embeddings, name=None):
        super(Encoder3D, self).__init__(name=name)
        num_groups = 4
        num_blk_per_group = 1
        num_layers = num_groups * num_blk_per_group
        post_gain = 1. / num_layers ** 2

        def _single_group(group_idx):
            blk_hidden_size = 2 ** group_idx * hidden_size
            res_blocks = [EncoderResBlock3D(blk_hidden_size, post_gain, name=f'blk_{res_blk}')
                          for res_blk in range(num_blk_per_group)]
            if group_idx < num_groups - 1:
                res_blocks.append(lambda x: tf.nn.max_pool3d(x, 2, strides=2, padding='SAME'))
            return snt.Sequential(res_blocks, name=f'group_{group_idx}')

        groups = [snt.Conv3D(hidden_size, 7, padding="SAME", name='input_group')]
        for groud_idx in range(num_groups):
            groups.append(_single_group(groud_idx))
        groups.append(
            snt.Sequential([tf.nn.relu, snt.Conv3D(num_embeddings, 1, padding="SAME", name='logits_conv')],
                           name='output_group'))

        self.blocks = snt.Sequential(groups, name='groups')

    def _build(self, img, **kwargs):
        return self.blocks(img)
