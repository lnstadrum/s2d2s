from . import space_to_depth, depth_to_space

import tensorflow as tf
import torch
import pytest


@pytest.mark.parametrize("block_size", [2, 3])
def test_tensorflow(block_size):
    """ Compares with TensorFlow.
    """
    # sample random input, compute outputs
    x = torch.rand(3, 5 * (block_size**2), 12, 34)
    x1 = depth_to_space(x, block_size)
    x2 = space_to_depth(x1, block_size)

    # permute to channes-last for Tensorflow
    y = tf.transpose(x.numpy(), [0, 2, 3, 1])

    # compute the reference outputs
    y1 = tf.nn.depth_to_space(y, block_size)
    y2 = tf.nn.space_to_depth(y1, block_size)

    # NHWC -> NCHW
    y = tf.transpose(y, [0, 3, 1, 2])
    y1 = tf.transpose(y1, [0, 3, 1, 2])
    y2 = tf.transpose(y2, [0, 3, 1, 2])

    # compare
    assert tf.reduce_all(x.numpy() == y)
    assert tf.reduce_all(x1.numpy() == y1)
    assert tf.reduce_all(x2.numpy() == y2)
