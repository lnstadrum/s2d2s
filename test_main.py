import pytest
import torch

from . import space_to_depth, depth_to_space, SpaceToDepth, DepthToSpace


@pytest.mark.parametrize("outer_dims", [(), (3, ), (2, 1), (2, 3, 4)])
@pytest.mark.parametrize("block_size", [2, 3])
def test_round_trip_functional_api(outer_dims, block_size):
    """ Ensures consistency of D2S and S2D operations with each other for different number of dimensiosn and block sizes.
    """
    x1 = torch.rand(*outer_dims, 5 * (block_size**2), 19, 29)
    x2 = depth_to_space(x1, block_size)
    x3 = space_to_depth(x2, block_size)
    assert (x1 == x3).all()

    x1 = torch.rand(*outer_dims, 2, 20 * block_size, 30 * block_size)
    x2 = space_to_depth(x1, block_size)
    x3 = depth_to_space(x2, block_size)
    assert (x1 == x3).all()


@pytest.mark.parametrize("outer_dims", [(), (2, ), (3, 4)])
@pytest.mark.parametrize("block_size", [4, 5])
def test_round_trip_module_api(outer_dims, block_size):
    """ Ensures consistency of D2S and S2D operations with each other for different number of dimensiosn and block sizes.
    """

    d2s = DepthToSpace(block_size)
    s2d = SpaceToDepth(block_size)

    x1 = torch.rand(*outer_dims, 1, 2 * block_size, 3 * block_size)
    x2 = s2d(x1)
    x3 = d2s(x2)
    assert (x1 == x3).all()

    x1 = torch.rand(*outer_dims, 5 * (block_size**2), 3, 4)
    x2 = d2s(x1)
    x3 = s2d(x2)
    assert (x1 == x3).all()


def test_argument_errors():
    """ Asserts on arguments check
    """
    # check input dimensions
    with pytest.raises(ValueError):
        space_to_depth(torch.ones(1, 2), 3)
    y = space_to_depth(torch.ones(1, 2, 3, 3), 3)
    assert y.shape[-2:] == (1, 1)

    with pytest.raises(ValueError):
        depth_to_space(torch.ones(1, 2), 3)
    depth_to_space(torch.ones(1, 9, 1, 1), 3)

    # checks number of channels/spatial dims
    with pytest.raises(ValueError):
        depth_to_space(torch.ones(1, 10, 4, 4), 3)
    depth_to_space(torch.ones(1, 9, 4, 4), 3)

    with pytest.raises(ValueError):
        space_to_depth(torch.ones(1, 1, 5, 4), 4)
    space_to_depth(torch.ones(1, 1, 4, 4), 4)


def test_repr():
    """ String representation check
    """
    print(SpaceToDepth(10))
    print(DepthToSpace(20))


def test_torchscript_export():
    """ TorchScrpit export check
    """
    x = torch.rand(2, 4, 8, 16)

    d2s = torch.jit.trace(DepthToSpace(2), x)
    d2s(x)

    s2d = torch.jit.trace(SpaceToDepth(2), x)
    s2d(x)
