import torch
from typing import Tuple


def _depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x


def _space_to_depth(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Space-to-Depth core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with H and W divisible by {block_size}, but got H={h} and W={w}"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the spatial ones (H and W)
    x = x.view(-1, c, h // block_size, block_size, w // block_size, block_size)

    # putting the new dimensions in front of C
    x = x.permute(0, 3, 5, 1, 2, 4)

    # merging the two new dims with C
    x = x.contiguous().view(*outer_dims, c * (block_size**2), h // block_size,
                            w // block_size)

    return x


class _DepthToSpaceFunction(torch.autograd.Function):
    """ Depth-to-Space autograd function with onnx op support
    """

    @staticmethod
    def symbolic(g, x, blocksize: int):
        return g.op("DepthToSpace", x, blocksize_i=blocksize)

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int) -> torch.Tensor:
        ctx.block_size = block_size
        return _depth_to_space(x, block_size)

    @staticmethod
    def backward(ctx, y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return _space_to_depth(y, ctx.block_size), None


class _SpaceToDepthFunction(torch.autograd.Function):
    """ Space-to-Depth autograd function with onnx op support
    """

    @staticmethod
    def symbolic(g, x, blocksize: int):
        return g.op("SpaceToDepth", x, blocksize_i=blocksize)

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int) -> torch.Tensor:
        ctx.block_size = block_size
        return _space_to_depth(x, block_size)

    @staticmethod
    def backward(ctx, y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return _depth_to_space(y, ctx.block_size), None


def depth_to_space(x: torch.Tensor, block_size: int):
    """ Depth-to-Space DCR mode (depth-column-row) functional implementation

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    return _DepthToSpaceFunction.apply(x, block_size)


def space_to_depth(x: torch.Tensor, block_size: int):
    """ Space-to-Depth functional implementation

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    return _SpaceToDepthFunction.apply(x, block_size)


class DepthToSpace(torch.nn.Module):
    """ Depth-to-Space DCR mode torch module

        Args:
            block_size (int): block side size
    """

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depth_to_space(x, self.block_size)

    def extra_repr(self) -> str:
        return super().extra_repr() + str(self.block_size)


class SpaceToDepth(torch.nn.Module):
    """ Space-to-Depth (depth-column-row) torch module

        Args:
            block_size (int): block side size
    """

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return space_to_depth(x, self.block_size)

    def extra_repr(self) -> str:
        return super().extra_repr() + str(self.block_size)
