# Overview

`PixelShuffle`/`PixelUnshuffle` in PyTorch and `depth_to_space`/`space_to_depth` in TensorFlow are very similar, but they are not numerically identical unless the upsampled image contains a single channel. This can be easily verified by hand: while the tensor dimensions match between the PyTorch and TensorFlow worlds, the output channels do not follow the same order.

However, in some deployment setups, there might be performance benefits to using the space-to-depth/depth-to-space variant. For example, at the moment of writing, [Android NN API](https://developer.android.com/ndk/guides/neuralnetworks) only supports depth-to-space and space-to-depth.

This repository provides a unit-tested space-to-depth/depth-to-space implementation for PyTorch, supporting conversion to [SpaceToDepth](http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx__SpaceToDepth.html)/[DepthToSpace](http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx__DepthToSpace.html) ONNX ops when exporting the model as an ONNX graph for further deployment.

The provided implementation follows channels-first PyTorch standard, allowing for arbitrary number of outer dimensions, i.e. it supports tensors of `[..., C, H, W]` layouts.

# Installation

```
python3 -m pip install s2d2s
```

# Usage

Both functional implementations and `nn.Module`s are available and can be used as follows:

```
from s2d2s import space_to_depth, depth_to_space

y = space_to_depth(x, 2)
```

```
from s2d2s import SpaceToDepth, DepthToSpace

module = SpaceToDepth(2)
y = module(x)
```
