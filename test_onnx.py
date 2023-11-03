import io
import onnxruntime
import pytest
import torch

from s2d2s import SpaceToDepth, DepthToSpace


@pytest.mark.parametrize("module", [SpaceToDepth, DepthToSpace])
def test_onnx_export(module):
    # make a model
    model = module(2)

    # compute reference output
    input = torch.rand(3, 8, 10, 10)
    ref_output = model(input)

    # export to onnx
    buffer = io.BytesIO()
    torch.onnx.export(model,
                      input,
                      buffer,
                      opset_version=9,
                      input_names=["input"],
                      output_names=["output"])

    # run with onnx
    buffer.seek(0)
    session = onnxruntime.InferenceSession(buffer.read())
    test_output, = session.run(None, {"input": input.numpy()})

    # enxure consistency
    assert (ref_output.numpy() == test_output).all()
