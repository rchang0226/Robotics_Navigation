import io
import numpy as np
import torch.onnx
import torch
import os
from utils import *
import pickle
import onnx
import onnxruntime

policy_net, value_net, running_state = pickle.load(
    open(
        os.path.join(
            assets_dir(), "learned_models/{}_ppo_rand_best.p".format("navigation", 5)
        ),
        "rb",
    )
)

torch.save(policy_net.state_dict(), "policy.pth")

# x = torch.randn(1, 5, 128, 160, requires_grad=True, dtype=torch.double)
# y = torch.randn(1, 5, 3, requires_grad=True, dtype=torch.double)
# z = torch.randn(1, 5, 1, requires_grad=True, dtype=torch.double)
# w = torch.randn(1, 5, 2, requires_grad=True, dtype=torch.double)

# torch_out = policy_net.forward(x, y, z, w)

# onnx_model = onnx.load("policy.onnx")
# print(onnx.checker.check_model(onnx_model))

# ort_session = onnxruntime.InferenceSession("policy.onnx")


# def to_numpy(tensor):
#     return (
#         tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#     )


# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# Not dealing with dynamic axes for now
# torch.onnx.export(
#     policy_net,
#     (x, y, z, w),
#     "policy.onnx",
#     export_params=True,
#     opset_version=10,
#     do_constant_folding=True,
#     input_names=["depth", "goal", "ray", "act"],
#     output_names=["action_mean", "action_log_std", "action_std"],
# )
