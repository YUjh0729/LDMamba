import torch.nn as nn
from thop import profile, clever_format
import torch
from functools import partial
from fvcore.nn import FlopCountAnalysis, flop_count

# ---------------------yjh----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randn(1, 1, 320, 320).to(device)
model = model.to(device)


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    assert not with_complex
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    # print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# 定义支持的操作字典
supported_ops = {
    "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
}
# 计算 FLOPs
Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
print("-----------------------yjh--------------------------")
# 计算总的 FLOPs
total_flops = sum(Gflops.values())
# 转换为 GFLOPs
total_gflops = total_flops / 1e9
# 打印结果
print("Total FLOPs:", total_flops, "FLOPs")
# print("Total GFLOPs:", total_gflops, "GFLOPs")
# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
params = clever_format([total_params], "%.3f")
print("Total parameters:", params)