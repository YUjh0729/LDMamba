from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

import torch
from torch.optim import AdamW
from torch import nn

from monai.networks.nets import HighResNet

import torch.nn as nn
from thop import profile, clever_format
import torch
from functools import partial
from fvcore.nn import FlopCountAnalysis, flop_count


class nnUNetTrainerHighResNet(nnUNetTrainerNoDeepSupervision):
    def __init__(
                self,
                plans: dict,
                configuration: str,
                fold: int,
                dataset_json: dict,
                unpack_dataset: bool = True,
                device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.initial_lr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 0.01

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)
        spatial_dims = len(configuration_manager.patch_size)

        model = HighResNet(
            spatial_dims=spatial_dims,  # 输入数据的空间维度
            in_channels=num_input_channels,  # 输入图像的通道数
            out_channels=label_manager.num_segmentation_heads,  # 输出图像的通道数
        )

        # # print("HighResNet: {}".format(model))
        #
        # # ---------------------yjh----------------------
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input = torch.randn(1, 1, 320, 320).to(device)
        # model = model.to(device)
        #
        # def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
        #     assert not with_complex
        #     flops = 9 * B * L * D * N
        #     if with_D:
        #         flops += B * D * L
        #     if with_Z:
        #         flops += B * D * L
        #     return flops
        #
        # def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
        #     # print_jit_input_names(inputs)
        #     B, D, L = inputs[0].type().sizes()
        #     N = inputs[2].type().sizes()[1]
        #     flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
        #     return flops
        #
        # # 定义支持的操作字典
        # supported_ops = {
        #     "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn),
        # }
        # # 计算 FLOPs
        # Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        # print("-----------------------yjh--------------------------")
        # # 计算总的 FLOPs
        # total_flops = sum(Gflops.values())
        # # 转换为 GFLOPs
        # total_gflops = total_flops / 1e9
        # # 打印结果
        # print("FLOPs:", total_flops)
        # # print("Total GFLOPs:", total_gflops, "GFLOPs")
        # # 计算参数数量
        # total_params = sum(p.numel() for p in model.parameters())
        # params = clever_format([total_params], "%.3f")
        # print("Params:", params)
        #
        # print("-----------------------thop计算结果--------------------------")
        # # 使用 thop 库计算 FLOPs 和参数数量
        # macs, params_thop = profile(model, inputs=(input,))
        # # 计算 FLOPs
        # flops_thop = 2 * macs  # 一个 MAC 操作等价于两次 FLOPs
        # # 使用 clever_format 函数格式化输出
        # flops_thop, params_thop = clever_format([flops_thop, params_thop], "%.3f")
        # print("GFLOPs:", flops_thop)
        # print("Params:", params_thop)
        return model
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.0)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass