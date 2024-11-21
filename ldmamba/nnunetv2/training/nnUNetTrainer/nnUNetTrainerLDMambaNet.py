import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from nnunetv2.nets.LDMambaNet_2d import get_LDMambaNet_from_plans_2d
from nnunetv2.nets.LDMambaNet_3d import get_LDMambaNet_from_plans_3d



class nnUNetTrainerLDMambaNet(nnUNetTrainer):
    """
    MambaUNet Encoder + Residual Decoder + Skip Connections
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            print("-----------------2d_model-------------------")
            model = get_LDMambaNet_from_plans_2d(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            print("-----------------3d_model-------------------")
            model = get_LDMambaNet_from_plans_3d(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        return model