# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
import copy
from torch import nn
import torch
from torch.nn import ReLU
from ..base import SearchSpaceBase
from .....passes.graph.transforms.quantize import (
    QUANTIZEABLE_OP,
    quantize_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target, get_parent_name
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict

from nas_201_api import NASBench201API as API
from .xautodl.models import get_cell_based_tiny_net

from .model_spec import ModelSpec

# from nas_graph import load_bench_arch

### default architecture is the architecuture returned 
### by api.get_net_config(0, 'cifar10') in nasbench201
DEFAULT_ZERO_COST_ARCHITECTURE_CONFIG = {
    "config": {'name': ['infer.tiny'], 
    'C': [16], 
    'N': [5], 
    'op_0_0': [0], 
    'op_1_0': [4], 
    'op_2_0': [2], 'op_2_1': [1], 
    'op_3_0': [2], 'op_3_1': [1], 'op_3_2': [1], 
    'number_classes': [10]}
}

print("loading api")
api = API('./third_party/NAS-Bench-201-v1_1-096897.pth', verbose=False)
# api = API('/home/xz2723/mase_xinyi/machop/third_party/NAS-Bench-201-v1_1-096897.pth', verbose=False)
print("api loaded")

class ZeroCostProxy(SearchSpaceBase):
    """
    Post-Training quantization search space for mase graph.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_ZERO_COST_ARCHITECTURE_CONFIG
        
        # quantize the model by type or name
        # assert (
        #     "by" in self.config["setup"]
        # ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"

    def rebuild_model(self, sampled_config, is_eval_mode: bool = False):
        print("sampled_config")
        print(sampled_config)
        
        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()
        # import pdb; pdb.set_trace()
        if "nas_zero_cost" in sampled_config:
            nas_config = generate_configs(sampled_config["nas_zero_cost"])
        else:
            nas_config = generate_configs(sampled_config["default"])
        # print("nas_config")
        # print(nas_config)
        
        arch = nas_config["arch_str"]
        index = api.query_index_by_arch(arch)
        print("index")
        print(index)
        results = api.query_by_index(index, 'cifar10')
        print("results")
        print(results)
        data = api.get_more_info(index, 'cifar10')
        
        model_arch = get_cell_based_tiny_net(nas_config)
        model_arch = model_arch.to(self.accelerator)

        return model_arch, data

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the zero-cose
        """
        choices = {}
        choices["nas_zero_cost"] = self.config["nas_zero_cost"]["config"]

        print("choices: +=================+")
        print(choices)

        for key, value in DEFAULT_ZERO_COST_ARCHITECTURE_CONFIG["config"].items():
            if key in choices["nas_zero_cost"]:
                continue
            else:
                choices["nas_zero_cost"][key] = value
        
        # flatten the choices and choice_lengths
        # self.choices_flattened = {}
        flatten_dict(choices, flattened=self.choices_flattened)
        
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }
        

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        flattened_config = {}
        # import pdb; pdb.set_trace() 
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        return config
    
    # def build_search_space(self, config_all):
    #     # choises = {}
        
    #     self.choices_flattened = generate_configs(config_all)
        
    #     self.choice_lengths_flattened = {
    #         k: len(v) for k, v in self.choices_flattened.items()
    #     }

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def instantiate_relu(inplace):
    return ReLU(inplace)

def instantiate_batchnorm(num_features, eps, momentum, affine, track_running_stats):
    return nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

def instantiate_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        bias = bias,
        padding_mode = 'same',
        device = None,
        dtype = None,
    )

import itertools

def generate_configs(config_dict):

    name = config_dict['name']
    C = config_dict['C']
    N = config_dict['N']
    num_classes = config_dict['number_classes']
    op_map = {0:'skip_connect', 1:'none', 2:'nor_conv_3x3', 3:'nor_conv_1x1', 4:'avg_pool_3x3'}

    ### generate combination
    arch_str = ""
    for target_neuro in range(1, 4):
        arch_str += "|"
        for exert_neuro in range(0, target_neuro):
            op = f"op_{target_neuro}_{exert_neuro}"
            op_str = op_map[config_dict[op]]
            op_str += f"~{exert_neuro}"
            arch_str += (op_str + "|")
        if target_neuro < 3:
            arch_str += "+"
    
    config = {
        'name': name,
        'C': C,
        'N': N,
        'arch_str': arch_str,
        'num_classes': num_classes
    }

    return config