#___
# Lab 2 - Optional: Write pass to count FLOPs and BitOPs
#___

# from chop.passes.graph.analysis.flop_estimator.calc_modules import calculate_modules

import torch


def calculate_modules(module, in_data, out_data):
    # Collect computation statistics.
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        # One computation per input pixel - window size is chosen adaptively
        # and windows never overlap (?).
        assert len(in_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        computations = input_size
        backward_computations = input_size
        return {
            "total_parameters": 0,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Embedding):
        total_parameters = module.embedding_dim * in_data[0].numel()
        return {
            "total_parameters": total_parameters,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": 0,
            "output_buffer_size": 0,
        }
    elif isinstance(module, torch.nn.AvgPool2d) or isinstance(
        module, torch.nn.MaxPool2d
    ):
        # Each output pixel requires computations on a 2D window of input.
        if type(module.kernel_size) == int:
            # Kernel size here can be either a single int for square kernel
            # or a tuple (see
            # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d )
            window_size = module.kernel_size**2
        else:
            window_size = module.kernel_size[0] * module.kernel_size[1]

        # Not sure which output tensor to use if there are multiple of them.
        assert len(out_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        computations = output_size * window_size
        backward_computations = input_size * window_size
        return {
            "total_parameters": 0,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Conv2d):
        # Each output pixel requires computations on a 3D window of input.
        # Not sure which input tensor to use if there are multiple of them.
        assert len(in_data) == 1
        _, channels, _, _ = in_data.size()
        window_size = module.kernel_size[0] * module.kernel_size[1] * channels

        # Not sure which output tensor to use if there are multiple of them.
        assert len(out_data) == 1
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()

        computations = output_size * window_size
        backward_computations = input_size * window_size * 2
        return {
            "total_parameters": module.weight.numel(),
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.Dropout2d) or isinstance(
        module, torch.nn.modules.dropout.Dropout
    ):
        return {
            "total_parameters": 0,
            "computations": 0,
            "backward_computations": 0,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.Linear):
        # One computation per weight, for each batch element.

        # Not sure which input tensor to use if there are multiple of them.
        # TODO: check if this is correct
        # TODO: also consider bias?
        assert len(in_data) == 1
        batch = in_data[0].numel() / in_data[0].shape[-1]

        computations = module.weight.numel() * batch
        backward_computations = module.weight.numel() * batch * 2
        input_size = in_data[0].numel()
        output_size = out_data[0].numel()
        return {
            "total_parameters": module.weight.numel(),
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }

    elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(
        module, torch.nn.modules.activation.ReLU6
    ):
        # ReLU does a single negation check
        return {
            "total_parameters": 0,
            "computations": in_data[0].numel(),
            "backward_computations": in_data[0].numel(),
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.LayerNorm):
        return {
            "total_parameters": 0,
            "computations": in_data[0].numel() * 5,
            "backward_computations": in_data[0].numel() * 5,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        # Accesses to E[x] and Var[x] (all channel size)
        total_parameters = 2 * module.num_features
        # (x-running_mean)/running variance
        # multiply by gamma and beta addition
        computations = 4 * in_data[0].numel()
        backward_computations = 4 * in_data[0].numel()
        return {
            "total_parameters": total_parameters,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }

    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
        # Accesses to E[x] and Var[x] (all channel size)
        total_parameters = 2 * module.num_features
        # (x-running_mean)/running variance
        # multiply by gamma and beta addition
        computations = 4 * in_data[0].numel()
        backward_computations = 4 * in_data[0].numel()
        return {
            "total_parameters": total_parameters,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": in_data[0].numel(),
            "output_buffer_size": out_data[0].numel(),
        }
    else:
        print("Unsupported module type for analysis:", type(module))
        
########################################################################################
## bitops calculation number

def calculate_modules_bitop(node, module, in_data, out_data):
    if isinstance(module, torch.nn.Linear):
        assert len(in_data) == 1
        batch = in_data[0].numel() / in_data[0].shape[-1]
        computations = module.weight.numel() * batch
        backward_computations = module.weight.numel() * batch * 2
        bitops = computations * 32
        
        if node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"] == torch.float32:
            bitops = computations * 32
        elif node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"] == torch.float16:
            bitops = computations * 16
        else:
            bitops = computations * 8
        return {
            "computations": computations,
            "bitops": bitops,
        }

    elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(
        module, torch.nn.modules.activation.ReLU6
    ):
        computations = in_data[0].numel()
        if node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"] == torch.float32:
            bitops = computations * 32
        elif node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"]== torch.float16:
            bitops = computations * 16
        else:
            bitops = computations * 8
        return {
            "computations": computations,
            "bitops": bitops,
        }

    elif isinstance(module, torch.nn.LayerNorm):
        computations = in_data[0].numel() * 5
        if node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"] == torch.float32:
            bitops = computations * 32
        elif node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"] == torch.float16:
            bitops = computations * 16
        else:
            bitops = computations * 8
        return {
            "computations": computations,
            "bitops": bitops,
        }

    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
        # Accesses to E[x] and Var[x] (all channel size)
        total_parameters = 2 * module.num_features
        computations = 4 * in_data[0].numel()
        backward_computations = 4 * in_data[0].numel()
        # import pdb; pdb.set_trace()
        
        if node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"] == torch.float32:
            bitops = computations * 32
        elif node.meta['mase'].parameters['common']['args']['data_in_0']["torch_dtype"] == torch.float16:
            bitops = computations * 16
        else:
            bitops = computations * 8
        return {
            "computations": computations,
            "bitops": bitops,
        }
    else:
        print("Unsupported module type for analysis:", type(module))


import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target

# figure out the correct path, change tomy tree
machop_path = Path(".").resolve().parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model


import torch
import torch.nn as nn
import numpy as np

# from chop.actions.search.strategies.lab3_brute_force import BruteForceSearchStrategy

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})



from chop.passes.graph.utils import get_node_actual_target

flop = 0
store_flops_calculation = {}
# import pdb; pdb.set_trace()
for node in mg.fx_graph.nodes:
    try:
        in_data = (node.meta['mase'].parameters['common']['args']['data_in_0']['value'],)
    except KeyError:
        in_data = (None,)
    out_data = (node.meta['mase'].parameters['common']['results']['data_out_0']['value'],)

    module = get_node_actual_target(node)
    if isinstance(module, torch.nn.Module):
        current_flops = calculate_modules(module, in_data, out_data)
        store_flops_calculation[module] = current_flops
        flop += current_flops['computations']
        
print("store_flops_data", store_flops_calculation)

bitops = 0
store_bitops_calculation = {}
for node in mg.fx_graph.nodes:
    try:
        in_data = (node.meta['mase'].parameters['common']['args']['data_in_0']['value'],)
    except KeyError:
        in_data = (None,)
    out_data = (node.meta['mase'].parameters['common']['results']['data_out_0']['value'],)

    module = get_node_actual_target(node)
    if isinstance(module, torch.nn.Module):
        current_bitops = calculate_modules_bitop(node, module, in_data, out_data)
        store_bitops_calculation[module] = current_bitops
        bitops += current_bitops['bitops']

print("store_bitops_data", store_bitops_calculation)
print("flops", flop)
print("bitops", bitops)