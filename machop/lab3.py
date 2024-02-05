import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import time

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target

# figure out the correct path, change tomy tree
machop_path = Path(".").resolve().parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model


#########################################################################################################
### build the mase graph model

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

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

#########################################################################################################
### defining a search space

pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))

#########################################################################################################
### defining a search strategy and a runner
# grid search


import torch
import torch.nn as nn
import torchmetrics
import numpy as np
from torchmetrics.classification import MulticlassAccuracy

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

recorded_accs = []
model_size = 0
recorded_size = []
layers_number = 0
layer_numbers_2 = 0

relu_flop_number = 0
flops_number = 0
batch_norm1d_flop_number = 0
linear_flop_number = 0

relu_flop_number_node = 0
flops_number_node = 0
batch_norm1d_flop_number_node = 0
linear_flop_number_node = 0

recorded_latencies = []

def get_model_size(mg):
    model_size = 0
    for param in mg.model.parameters():
        print('parameters of element size', param.nelement(), param.element_size())
        model_size += param.nelement() * param.element_size()
    return model_size

for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    # evaluate(mg, data_module, metric, num_batchs, recorded_accs)
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    latency = 0
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start = time.time()
        preds = mg.model(xs)
        end = time.time()
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1
        # layer = get_mase_type(mg.fx_graph.nodes[1])
        # if layer == 'linear':
        #     print('in out features in path', mg.model.layer.in_features, mg.model.layer.out_features)
        #     flops_number += mg.model.layer.in_features * mg.model.layer.out_features
        #     print('flop numbers 1', flops_number)
        #     layers_number += 1
        #     print('layers number', layers_number)

        latency += end - start
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    
    # # model size
    # model_size = get_model_size(mg)
    # print("model size: ", model_size)
    # recorded_size.append(model_size)
    # # print(model.relu.weight)
    
    for node in mg.fx_graph.nodes:
        # print(mg.model.linear.weight)
        # if node.op == 'call_module':
        #     layer = get_node_actual_target(node)
        #     if isinstance(layer, nn.Linear):
        #         print('in out features by modules', layer.in_features, layer.out_features)
        #         flops_number += layer.in_features * layer.out_features
        #         print('flop numbers 1', flops_number)
        #         layers_number += 1
        #         print('layers number', layers_number)
            
    # for node in mg.fx_graph.nodes:
        if get_mase_op(node) == 'linear':
            # batchsize = mg.model.in_data.shape[0] / mg.model.in_data.shape[-1]
            # print('batch size number', batchsize)
            # module = get_node_actual_target(node)
            # batch_size = module.in_data.shape[0] / module.in_data.shape[-1]
            # batch_size = node.meta['mase'].parameters['common']['args']['data_in_0'].numel() / node.meta['mase'].parameters['common']['args']['weight']['shape'][-1]
            # print('batch size number', batch_size)
            linear_flop_number += batch_size * node.meta['mase'].parameters['common']['args']['data_in_0']['shape'][1] * node.meta['mase'].parameters['common']['args']['data_in_0']['shape'][0]
            print('linear flop numbers', linear_flop_number)
        
        if get_mase_op(node) == 'relu':
            relu_flop_number += node.meta['mase'].parameters['common']['args']['data_in_0']['shape'][1]
            print('relu flop numbers', relu_flop_number)
        
        if get_mase_op(node) == 'batch_norm1d':
            batch_norm1d_flop_number += 4 * node.meta['mase'].parameters['common']['args']['data_in_0']['shape'][1]
            print('batch_norm1d flop numbers', batch_norm1d_flop_number)
        
        flops_number = relu_flop_number + batch_norm1d_flop_number + linear_flop_number
        print('flop numbers total', flops_number)

recorded_accs.append(acc_avg)
recorded_latencies.append(latency)
# recorded_model_sizes.append(model_size)
# recorded_flops.append(flops)
print("recorded_acc", recorded_accs)
print("recorded_latency", recorded_latencies)