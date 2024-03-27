# 4. Write some code to traverse both `mg` and `ori_mg`, 
# check and comment on the nodes in these two graphs. 
# You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity
import logging
import os
import torch.nn as nn

import numpy as np
import pandas as pd
from tabulate import tabulate

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target

from chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    report_graph_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph

from chop.models import get_model_info, get_model

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph
from chop.ir.graph.mase_metadata import MaseMetadata

set_logging_verbosity("info")

from chop.passes.graph.transforms.quantize.quantized_modules.linear import _LinearBase, LinearInteger

from functools import partial



#########################################################################################################

### instantiate the same dataset and model as in Lab1
# Set up parameters, including batch size, model name, and dataset name
batch_size = 8
model_name = "jsc-toy"
dataset_name = "jsc"

# Load model, dataset, and data module
data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

# 📝️ change this CHECKPOINT_PATH to the one you trained in Lab1
CHECKPOINT_PATH = "/home/xinyi/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

#########################################################################################################

### get dummy data in
# dummy data is a dictionary that contains the input data, this input data is a dictionary that contains the input tensor
# get the input generator
input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

########################################################################################################
#pass_args is a dictionary that contains the arguments for the pass
# first declare the configuration for the pass, then pass the configuration to the pass of the graph
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "relu": {
        "config": {
            "name": "integer",
            "data_in_width": 16,
            "data_in_frac_width": 8,
            "weight_width": 16,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}
#########################################################################################################
### generate a mase model and apply passes
# generate a mase model
# this is original mase model

ori_mg = MaseGraph(model=model)

#########################################################################################################
### running a analysis pass
# analysis passes
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})
ori_mg, _ = add_software_metadata_analysis_pass(ori_mg, None)

# report graph analysis, there's a print out
_ = report_graph_analysis_pass(ori_mg)
print("report original pass")
ori_mg, _ = report_node_meta_param_analysis_pass(ori_mg, {"which": ("software",)})

mg = ori_mg
# mg = MaseGraph(model=model)
# mg, _ = init_metadata_analysis_pass(mg, None)
# mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
# mg, _ = add_software_metadata_analysis_pass(mg, None)
mg, _ = quantize_transform_pass(ori_mg, pass_args)
    
_ = report_graph_analysis_pass(mg)
print("report transform pass")
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})

print("show the diff")
summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")

print("finish jsc-toy analysis pass")

result_integer_number = 0
args_weights_precision = 0
args_bias_precision = 0

for node in mg.fx_graph.nodes:
    if get_mase_op(node) == 'linear':
        result_integer_number = node.meta['mase'].parameters['common']['args']['data_in_0']['precision']
        print('the results shape', result_integer_number)
        args_weights_precision = node.meta['mase'].parameters['common']['args']['weight']['precision']
        print('the weights torch type', args_weights_precision)
        args_bias_precision = node.meta['mase'].parameters['common']['args']['bias']['precision']
        print('the bias torch type', args_bias_precision)
        
    if node.op == 'call_module':
        layer = get_node_actual_target(node)
        if isinstance(layer, nn.Linear):
            result_integer_number = layer.weight.data.shape
            print('the results torch type', result_integer_number)
            args_weights_precision = layer.weight.data.shape
            print('the weights torch type', args_weights_precision)