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

set_logging_verbosity("info")


#########################################################################################################

### instantiate the same dataset and model as in Lab1
# Set up parameters, including batch size, model name, and dataset name
batch_size = 32
model_name = "test"
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
CHECKPOINT_PATH = "/home/xinyi/ADL/mase_xinyi/mase_output/test_classification_jsc_2024-01-23/software/training_ckpts/best.ckpt"
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

#########################################################################################################
### generate a mase model and apply passes
# generate a mase model
# this is original mase model

ori_mg = MaseGraph(model=model)

#########################################################################################################
# pass_args is a dictionary that contains the arguments for the pass
# first declare the configuration for the pass, then pass the configuration to the pass of the graph
pass_args = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}

# info = {
#     "weight_statistics": {
#         "variance_precise": {"device": "cpu", "dims": "all"},
#     },
# }
# format of the path, how to use this function for the definition???????????????????????????????????????????????
# def pass(mg, pass_args):
#     pass_args = {
#         "by": "type",
#         "default": {"config": {"name": None}},
#         "linear": {
#             "config": {
#                 "name": "integer",
#                 # data
#                 "data_in_width": 8,
#                 "data_in_frac_width": 4,
#                 # weight
#                 "weight_width": 8,
#                 "weight_frac_width": 4,
#                 # bias
#                 "bias_width": 8,
#                 "bias_frac_width": 4,
#             }
#         },
#     }
#     info = {}
#     return mg, info



#########################################################################################################
### running a analysis pass
# analysis passes
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})
ori_mg, _ = add_software_metadata_analysis_pass(ori_mg, None)

# report graph analysis, there's a print out
_ = report_graph_analysis_pass(ori_mg)

print("finish the first pass")

#########################################################################################################

print("node after init metadata analysis pass")
for node in ori_mg.fx_graph.nodes:
    print(node)

#########################################################################################################
### run transform pass, quantize
# args
pass_args_1 = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 16,
            "data_in_frac_width": 8,
            # weight
            "weight_width": 16,
            "weight_frac_width": 8,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

mg = ori_mg
# mg, _ = init_metadata_analysis_pass(mg, None)
# mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

mg, _ = quantize_transform_pass(ori_mg, pass_args_1)
print("report transform pass")
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})
print("show the diff")
summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")

print("print out the difference between the original graph and the quantized graph")

#########################################################################################################
# traverse the graph and print out the nodes
print("node after quantize transform pass")
for node in mg.fx_graph.nodes:
    print(node)


#########################################################################################################
#########################################################################################################

batch_size2 = 8
model_name2 = "jsc-tiny"
dataset_name = "jsc"

# Load model, dataset, and data module
data_module2 = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size2,
    model_name=model_name2,
    num_workers=0,
)
data_module2.prepare_data()
data_module2.setup()

# 📝️ change this CHECKPOINT_PATH to the one you trained in Lab1
CHECKPOINT_PATH2 = "/home/xinyi/ADL/mase_xinyi/mase_output/jsc-tiny_classification_jsc_2024-01-24/software/training_ckpts/best.ckpt"
model_info2 = get_model_info(model_name2)
model2 = get_model(
    model_name2,
    task="cls",
    dataset_info=data_module2.dataset_info,
    pretrained=False)

model2 = load_model(load_name=CHECKPOINT_PATH2, load_type="pl", model=model2)


#########################################################################################################

### get dummy data in
# dummy data is a dictionary that contains the input data, this input data is a dictionary that contains the input tensor
# get the input generator

# a demonstration of how to feed an input value to the model
dummy_in2 = next(iter(input_generator))
_ = model2(**dummy_in2)

#########################################################################################################
### generate a mase model and apply passes
# generate a mase model
# this is original mase model

new_mg = MaseGraph(model=model2)

# analysis passes
new_mg, _ = init_metadata_analysis_pass(new_mg, None)
new_mg, _ = add_common_metadata_analysis_pass(new_mg, {"dummy_in": dummy_in})
new_mg, _ = add_software_metadata_analysis_pass(new_mg, None)

# report graph analysis, there's a print out
_ = report_graph_analysis_pass(new_mg)

print("finish the first pass")

new_mg, _ = report_node_meta_param_analysis_pass(new_mg, {"which": ("software",)})
print("show the diff")
summarize_quantization_analysis_pass(new_mg, ori_mg, save_dir="quantize_summary")