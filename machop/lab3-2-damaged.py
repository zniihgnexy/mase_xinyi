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


import torch
import torch.nn as nn
import torchmetrics
import numpy as np
from torchmetrics.classification import MulticlassAccuracy

from chop.actions.search.strategies.lab3_brute_force import BruteForceSearchStrategy

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

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

def parse_accelerator(accelerator: str):
    if accelerator == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif accelerator == "gpu":
        device = torch.device("cuda:0")
    elif accelerator == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported accelerator {accelerator}")
    return device

class DummyVisualizer:
    def __init__(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def show(self):
        pass

dummy_visualizer = DummyVisualizer()

for i, config in enumerate(search_spaces):
    search_strategy = BruteForceSearchStrategy(
        model_info=model_info,
        data_module=data_module,
        dataset_info=data_module.dataset_info,
        task="cls",
        config=config,
        accelerator=parse_accelerator('cpu'),
        save_dir=Path("./lab3/result"),
        visualizer=dummy_visualizer
    )

    recorded_accs, recorded_sizes = search_strategy.search(search_spaces)
    
    for i, (acc, size) in enumerate(zip(recorded_accs, recorded_sizes), 1):
        print(f"Configuration {i}:")
        print("Recorded Accuracy:", acc)
        print("Recorded Size:", size)
        
#########################################################################################################

# brute force
