import toml
import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
import time
import torch
import torch.nn as nn
# from chop.ir.graph.utils import get_parent_name
from torchmetrics.classification import MulticlassAccuracy

from chop.passes.graph.utils import get_node_actual_target

# figure out the correct path
machop_path = Path(".").resolve().parent /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
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
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model
import copy
from chop.actions.train import train

config_path = '/home/xinyi/mase_xinyi/machop/configs/examples/lab4_jsc.toml'

batch_size = 128
model_name = "jsc-three-linear-layers"
dataset_name = "jsc"

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=19,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

with open(config_path, 'r') as config_file:
    config = toml.load(config_file)

model_name = config.get('model', 'default_model')
dataset_name = config.get('dataset', 'default_dataset')
task_type = config.get('task', 'cls')
max_epochs = config.get('max_epochs', 1)
batch_size = config.get('batch_size', 128)
learning_rate = config.get('learning_rate', 1e-2)

model_info = model_info
data_module = data_module
dataset_info = get_dataset_info(dataset_name)

train(
    model=model_info,
    model_info=model_info,
    data_module=data_module,
    dataset_info=dataset_info,
    task=task_type,
    optimizer="adam",
    learning_rate=learning_rate,
    weight_decay=config.get('weight_decay', 1e-3),
    plt_trainer_args={
        "max_epochs": 1,
    },
    auto_requeue=False,
    save_path=None,
    visualizer=None,
    load_name=None,
    load_type=None,
)
