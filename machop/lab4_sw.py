import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

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
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model
from chop.passes.graph.utils import get_node_actual_target

from copy import deepcopy

set_logging_verbosity("info")

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
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

from torch import nn
from chop.passes.graph.utils import get_parent_name

# define a new model
# class JSC_Three_Linear_Layers(nn.Module):
#     def __init__(self):
#         super(JSC_Three_Linear_Layers, self).__init__()
#         self.seq_blocks = nn.Sequential(
#             nn.BatchNorm1d(16),  # 0
#             nn.ReLU(16),  # 1
#             nn.Linear(16, 16),  # linear  2
#             nn.Linear(16, 16),  # linear  3
#             nn.Linear(16, 5),   # linear  4
#             nn.ReLU(5),  # 5
#         )
#
#     def forward(self, x):
#         return self.seq_blocks(x)

# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)


model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)
# print("original one")
# _ = report_node_meta_param_analysis_pass(mg)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(ori_graph, pass_args=None):
    # return a copy of origin graph, otherwise the number of channels will keep growing
    graph = deepcopy(ori_graph)
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    for node in graph.fx_graph.nodes:
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        if isinstance(get_node_actual_target(node), nn.Linear):
            name = config.get("name", None)
            if name is not None:
                ori_module = graph.modules[node.target]
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier"]
                elif name == "both":
                    in_features = in_features * main_config.get(config['prev_link'], default)['config']["channel_multiplier"]
                    out_features = out_features * config["channel_multiplier"]
                elif name == "input_only":
                    in_features = in_features * main_config.get(config['prev_link'], default)['config']["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
        elif isinstance(get_node_actual_target(node), nn.BatchNorm1d):
            prev_link = config.get("prev_link", None)
            if prev_link is not None:
                ori_module = graph.modules[node.target]
                num_features, eps, momentum, affine = ori_module.num_features, ori_module.eps, ori_module.momentum, ori_module.affine
                num_features = num_features * main_config.get(prev_link, default)['config']["channel_multiplier"]
                new_module = nn.BatchNorm1d(num_features, eps, momentum, affine)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
    return graph, {}



pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "prev_link": "seq_blocks_2",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "prev_link": "seq_blocks_4",
        "channel_multiplier": 2,
        }
    },
}

# this performs the architecture transformation based on the config
# mg, _ = redefine_linear_transform_pass(
#     graph=mg, pass_args={"config": pass_config})
#
# for node in mg.fx_graph.nodes:
#     if node.meta["mase"].module is not None:
#         print(node.name, ": ",node.meta["mase"].module)

from torchmetrics.classification import MulticlassAccuracy
import time
import matplotlib.pyplot as plt
import copy

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

# build a search space
channel_multipliers = [1,2,3,4,5,6]
search_spaces = []
for cm_config in channel_multipliers:
    pass_config['seq_blocks_2']['config']['channel_multiplier'] = cm_config
    pass_config['seq_blocks_4']['config']['channel_multiplier'] = cm_config
    # pass_config['seq_blocks_6']['config']['channel_multiplier'] = cm_config
    search_spaces.append(copy.deepcopy(pass_config))

recorded_accs = []
recorded_latencies = []
recorded_model_sizes = []
for i, config in enumerate(search_spaces):
    new_mg, _ = redefine_linear_transform_pass(
        ori_graph=mg, pass_args={"config": config})
    print("new one")
    # _ = report_node_meta_param_analysis_pass(new_mg)
    print(config)
    j = 0

    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    latency, model_size = 0, 0
    flag = True
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start = time.time()
        preds = new_mg.model(xs)
        end = time.time()
        loss = nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1

        latency += end - start
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    print('accs: ', accs)
    print('losses: ', losses)
    recorded_accs.append(acc_avg)
    recorded_latencies.append(latency)
    recorded_model_sizes.append(model_size)

# print('recorded_accs: ', recorded_accs)
# print('recorded_latencies: ', recorded_latencies)
# print('recorded_model_sizes: ', recorded_model_sizes)
