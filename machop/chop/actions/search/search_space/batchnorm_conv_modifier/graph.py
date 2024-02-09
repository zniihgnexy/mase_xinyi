# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
from torch import nn
from chop.passes.graph.utils import get_parent_name
from ..base import SearchSpaceBase
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict

DEFAULT_VGG_MULTIPLIER_CONFIG = {
    "config": {
        "name": None,
    }
}


class BatchNormConvModifierZXY(SearchSpaceBase):
    """
    Post-Training quantization search space for mase graph.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_VGG_MULTIPLIER_CONFIG
        
        # quantize the model by type or name
        # assert (
        #     "by" in self.config["setup"]
        # ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"


    def rebuild_model(self, sampled_config, is_eval_mode: bool = False):
        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        if self.mg is None:
            assert self.model_info.is_fx_traceable, "Model must be fx traceable"
            mg = MaseGraph(self.model)
            mg, _ = init_metadata_analysis_pass(mg, None)
            mg, _ = add_common_metadata_analysis_pass(
                mg, {"dummy_in": self.dummy_input}
            )
            self.mg = mg
        if sampled_config is not None:
            # ori_mg = mg.detach()
            mg, _ = redefine_transform_pass(mg, {"config": sampled_config})
            # model = mg.model
            # data_module = self.data_module
            # dataset_info = self.dataset_info
        mg.model.to(self.accelerator)
        # mg.model.train()      
        # train(
        # model,
        # model_info = self.model_info,
        # data_module = data_module,
        # dataset_info = dataset_info,
        # task="cls",
        # optimizer="adam",
        # learning_rate=1e-3,
        # weight_decay=1e-3,
        # plt_trainer_args={
        #     "max_epochs": 1,
        # },
        # auto_requeue=False,
        # save_path=None,
        # visualizer=None,
        # load_name=None,
        # load_type=None,
        # )
        
        return mg

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the mase graph (only quantizeable ops)
        """
        # Build a mapping from node name to mase_type and mase_op.
        
        # import pdb; pdb.set_trace()
        
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)
        choices = {}
        seed = self.config["seed"]
        
        # import pdb; pdb.set_trace()
        
        for node in mase_graph.fx_graph.nodes:
            if node.name in seed:
                choices[node.name] = deepcopy(seed[node.name])
            else:
                choices[node.name] = deepcopy(seed["default"])
                # choices[node.name] = DEFAULT_CHANNEL_SIEZ_MODIFIER_CONFIG["config"]
        # import pdb; pdb.set_trace()
        # flatten the choices and choice_lengths
        # self.choices_flattened = {}
        flatten_dict(choices, flattened=self.choices_flattened)
        print(self.choices_flattened)
        
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


# following functions are copied form lab4_software
def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)


def redefine_transform_pass(graph, pass_args=None):
    # return a copy of origin graph, otherwise the number of channels will keep growing
    # graph = deepcopy(ori_graph)
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    for node in graph.fx_graph.nodes:
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        new_module = None
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
                    in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
                    out_features = out_features * config["channel_multiplier"]
                elif name == "input_only":
                    in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
        elif isinstance(get_node_actual_target(node), nn.Conv2d):
            name = config.get("name", None)
            if name is not None:
                ori_module = graph.modules[node.target]
                in_channels = ori_module.in_channels
                out_channels = ori_module.out_channels
                bias = ori_module.bias
                if name == "output_only":
                    out_channels = out_channels * config["channel_multiplier"]
                elif name == "both":
                    in_channels = in_channels * main_config.get(config['parent_block_name'], default)['config'][
                        "channel_multiplier"]
                    out_channels = out_channels * config["channel_multiplier"]
                elif name == "input_only":
                    in_channels = in_channels * main_config.get(config['parent_block_name'], default)['config'][
                        "channel_multiplier"]
                new_module = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=ori_module.kernel_size, stride=ori_module.stride,
                                       padding=ori_module.padding, dilation=ori_module.dilation,
                                       groups=ori_module.groups, bias=ori_module.bias is not None,
                                       padding_mode=ori_module.padding_mode)
        elif isinstance(get_node_actual_target(node), nn.BatchNorm1d):
            prev_link = config.get("parent_block_name", None)
            if prev_link is not None:
                ori_module = graph.modules[node.target]
                num_features, eps, momentum, affine = ori_module.num_features, ori_module.eps, ori_module.momentum, ori_module.affine
                num_features = num_features * main_config.get(prev_link, default)['config']["channel_multiplier"]
                new_module = nn.BatchNorm1d(num_features, eps, momentum, affine)
        elif isinstance(get_node_actual_target(node), nn.BatchNorm2d):
            prev_link = config.get("parent_block_name", None)
            if prev_link is not None:
                ori_module = graph.modules[node.target]
                num_features, eps, momentum, affine = ori_module.num_features, ori_module.eps, ori_module.momentum, ori_module.affine
                num_features = num_features * main_config.get(prev_link, default)['config']["channel_multiplier"]
                new_module = nn.BatchNorm2d(num_features, eps, momentum, affine)

        if new_module is not None:
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)

    return graph, {}