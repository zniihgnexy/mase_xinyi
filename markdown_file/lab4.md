### Lab 4 (Software Stream) for Advanced Deep Learning Systems (ADLS)

###### 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.

By modify the redefine function, we can double the input and output feature numbers by using a multiplier number 2 for calculation. I wrote a full configuration filefor the redefine function to read. The code is as follows:

```python
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
"seq_blocks_3": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}
```

After that i print out the configuration i have now for doubling the layers and here's the result i got:

![](lab4pic\doubleQ1.png)

###### 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

Using the grid search means to traverse the search space and find the best accuracy. In this case, i defined a loop to read the configuration and build the full search space based on this. The code and results are as follows:



###### 3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following: Can you then design a search so that it can reach a network that can have this kind of structure?

In this case, in order to change the input layer scales by 2 and output layer scaled by 4, we need to let this linear layer number two read the previous layer's configuration. The network of linear layers can only be correct if the input feature number and the previous output feature number are the same.

So, to achieve the input layer feature number adjustment, i added a new configuration class called parent_channel. This config can read the block name of the parent layer, then i can read the corresponding configuration of the parent layer and get the multiplier i need.

The code i wrote to get the parent multiplier is as follows:

```python
pass_config = {
"by": "name",
"default": {"config": {
    "name": None,
    "channel_multiplier": 1,
    }
},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        "parent_block_name": "seq_blocks_1"
        }
    },
"seq_blocks_3": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        "parent_block_name": "seq_blocks_2"
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        "parent_block_name": "seq_blocks_3"
        }
    },
}

```

After i have the previous multiplier, i also changed my redefine function of building a new model of the network. In this new network, i change the calculation of linear layer as follows:

```python
if name == "output_only":
out_features = out_features * config["channel_multiplier"]
in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]

elif name == "both":
in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
out_features = out_features * config["channel_multiplier"]
# in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
# out_features = out_features * config["channel_multiplier"]

elif name == "input_only":
in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]

new_module = instantiate_linear(in_features, out_features, bias)
parent_name, name = get_parent_name(node.target)
setattr(graph.modules[parent_name], name, new_module)
```

After my modification, i can successfully run this code and change the third linear layer's input scaled by 2 and output scaled by 4.

![](lab4pic\doubleQ3.png)

In this picture, we can get the initial values of the layers of: 

[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]

Then I added the output layer's parameters and change the input scaled by 2 output scaled by 4. The results are as follows:

[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=32, bias=True), Linear(in_features=32, out_features=32, bias=True), Linear(in_features=32, out_features=5, bias=True), ReLU(inplace=True)]

```python
# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 32),  # output scaled by 2
            nn.ReLU(32),  # scaled by 2
            nn.Linear(32, 64),  # input scaled by 2 but output scaled by 4
            nn.ReLU(64),  # scaled by 4
            nn.Linear(64, 5),  # scaled by 4
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

###### 4. Integrate the search to the chop flow, so we can run it from the command line.

To integrate my search to the chop flow, i opened a new folder parallel with the quantize search space folder, in order to change my search space to my own defined space of multiplier number of the layers. As shown in the graph below is my defined search space .toml file:

```toml
[search.search_space]
name = "graph/quantize/channel_size_modifier"

#[search.search_space.setup]
#by = 'name'

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]
channel_multiplier = [1]

[search.search_space.seed.seq_blocks_2.config]
name = ["output_only"]
channel_multiplier = [1, 2]
parent_block_name = ["seq_blocks_1"]

[search.search_space.seed.seq_blocks_3.config]
name = ["both"]
parent_block_name = ["seq_blocks_2"]
channel_multiplier = [1, 2, 3]

[search.search_space.seed.seq_blocks_4.config]
name = ["input_only"]
parent_block_name = ["seq_blocks_3"]
channel_multiplier = [1, 2]

```

For i'm using the newest three layers version of network, i need to consider the connection between different laters. In this case, the input of the next linear layer has to be the same as the previous output layer. To achieve this, apart from modify the changes of multiplier channel settings, i also added a parent level multiplier configuration as shown in previous code bar. Using this, i can read the configuration of the parent node's multiplier. The input of the latter layer will be the original output layer multiplies with the multiplier of the parent layer. And this keeps the network parameters calculatable.

Then is how to define the new generated graph and build the search procedure in order to use optuna search strategy. I inherited SearchBase  class and using a redefine function here to build the new graph. The redefine function is as below:

```python
def redefine_transform_pass(graph, pass_args=None):
    # import pdb; pdb.set_trace()
    # graph = copy.deepcopy(ori_graph)
    # graph = torch.nn.DataParallel(graph)
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    
    # import pdb; pdb.set_trace()
    
    
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        # print("this iteration's config:")
        print(config)
        # if isinstance(get_node_actual_target(node), nn.Linear):
        #     parent_config = main_config.get(config['parent_block_name'], default)['config']
        name = config.get("name", None)
        if isinstance(get_node_actual_target(node), nn.Linear):
            if name is not None:
                ori_module = graph.modules[node.target]
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias
                
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier"]
                    in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
                elif name == "both":
                    in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
                    out_features = out_features * config["channel_multiplier"]
                # in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
                # out_features = out_features * config["channel_multiplier"]
                elif name == "input_only":
                    in_features = in_features * main_config.get(config['parent_block_name'], default)['config']["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
        elif isinstance(get_node_actual_target(node), ReLU):
            if name is not None:
                ori_module = graph.modules[node.target]
                inplace = ori_module.inplace
                new_module = instantiate_relu(inplace)
                setattr(graph.modules[node.target], "inplace", inplace)
        elif isinstance(get_node_actual_target(node), nn.BatchNorm1d):
            # import pdb; pdb.set_trace()
            # input_channel_number = config.get("input_channel_number", 16)
            if name is not None:
                ori_module = graph.modules[node.target]
                num_features = ori_module.num_features
                eps = ori_module.eps
                momentum = ori_module.momentum
                affine = ori_module.affine
                track_running_stats = ori_module.track_running_stats
                new_module = instantiate_batchnorm(num_features, eps, momentum, affine, track_running_stats)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
    return graph, {}
```

In this function, i build new nodes based on the type of the layers. Different layers have different calculation function. For example, the linear layer has input and output features, these features defined thid module's calculation. So when rebuild the linear layer, we have to give them the correct numbers of input output features. Also, ReLU layer only has a place holder, so in order to rebuild the ReLU layer, we only need to give the correct placeholder value, which is a bool value.

For i'm using the jsc-three-layer model, i only wrote three types of layers definition for rebuilding the graph. And the definition functions are as follows:

```python
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
```

The result of searching is as follows:
![](lab4pic\search.png)

![](lab4pic\search-2.png)

![](lab4pic\search-3.png)

![](lab4pic\search-4.png)

![](lab4pic\search-5.png)

From the searching results above we can see the best configuration here is number 11 and the setting is channel multiplier {linear 1: 1, linear 2: 2, linear 3: 1} and the loss value is 1.623, accuracy is 0.21 for software metrics calculation.

Next i tried other multipliers numeber settings, and the results are as follows:
