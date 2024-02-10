## Optional task of lab2: FLOPs and BitOPs

I use the funtion defined in the source code to calculate the FLOPs. About BitOPs, i simply calculate the number of bit calculation persuming all the data are multiplied between layers. The results are as below:

```python
store_flops_data {BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): {'total_parameters': 32, 'computations': 512, 'backward_computations': 512, 'input_buffer_size': 128, 'output_buffer_size': 128}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 128, 'backward_computations': 128, 'input_buffer_size': 128, 'output_buffer_size': 128}, Linear(in_features=16, out_features=5, bias=True): {'total_parameters': 80, 'computations': 640.0, 'backward_computations': 1280.0, 'input_buffer_size': 128, 'output_buffer_size': 40}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 40, 'backward_computations': 40, 'input_buffer_size': 40, 'output_buffer_size': 40}}
store_bitops_data {BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): {'computations': 512, 'bitops': 16384}, ReLU(inplace=True): {'computations': 128, 'bitops': 4096}, Linear(in_features=16, out_features=5, bias=True): {'computations': 640.0, 'bitops': 20480.0}, ReLU(inplace=True): {'computations': 40, 'bitops': 1280}}
flops 1320.0
bitops 42240.0

parameters of element size 16 4
parameters of element size 16 4
parameters of element size 80 4
parameters of element size 5 4
model size:  468
```

# Lab 3 for Advanced Deep Learning Systems (ADLS)

## 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

#### for counting model size: Using model size as a quality metric, i defined a function as follows:
```python
def get_model_size(mg):
    model_size = 0
    for param in mg.model.parameters():
        print('parameters of element size', param.nelement(), param.element_size())
        model_size += param.nelement() * param.element_size()
    return model_size

```
in the calculation here, i used the chop library to calculate the FLOPs, which is the calculate_modules function. the calculation of different layers are as follows:

(1) for the linear layer:
```python
batch = in_data[0].numel() / in_data[0].shape[-1]
computations = module.weight.numel() * batch
backward_computations = module.weight.numel() * batch
```
(2) for the relu layer:
```python
"computations": in_data[0].numel(),
"backward_computations": in_data[0].numel(),
```
(3) for the batch_norm1d layer:
```python
total_parameters = 2 * module.num_features
computations = 4 * in_data[0].numel()
backward_computations = 4 * in_data[0].numel()
```

these are the three layers i used here.

in my functions, i used my self-defined functions to calculate the model size and FLOPs, and the results are as follows:

(1) model size and FLOPs:
```python
store_flops_data {BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): {'total_parameters': 32, 'computations': 512, 'backward_computations': 512, 'input_buffer_size': 128, 'output_buffer_size': 128}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 128, 'backward_computations': 128, 'input_buffer_size': 128, 'output_buffer_size': 128}, Linear(in_features=16, out_features=5, bias=True): {'total_parameters': 80, 'computations': 640.0, 'backward_computations': 1280.0, 'input_buffer_size': 128, 'output_buffer_size': 40}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 40, 'backward_computations': 40, 'input_buffer_size': 40, 'output_buffer_size': 40}}
store_bitops_data {BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): {'computations': 512, 'bitops': 16384}, ReLU(inplace=True): {'computations': 128, 'bitops': 4096}, Linear(in_features=16, out_features=5, bias=True): {'computations': 640.0, 'bitops': 20480.0}, ReLU(inplace=True): {'computations': 40, 'bitops': 1280}}
flops 1320.0
bitops 42240.0
```

from the results above, the "computations" is the FLOPs, and the "total_parameters" is the model size. in order to calculate the full size of the model ,we need to add these numbers together, and the results are as follows:

model size: 468
FLOPs: 1320.0

## 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

### Combining Additional Quality Metrics with Accuracy and Loss

In model optimization, integrating various quality metrics beyond accuracy and loss can provide a more nuanced view of a model's performance. Latency, for instance, is an operational metric reflecting the model's response time or execution speed. Here’s an implementation of latency measurement, along with a discussion on how accuracy and loss can serve as congruent metrics in certain contexts.

### Latency Measurement

Latency is a critical metric, especially in real-time applications where response time is crucial. The following code measures the latency of a model by recording the time taken to make predictions:

```python
import time
import torch.nn as nn

accs = []
losses = []
latencies = []
num_batches = 100  # Define the number of batches to measure
j = 0

for inputs in data_module.train_dataloader():
    xs, ys = inputs
    start = time.time()
    preds = new_mg.model(xs)
    end = time.time()
    latency = end - start
    latencies.append(latency)
    
    loss = nn.functional.cross_entropy(preds, ys)
    losses.append(loss.item())
    
    acc = (preds.argmax(dim=1) == ys).float().mean()
    accs.append(acc.item())

    if j >= num_batches:
        break
    j += 1

# Calculate and print the average latency
average_latency = sum(latencies) / len(latencies)
print(f'Average Latency: {average_latency:.5f} seconds')
```

This snippet wraps the model's prediction step with time.time() calls to measure the processing time, contributing to the total latency. The average latency is then computed over a defined number of batches to provide a reliable estimate.

### Results and Interpretation
The initial results, showing accuracy and latency, were as follows:
```python
recorded_acc [tensor(0.1934)]
recorded_latency [0.002038717269897461]
```
### Accuracy and Loss as Congruent Metrics
Accuracy and loss are indeed often used interchangeably as quality metrics in classification tasks. This is particularly true when using a loss function like cross-entropy, which directly relates to classification accuracy. The cross-entropy loss is minimized when the predicted probability distribution aligns with the actual distribution, which simultaneously maximizes accuracy.
#### Cross-Entropy Loss Function

$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \text{Loss}^{(i)}
$$

$$
\text{Loss}^{(i)} = - \sum_{k=1}^{q} y_k^{(i)} \log(\hat{y}_k^{(i)})
$$

#### Multi-Class Accuracy

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} 1(y_i = \hat{y}_i)
$$

Both metrics rely on the concept of one-hot encoding for classification, where the prediction is considered correct if the highest probability is assigned to the true class. Thus, as the loss decreases (indicating better predictions), the accuracy inherently increases, demonstrating their interconnectedness.

#### Conclusion
Latency, when combined with accuracy and loss, can yield a holistic view of the model's operational performance. While accuracy and loss are intrinsically linked through the mechanics of the cross-entropy loss function in classification tasks, incorporating latency provides an additional dimension of quality, reflecting the practical deployment considerations of the model.



## 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

using the brute-force search, i defined the method using the optuna.py file already existed in the MASE repo, and the code is as follows:

```python
def sampler_map(self, name):
    match name.lower():
        case "random":
            sampler = optuna.samplers.RandomSampler()
        case "tpe":
            sampler = optuna.samplers.TPESampler()
        case "nsgaii":
            sampler = optuna.samplers.NSGAIISampler()
        case "nsgaiii":
            sampler = optuna.samplers.NSGAIIISampler()
        case "qmc":
            sampler = optuna.samplers.QMCSampler()
        case "brute_force":
            sampler = optuna.samplers.BruteForceSampler()
        case _:
            raise ValueError(f"Unknown sampler name: {name}")
    return sampler
```

in this function, i use the BruteforceSampler() function to define the sampler, referencing the optuna open-source library.

using this specific method by using the command line, i also modified the .toml file as follows:

```python
[search.strategy.setup]
n_jobs = 1
n_trials = 20
timeout = 20000
sampler = "brute-force"
sum_scaled_metrics = false # multi objective
```

Here i used "brute-force" sampler for strategy, so this file can directly use the sampler written in optuna.py file, which is BruteforceSampler() function.

#### 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.

compare these two methods, i only need to change the sampler in the .toml file:

```python
# when using brute force
sampler = "brute-force"
# when using tpe
sampler = "tpe"
```

i found that the brute-force search is much slower than the TPE based search, and the results are as follows:

This is the TPE strategy output:
```python
INFO     Initialising model 'jsc-toy'...
INFO     Initialising dataset 'jsc'...
INFO     Project will be created at /home/xinyi/mase_xinyi/mase_output/jsc-toy
INFO     Loaded pytorch lightning checkpoint from /home/xinyi/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt
INFO     Loaded model from /home/xinyi/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt.
INFO     Building search space...
INFO     Search started...

INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        0 | {'loss': 1.109, 'accuracy': 0.626} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.626, 'average_bitwidth': 0.4} |
INFO     Searching is completed
```

in this part, i change the search strategies (sampler) to brute-force, and the results are as follows:

This is the brute-force output:
```python
INFO     Initialising model 'jsc-toy'...
INFO     Initialising dataset 'jsc'...
INFO     Project will be created at /home/xinyi/mase_xinyi/mase_output/jsc-toy
INFO     Loaded pytorch lightning checkpoint from /home/xinyi/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt
INFO     Loaded model from /home/xinyi/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt.
INFO     Building search space...
INFO     Search started...

INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        0 | {'loss': 1.073, 'accuracy': 0.61}  | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.61, 'average_bitwidth': 1.6}  |
|  1 |        1 | {'loss': 1.141, 'accuracy': 0.588} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.588, 'average_bitwidth': 0.8} |
|  2 |       16 | {'loss': 1.141, 'accuracy': 0.58}  | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.58, 'average_bitwidth': 0.4}  |
INFO     Searching is completed
```
### Comparative Analysis of Search Methods: TPE vs. Brute-Force

The evaluation of different search methods reveals distinct variations in performance and efficiency. The two methods in question, Tree-structured Parzen Estimator (TPE) and brute-force, offer unique approaches to the search process in hyperparameter optimization.

#### Running Time Comparison

When comparing running times between the TPE method and brute-force:

- The TPE method shows minimal variance in execution time.
- The brute-force approach generally takes about one second longer.

While this may not appear significant, it highlights a trend towards greater efficiency with the TPE method.

#### Accuracy and Loss Metrics

Regarding the search outcomes, particularly within the context of software metrics:

- TPE achieves a higher accuracy and incurs a lower loss.
- These improvements suggest a more effective search strategy in identifying optimal parameters.

#### Sample Efficiency

The notion of sample efficiency is pivotal when contrasting these methods:

- **Brute-Force Search**: Characterized by its exhaustive nature, this method entails evaluating every possible combination within the search space. Despite its comprehensiveness, it is inherently time-intensive and exhibits low sample efficiency.

- **TPE-Based Search**: Utilizing Bayesian optimization, TPE methodically navigates the search space, focusing on regions with higher potential. This targeted approach not only accelerates the search but also significantly enhances sample efficiency.

#### Conclusion

The analysis clearly demonstrates the superior sample efficiency of TPE over brute-force search. By leveraging the Bayesian optimization framework, TPE efficiently converges to optimal solutions, thus saving time and computational resources. This efficiency is particularly evident in the improved accuracy and reduced loss metrics achieved using TPE, underscoring its effectiveness in hyperparameter tuning and model optimization.


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
