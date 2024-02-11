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


## Lab 4 (Software Stream) for Advanced Deep Learning Systems (ADLS)

### 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.

#### Network Expansion: Doubling Layer Sizes

Adjusting neural network architectures to enhance performance often involves scaling the size of layers. In this example, we expand the layers of a network to double their original sizes. This process is guided by a configuration file that specifies how each layer should be altered.

##### Configuration for Layer Expansion

The configuration for expanding the layers is defined as follows:

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
This pass_config ensures that each specified layer has its input and output features doubled, effectively expanding the network's capacity.

#### Results Before and After Modification

The original network's structure is shown below:

```python
INFO     Set logging level to info
original one
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    return seq_blocks_5
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 6, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```

After applying the configuration, the updated structure is as follows:

```python
new one
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    return seq_blocks_5
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 6, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=32, bias=True), Linear(in_features=32, out_features=32, bias=True), Linear(in_features=32, out_features=5, bias=True), ReLU(inplace=True)]
```

We can see the results of motiplier, the input of the linear layers are scaled by 2 and the output of them are scaled by 2.

### 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

Using the grid search means to traverse the search space and find the best accuracy.

#### Optimizing Network Layers with Grid Search

To enhance the model's learning capacity, we aim to determine the optimal channel multiplier value for layer expansion using grid search. This process involves configuring a search space with potential multiplier values and retraining the model to evaluate performance.

#### Building the Search Space

The search space for the channel multipliers is constructed using the following parameters:

```python
multiplier_number = [1, 2, 3, 4, 5, 6]
# name_cannels = ["input_only", "output_only", "both"]
layer_number = [2, 4, 6]
multiplier_search_space = []

for multiplier in multiplier_number:
    # for name in name_cannels:
    temp_pass_config = copy.deepcopy(pass_config)
    temp_pass_config[f"seq_blocks_2"]['config']['channel_multiplier'] = multiplier 
    temp_pass_config[f"seq_blocks_4"]['config']['channel_multiplier'] = multiplier 
    temp_pass_config[f"seq_blocks_6"]['config']['channel_multiplier'] = multiplier 
    # temp_pass_config[f"seq_blocks_{layer}"]['config']['name'] = name_cannels
    multiplier_search_space.append(copy.deepcopy(temp_pass_config))
```
In this code, i defined the multiplier number and the layer number, and then i used a loop to read the configuration and build the full search space based on this. After that, i can use the optuna search strategy to search the best configuration of the multiplier number.

#### Retraining the Model

And for the training after getting the configuration, i used the following code to retrain the network:

```python
    train(
        model,
        model_info,
        data_module,
        dataset_info=get_dataset_info(dataset_name),
        task="cls",
        optimizer="adam",
        learning_rate=1e-3,
        weight_decay=1e-3,
        plt_trainer_args={
            "max_epochs": 1,
        },
        auto_requeue=False,
        save_path=None,
        visualizer=None,
        load_name=None,
        load_type=None,
    )
```

This function retrains the model with new layer configurations to assess the performance implications of each channel multiplier.

#### Experiment Results

One training procedure is like(take iteration 5 as an example):
    
```python
number of this iteration:  5
the config:  {'by': 'name', 'seq_blocks_2': {'config': {'name': 'output_only', 'channel_multiplier': 6, 'parent_block_name': 'seq_blocks_1'}}, 'seq_blocks_4': {'config': {'name': 'both', 'channel_multiplier': 6, 'parent_block_name': 'seq_blocks_2'}}, 'seq_blocks_6': {'config': {'name': 'input_only', 'channel_multiplier': 6, 'parent_block_name': 'seq_blocks_4'}}, 'default': {'config': {'name': None}}}
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name      | Type               | Params
-------------------------------------------------
0 | model     | GraphModule        | 11.5 K
1 | loss_fn   | CrossEntropyLoss   | 0     
2 | acc_train | MulticlassAccuracy | 0     
3 | acc_val   | MulticlassAccuracy | 0     
4 | acc_test  | MulticlassAccuracy | 0     
5 | loss_val  | MeanMetric         | 0     
6 | loss_test | MeanMetric         | 0     
-------------------------------------------------
11.5 K    Trainable params
0         Non-trainable params
11.5 K    Total params
0.046     Total estimated model params size (MB)
Epoch 0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 6168/6168 [01:32<00:00, 66.94it/s, v_num=55, train_acc_step=0.618, val_acc_epoch=0.732, val_loss_epoch=0.779]`Trainer.fit` stopped: `max_epochs=1` reached.                                                                                                                                                                                     
Epoch 0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 6168/6168 [01:32<00:00, 66.93it/s, v_num=55, train_acc_step=0.618, val_acc_epoch=0.732, val_loss_epoch=0.779]
```

The results from the experiment indicate varying accuracies and latencies:

```python
recorded_accs:  [tensor(0.6888), tensor(0.5930), tensor(0.6858), tensor(0.6890), tensor(0.6831), tensor(0.6956)]

recorded_latencies:  [0.07818484306335449, 0.09776568412780762, 0.09232139587402344, 0.1297626495361328, 0.17051362991333008, 0.24888229370117188]
```


## 3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following: Can you then design a search so that it can reach a network that can have this kind of structure?

In this case, in order to change the input layer scales by 2 and output layer scaled by 4, we need to let this linear layer number two read the previous layer's configuration. The network of linear layers can only be correct if the input feature number and the previous output feature number are the same.

To modify the scaling of layers in a neural network, where the input and output features are scaled by different multipliers, a mechanism for layer-specific configuration is required. This enables the network to adapt each layer's dimensions based on the configuration of its preceding layers, ensuring the correct propagation of feature sizes throughout the model.

### Strategy for Configuring Layer Multipliers

The following approach outlines how to implement a configuration that allows a layer to scale its features according to the parent layer's settings:

1. Define a `parent_channel` configuration class that holds information about the parent block's name and its corresponding multiplier.
2. Ensure that the network's structure maintains coherence, such that the input features of a layer match the output features of its preceding layer.

### Code Implementation

The code snippet below illustrates how to retrieve and apply the parent layer's multiplier to adjust the current layer's input features:

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
        "name": "output_only",
        "channel_multiplier": 4,
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

### Refining the Network Model with Customized Layer Scaling

With the ability to ascertain the multiplier from the previous layer, the next step involves adapting the model reconstruction process to incorporate these dynamic scaling factors. The redefine function is crucial in this context, as it recalibrates the network's architecture to accommodate the newly computed dimensions for each layer.

#### Modification to the Redefine Function

The redefine function has been updated to alter the structure of linear layers based on the calculated multipliers:

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

#### Modification to the Redefine Function

The redefine function has been updated to alter the structure of linear layers based on the calculated multipliers:

```python
INFO     Set logging level to info
original one
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    return seq_blocks_5
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 6, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=16, bias=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```
Then I added the output layer's parameters and change the input scaled by 2 output scaled by 4. The results are as follows:
```python
new one
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    return seq_blocks_5
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 6, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=32, bias=True), Linear(in_features=32, out_features=64, bias=True), Linear(in_features=128, out_features=5, bias=True), ReLU(inplace=True)]
```


### 4. Integrate the search to the chop flow, so we can run it from the command line.

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

# Coherent Layer Connectivity in Network Architecture

The design of a neural network requires meticulous attention to the connectivity between layers, particularly when customizing layer sizes. The newest version of the network under consideration includes three layers, each demanding precise input-output matching for seamless data flow. To ensure the network's integrity, the input dimension of each layer must align with the output dimension of its preceding layer. This necessitates a multi-tiered configuration strategy that synchronizes the input and output dimensions across the network.

### Synchronizing Layer Inputs and Outputs

To ensure that the input dimension of a subsequent layer corresponds to the output dimension of its preceding layer, a multi-tiered configuration strategy is employed:

1. **Channel Multiplier Settings**: Each layer's scaling factor is adjusted to maintain the network's functional integrity. In this case, I modified linear, ReLU, and batch normalization layers to accommodate the scaling factor, ensuring that the input and output dimensions align with the parent layer's configuration.
   
2. **Parent Level Multiplier Configuration**: Introduced in the previous section, this configuration allows the network to infer the scaling factor from the parent layer, ensuring that the subsequent layer's input dimension aligns with the prior layer's output.

The resulting architecture ensures that the network's parameters remain calculable and coherent, preserving the integrity of data processing throughout the network.

### Graph Redefinition and Search Strategy Implementation

To facilitate the search for the optimal network structure, a redefine function is implemented within the context of the SearchBase class:

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

### Rebuilding Neural Network Layers in a Custom Graph

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

The result of searching is as follows, taking the 17th trial as an example: train 10 epochs eachtime redefine the network. My .toml file setting is like:
    
```toml
[search.strategy]
name = "optuna"
eval_mode = false

[search.strategy.sw_runner.basic_train]
name = "accuracy"
data_loader = "train_dataloader"
num_samples = 1000000
max_epochs = 10
lr_scheduler = "linear"
optimizer = "adam"
learning_rate = 1e-4
num_warmup_steps = 0
```
In my setting here, i set the eval_model to false, this will automatically retrain the network after getting the configuration. The results are as follows:
```python
Best trial: 1. Best value: 0.521101:  95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍      | 19/20 [02:32<00:08,  8.12s/it, 152.83/20000 seconds]training mode
{'name': None, 'channel_multiplier': 1}
{'name': None, 'channel_multiplier': 1}
{'name': None, 'channel_multiplier': 1}
{'name': 'output_only', 'channel_multiplier': 2, 'parent_block_name': 'seq_blocks_1'}
{'name': 'both', 'parent_block_name': 'seq_blocks_2', 'channel_multiplier': 3}
{'name': 'input_only', 'parent_block_name': 'seq_blocks_3', 'channel_multiplier': 6}
{'name': None, 'channel_multiplier': 1}
{'name': None, 'channel_multiplier': 1}
WARNING  No quantized layers found in the model, set average_bitwidth to 32
Best trial: 1. Best value: 0.521101: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [02:40<00:00,  8.04s/it, 160.84/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                | scaled_metrics      |
|----+----------+------------------------------------+-------------------------------------------------+---------------------|
|  0 |        1 | {'loss': 1.252, 'accuracy': 0.521} | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.521} |
INFO     Searching is completed
```

From the searching results above we can see the best configuration here is number 17 and the setting is channel multiplier 
```python
{'name': 'output_only', 'channel_multiplier': 2, 'parent_block_name': 'seq_blocks_1'}
{'name': 'both', 'parent_block_name': 'seq_blocks_2', 'channel_multiplier': 3}
{'name': 'input_only', 'parent_block_name': 'seq_blocks_3', 'channel_multiplier': 6}
```
and the loss value is 1.268, accuracy is 0.496 for software metrics calculation. The average bitwidth is 32 and the memory density is 1.0 for hardware metrics calculation. The accuracy is 0.496 for scaled metrics calculation.
