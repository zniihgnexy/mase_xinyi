## Lab 3 for Advanced Deep Learning Systems (ADLS)

### 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

#### for counting model size: Using model size as a quality metric, i defined a function as follows:
```python
def get_model_size(mg):
    model_size = 0
    for param in mg.model.parameters():
        print('parameters of element size', param.nelement(), param.element_size())
        model_size += param.nelement() * param.element_size()
    return model_size

```
#### for counting FLOPs: Using FLOPs as a quality metric, i defined calculations separately for different layer types as follows:
```python
from chop.passes.graph.analysis.flop_estimator.calc_modules import calculate_modules

store_flops_calculation = {}
# import pdb; pdb.set_trace()
for node in mg.fx_graph.nodes:
    try:
        in_data = (node.meta['mase'].parameters['common']['args']['data_in_0']['value'],)
    except KeyError:
        in_data = (None,)
    out_data = (node.meta['mase'].parameters['common']['results']['data_out_0']['value'],)

    module = get_node_actual_target(node)
    if isinstance(module, torch.nn.Module):
        current_flops = calculate_modules(module, in_data, out_data)
        store_flops_calculation[module] = current_flops
        
print("srote_data", store_flops_calculation)
```

in the calculation here, i used the chop library to calculate the FLOPs. the calculation of different layers are as follows:

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

#### for calculating latency: i defined the latency by using time module in python, and the latency is calculated as follows:
```python
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
```

I simply used the time.time() function to calculate the time difference between the start and end of the model running, and then added them together to get the total latency.

so my whole calculation percedure is as follows:

```python
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
        preds = new_mg.model(xs)
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
    recorded_accs.append(acc_avg)
    
    # model size
    model_size = get_model_size(mg)
    # print("model size: ", model_size)
    recorded_size.append(model_size)
    # print(model.relu.weight)
    recorded_latencies.append(latency)
    
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

print('recorded_accs: ', recorded_accs)
print('recorded_latencies: ', recorded_latencies)
print('recorded_model_sizes: ', recorded_model_sizes)
print('recorded_flops: ', recorded_flops)
```
in my functions, i used my self-defined functions to calculate the model size and FLOPs, and the results are as follows:

(1) model size:
(2) FLOPs:
srote_data {BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): {'total_parameters': 32, 'computations': 512, 'backward_computations': 512, 'input_buffer_size': 128, 'output_buffer_size': 128}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 128, 'backward_computations': 128, 'input_buffer_size': 128, 'output_buffer_size': 128}, Linear(in_features=16, out_features=5, bias=True): {'total_parameters': 80, 'computations': 640.0, 'backward_computations': 1280.0, 'input_buffer_size': 128, 'output_buffer_size': 40}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 40, 'backward_computations': 40, 'input_buffer_size': 40, 'output_buffer_size': 40}}

from the results above, the "computations" is the FLOPs, and the "total_parameters" is the model size. in order to calculate the full size of the model ,we need to add these numbers together, and the results are as follows:

model size: 112
FLOPs: 1280.0

(3) latency:
recorded_acc [tensor(0.2048), tensor(0.2500), tensor(0.2714), tensor(0.1655), tensor(0.1429), tensor(0.2839), tensor(0.2214), tensor(0.2500), tensor(0.2345), tensor(0.2190), tensor(0.3659), tensor(0.1821), tensor(0.2708), tensor(0.3125), tensor(0.2071), tensor(0.2075)]
recorded_latency [0.003651857376098633, 0.002660512924194336, 0.0023567676544189453, 0.002538442611694336, 0.004087686538696289, 0.0029108524322509766, 0.0020241737365722656, 0.0019147396087646484, 0.0019469261169433594, 0.0021681785583496094, 0.0025932788848876953, 0.002954721450805664, 0.002792835235595703, 0.0025196075439453125, 0.002095460891723633, 0.0026934146881103516]

### 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

### 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

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

in this part, i change the search strategies (sampler) to brute-force, and the results are as follows:

(add pictures here)

### 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.

compare these two methods, i only need to change the sampler in the .toml file:
    
```python
# when using brute force
sampler = "brute-force"
# when using tpe
sampler = "tpe"
```

i found that the brute-force search is much slower than the TPE based search, and the results are as follows:

(add pictures here)

this means, when using different search methods, the sample effeciency is different.
When using the brute-force search, the sample effeciency is much lower than the TPE based search, and the reason is that the brute-force search is a method that enumerates all possible combinations of the search space, and then evaluates each combination, and then selects the best combination. This method is very time-consuming, and the sample effeciency is very low. 

However, the TPE based search is a method that uses the Bayesian optimization algorithm to find the best combination, which is much faster than the brute-force search, and the sample effeciency is much higher than the brute-force search.