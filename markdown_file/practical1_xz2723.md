# lab1 for Advanced Deep Learning Systems (ADLS)

## 1. What is the impact of varying batch sizes and why?
### Impact of Varying Batch Sizes on Model Training

The choice of batch size in neural network training significantly influences several key aspects of the learning process. Below is a concise discussion on the effects of varying batch sizes:

#### Training Efficiency
- **Time**: Smaller batches increase training duration due to more iterations required.
- **Memory Usage**: Larger batches require more memory, while smaller batches are more resource-efficient.

#### Model Performance
- **Accuracy and Regularization**: Smaller batches may improve generalization by acting as a form of regularization, though this doesn't guarantee higher accuracy.
- **Generalization**: Smaller batch sizes can lead to better model generalization by providing more varied data per iteration.
- **Convergence and Stability**: Smaller batches can both aid in escaping local minima due to noisier gradient estimates and introduce training instability.

#### Practical Considerations
- **Running Time**: Smaller batches often mean longer training times to achieve similar accuracy levels as larger batches.
- **Overfitting Prevention**: Smaller batches help in preventing overfitting by ensuring the model learns generalized features.

Choosing the right batch size is a balance between computational resource constraints, training time, and model quality. It's essential to experiment with different batch sizes to find the optimal setting for a specific model and dataset.


###### different batch sizes results

| epoch | batch_size | learning_rate | training loss | validation loss |
| :---: | :--------: | :-----------: | :-----------: | :-------------: |
|  50   |     64     |    0.0001     |    0.7826     |      0.877      |
|  50   |    128     |    0.0001     |    0.9912     |     0.8580      |
|  50   |    256     |    0.0001     |    0.9283     |      0.860      |
|  50   |    512     |    0.0001     |     1.070     |      1.054      |
|  50   |    1024    |    0.0001     |     1.052     |      1.086      |

From the above form we can see that, different batch sizes have an influence on learning time. the higher the batch size is, the shorter the training time will be. this is because the training procedure is calculated by batch size and the time this batches were sent to the training network.

## 2. What is the impact of varying maximum epoch number?

#### Impact of Varying Maximum Epoch Number on Model Training

Adjusting the maximum number of epochs in neural network training has direct implications on the model's learning dynamics. Here's a succinct overview of the potential impacts:

#### Training Efficiency
- **Training Time**: Increasing the maximum epoch number extends the training duration due to more iterations required to complete the training process.

#### Model Quality
- **Accuracy**: While more epochs provide more learning opportunities, it does not guarantee improved accuracy, as the effect varies by model and dataset.
- **Generalization**: A higher number of epochs increases the risk of overfitting, potentially harming the model's ability to generalize to unseen data.
- **Memory Usage**: Similar to batch sizes, more epochs mean more iterations, impacting memory usage and computational resources.

#### Practical Implications
- **Running Time and Overfitting**: An increased epoch count can significantly prolong training time and elevate the risk of overfitting, emphasizing the need for careful selection of epoch numbers based on specific training goals and data characteristics.

Choosing the optimal maximum epoch number is crucial for balancing training efficiency with model performance, highlighting the importance of experimentation and validation to identify the best configuration for a given scenario.


###### different max epoch experiences

| epoch | batch_size | learning_rate | training loss | validation loss |
| :---: | :--------: | :-----------: | :-----------: | :-------------: |
|  10   |    256     |    0.0001     |    0.9919     |     0.9922      |
|  50   |    256     |    0.0001     |    0.9283     |      0.859      |
|  100  |    256     |    0.0001     |     1.073     |     0.8435      |
|  150  |    256     |    0.0001     |    0.8844     |      0.84       |
|  200  |    256     |    0.0001     |    0.6919     |     0.8382      |

The table show the training loss and validation loss with respect to the change of epochs. When the epoch is small than 50, the training loss decreases with increasing epochs. When the epoch is larger than 150, the training loss is much smaller than validation loss, indicating an overfitting problem.

## 3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
### Learning Rate Impacts and Its Relationship with Batch Sizes

Understanding the impact of learning rate settings and their interaction with batch sizes is crucial for optimizing neural network training. Here's an exploration of how different learning rates affect model training and their relation to batch sizes.

#### Effects of Learning Rate Sizes

- **Large Learning Rates**: While they can accelerate convergence, large learning rates risk surpassing the optimal solution, potentially leading to lower model accuracy. This occurs because significant updates can cause the training process to overshoot the minimum point of the loss function.

- **Small Learning Rates**: Conversely, small learning rates ensure a more gradual approach towards the optimal point, favoring higher accuracy. However, this meticulous pace results in slower convergence, requiring more epochs to reach comparable levels of performance to those achieved with higher learning rates.

#### Interaction with Batch Sizes

- **Correlation with Batch Size**: The synergy between learning rates and batch sizes plays a pivotal role in model training. Smaller batch sizes, akin to a regularization method, necessitate lower learning rates to counteract the increased noise in gradient estimates. This combination helps in preventing overfitting, facilitating a more controlled and precise convergence towards the optimum.

Selecting an appropriate learning rate in conjunction with the right batch size is essential for achieving a balance between training efficiency and model accuracy. It involves careful experimentation to identify the optimal settings that prevent overfitting while ensuring timely convergence.


###### different learning rates

| epoch | batch_size | learning_rate | training loss | validation loss |
| :---: | :--------: | :-----------: | :-----------: | :-------------: |
|  50   |    256     |      0.1      |     1.229     |      1.197      |
|  50   |    256     |     0.001     |    0.9184     |      0.832      |
|  50   |    256     |    0.0001     |    0.9283     |      0.859      |
|  50   |    256     |    0.00001    |     1.214     |      1.094      |
|  50   |    256     |   0.000001    |     1.388     |      1.354      |

## 4. Implement a network that has in total around 10x more parameters than the toy network.
using the similiar structure as the toy network, I implemented a network that has in total around 10x more parameters. the structure is shown below:
```python
class Test(nn.Module):
    def __init__(self, info):
        super(Test, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 64),  # linear              # 2
            nn.BatchNorm1d(64),  # output_quant       # 3
            nn.ReLU(64),  # 4
            # 2nd LogicNets Layer
            nn.Linear(64, 128),  # 5
            nn.BatchNorm1d(128),  # 6
            nn.ReLU(128),  # 7
            # 3rd LogicNets Layer
            nn.Linear(128, 8),  # 8
            nn.BatchNorm1d(8),  # 9
            nn.ReLU(8),
            nn.Linear(8, 8),  # 5
            nn.BatchNorm1d(8),  # 6
            nn.ReLU(8),  # 7
            # 3rd LogicNets Layer
            nn.Linear(8, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)

def get_test(info):
    return Test(info)

```
in this network, i name it as `Test`. the network has 9 layers. 


## 5. Test your implementation and evaluate its performance.

using my own implementation, I trained the network using jsc dataset. the other setting of this network in the __init__ function is the same as the toy network. the training result is shown below:

###### Train my own network

Implement new network called **test** with **11.7k** trainable parameters to be trained (jsc-tiny has 127 trained parameters). Train the new network with hyperparameters as follows and evalute the performance in test set. We can get the following results.

| epoch | batch_size | learning_rate | validation acc | validation loss | test acc | test loss |
| :---: | :--------: | :-----------: | :------------: | :-------------: | :------: | :-------: |
|  50   |     64     |    0.00001    |     0.724      |      0.832      |  0.722   |   0.825   |







# Lab 2 for Advanced Deep Learning Systems (ADLS)

## 1. Explain the functionality of report_graph_analysis_pass and its printed jargons such as placeholder, get_attr … You might find the doc of torch.fx useful.

### Understanding `report_graph_analysis_pass` Functionality

The `report_graph_analysis_pass` function is designed to analyze and report the composition of a model's computational graph, particularly useful when working with PyTorch's `torch.fx` module for graph manipulation and analysis. This function provides insights into the graph structure by enumerating different types of nodes and their occurrences. Below is an explanation of its functionality and the terminologies used:

### Function Overview

- **Graph Representation**: It starts by converting the model's computational graph into a string representation (`str(graph.fx_graph)`) for visualization.
- **Node Type Counting**: The function then iterates over all nodes in the graph, counting occurrences of each node type. These types include:
  - `placeholder`: Represents input nodes to the graph.
  - `get_attr`: Nodes that retrieve an attribute from the model.
  - `call_function`: Nodes that call a standalone function.
  - `call_method`: Nodes that invoke a method of an object.
  - `call_module`: Nodes that call a module within the model.
  - `output`: The final output node of the graph.
- **Layer Types Analysis**: It collects and lists the types of layers (modules) present in the graph based on metadata.

### Printed Jargons Explained

- **Placeholder**: Input nodes, signifying the entry points for data.
- **Get_attr**: Nodes that access attributes of the model, such as weights or layers.
- **Call_function**: Nodes calling Python functions, often representing operations.
- **Call_method**: Nodes invoking methods on objects, like tensor operations.
- **Call_module**: Nodes representing calls to modules (layers) defined in the model.
- **Output**: Denotes the graph's output node, marking the end of computation.

### Output Format

The output includes a detailed overview of the network, highlighting the count of each node type and the diversity of layer types within the model. This analysis can be printed to the console or saved to a file, depending on whether a `file_name` is provided.

This functionality is invaluable for developers and researchers aiming to understand the underlying structure of their models, facilitating debugging, optimization, and educational purposes.


## 2. What are the functionalities of profile_statistics_analysis_pass and report_node_meta_param_analysis_pass respectively?

### Comprehensive Analysis of Computational Graphs in Neural Networks

Understanding and optimizing computational graphs in neural networks require detailed analysis of both the statistical properties of weights and activations, and the meta parameters that define the graph's structure and operations. To this end, two pivotal functions, `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass`, play crucial roles in dissecting and reporting the intricate details of computational graphs. Below, we delve into the functionalities and interconnections of these functions, highlighting their contributions to the analysis of neural network graphs.

### Profile Statistics Analysis Pass

The `profile_statistics_analysis_pass` function embarks on a systematic journey through the computational graph, aimed at gathering and analyzing statistical data on weights and activations. This process is divided into several key steps, each designed to capture specific aspects of the graph's statistical landscape:

#### Key Functionalities

1. **Statistics Registration**: The journey begins with the registration of statistical collections on the graph's nodes, setting the stage for a detailed statistical analysis.

2. **Weight Profiling**: Following registration, the function proceeds to profile the graph's weights, meticulously updating the statistical collections with parameter data extracted from the nodes.

3. **Activation Profiling**: The analysis continues with activation profiling, where inputs are processed through the graph to gather valuable activation statistics.

4. **Statistics Finalization**: The process culminates in the computation and unregistration of statistics, synthesizing the collected data for further analysis.

#### Implementation Details

- The function facilitates targeted statistical registration by node names or types and ensures a comprehensive analysis through iterative profiling of weights and activations.
- It manages the full lifecycle of statistical data, from registration to final computation, offering a holistic view of the graph's statistical properties.

### Transition to Meta Parameter Analysis

Building on the foundation laid by statistical analysis, the `report_node_meta_param_analysis_pass` function shifts the focus to meta parameter analysis, providing a granular view of the nodes' defining characteristics. This function complements the statistical analysis by offering insights into the operational parameters that influence the graph's behavior.

#### Key Functionalities

1. **Report Generation**: It meticulously constructs a report detailing the common, hardware, and software parameters of the nodes, enriching the understanding of the graph's operational dynamics.

2. **Customizable Analysis**: The function's flexible design allows for selective analysis and reporting based on the user's specific interests, enhancing the relevance of the information provided.

3. **Structured Output**: The generated report, presented in a tabulated format, offers a clear and organized depiction of the nodes' meta parameters, facilitating easy interpretation and further analysis.

#### Implementation Highlights

- By allowing for parameter filtering and dynamic reporting, the function enables focused investigations into the graph's structure and operations.
- The option to save the report provides a means for documentation and facilitates deeper examination of the graph's characteristics.

### Conclusion

Together, `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` form a comprehensive toolkit for the analysis of computational graphs in neural networks. Starting with a deep dive into the statistical properties of weights and activations, and transitioning to a detailed examination of meta parameters, these functions offer a layered approach to understanding and optimizing neural network models. The synergy between statistical analysis and meta parameter reporting illuminates the complex workings of computational graphs, guiding researchers and practitioners towards informed optimization strategies.


## 3. Explain why only 1 OP is changed after the quantize_transform_pass .

The `quantize_transform_pass` function is integral to applying quantization transformations to a neural network's computational graph. This transformation is pivotal for optimizing model size and inference speed by reducing the precision of the numbers used in computations. Here’s a streamlined explanation of why only one operation (OP) might be changed after executing this function.

### Function Overview

The primary purpose of `quantize_transform_pass` is to traverse the graph and apply quantization based on specific criteria determined by the `pass_args` argument. The function supports different quantization schemes, including quantization by type, name, or regex pattern of names. Depending on the scheme, the function selectively quantizes nodes within the graph.

#### Quantization Logic

- **Scheme Selection**: The function decides on the quantization approach (`type`, `name`, or `regex_name`) based on the `by` argument in `pass_args`.
  
- **Selective Quantization**: Based on the chosen scheme, only nodes that match the criteria are considered for quantization. For instance, when quantizing by name, only nodes with names matching the specified criteria in the configuration are targeted for transformation.

### Why Only One OP Changes

The observation that only one operation changes after the quantization pass is primarily due to the selective nature of the transformation. Specifically:

- **Targeted Approach**: The function looks for nodes that match specific criteria (e.g., a particular name when using `by: "name"`). If the configuration or criteria are set to target a single operation, then only that operation will be transformed.

- **Configuration Specificity**: In the process of quantizing by name, the function relies on a detailed configuration that maps node names to their quantization parameters. If this configuration specifies parameters for only one node, then only that node's operation is altered during the pass.

#### Detailed Process

The `graph_iterator_quantize_by_name` function illustrates this selective process by iterating over nodes, checking for quantizable operations based on a predefined list (`QUANTIZEABLE_OP`), and then applying quantization only to those nodes that meet the specified criteria in the configuration.

- **Node Filtering**: Nodes are filtered based on their compatibility with quantization and the criteria defined in `pass_args`.
  
- **Configuration Parsing**: For each eligible node, its configuration is parsed to determine how it should be quantized, affecting only those nodes explicitly mentioned in the configuration.

- **Transformation Application**: The transformation is applied by creating new modules or functions with quantized parameters, effectively changing the operation of the targeted nodes.

### Conclusion

The `quantize_transform_pass` function's selective quantization mechanism ensures that only nodes matching specific criteria are transformed. This targeted approach is why, in certain scenarios, you might observe that only one operation is altered post-quantization. Such specificity is crucial for fine-tuning model performance and efficiency through quantization, allowing for precise control over which parts of the model are optimized.


## 4. Write some code to traverse both mg and ori_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize_quantization_analysis_pass useful.

to traverse two graphs, i write a simple code block as below:
```python
for node in mg.fx_graph.nodes:
    print(node)
    print(node.args)
    print(node.kwargs)
    print(node.op)
    print(node.target)
    print(node.name)
    print(node.meta)
    print(node.meta["mase"].parameters)

for node in ori_mg.fx_graph.nodes:
    print(node)
    print(node.args)
    print(node.kwargs)
    print(node.op)
    print(node.target)
    print(node.name)
    print(node.meta)
    print(node.meta["mase"].parameters)
```

these two parts of code will print out the nodes in the two graphs. the output in the two graphs are as below:

```python
the original graph

x
{}
seq_blocks_0
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32]}, 'weight': {'type': 'float', 'precision': [32], 'shape': [16], 'from': None}, 'bias': {'type': 'float', 'precision': [32], 'shape': [16], 'from': None}}
seq_blocks_1
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32]}}
seq_blocks_2
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32]}, 'weight': {'type': 'float', 'precision': [32], 'shape': [5, 16], 'from': None}, 'bias': {'type': 'float', 'precision': [32], 'shape': [5], 'from': None}}
seq_blocks_3
{'data_in_0': {'shape': [8, 5], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32]}}
output
{}

the modified graph

x
{}
seq_blocks_0
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32]}, 'weight': {'type': 'float', 'precision': [32], 'shape': [16], 'from': None}, 'bias': {'type': 'float', 'precision': [32], 'shape': [16], 'from': None}}
seq_blocks_1
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'integer', 'precision': [16, 8]}}
seq_blocks_2
{'data_in_0': {'shape': [8, 16], 'torch_dtype': torch.float32, 'type': 'float', 'precision': [32]}, 'weight': {'type': 'float', 'precision': [32], 'shape': [5, 16], 'from': None}, 'bias': {'type': 'float', 'precision': [32], 'shape': [5], 'from': None}}
seq_blocks_3
{'data_in_0': {'shape': [8, 5], 'torch_dtype': torch.float32, 'type': 'integer', 'precision': [16, 8]}}
output
{}

```

to reference the function summarize_quantization_analysis_pass, i write a simple code block as below:# Comparative Analysis of Original and Quantized Computational Graphs

The process of quantization in neural networks is crucial for enhancing model efficiency, particularly for deployment in resource-constrained environments. This analysis delves into the comparison between an original computational graph (`ori_mg`) and its quantized counterpart (`mg`), focusing on the transformations applied during the quantization process. To facilitate this comparison, the `summarize_quantization_analysis_pass` function alongside supporting iteration functions are employed, providing a systematic approach to analyzing changes induced by quantization.

## Overview of Quantization Analysis

The `quantize_transform_pass` function applies quantization to the computational graph based on predefined criteria, targeting specific nodes for precision reduction. This selective quantization is designed to optimize the graph's performance without compromising its accuracy significantly. The process is detailed below:

### Initial Analysis

Two loops traverse the original and quantized graphs, respectively, printing each node's attributes, including arguments (`args` and `kwargs`), operation type (`op`), target function or module (`target`), name, and metadata (`meta`). This traversal highlights the nodes' initial and transformed states, providing insights into the quantization's effects.

### Comparative Analysis

#### Functionality of `summarize_quantization_analysis_pass`

This function orchestrates a comprehensive comparison between `ori_graph` and `graph`, utilizing two key iteration functions:

1. **`graph_iterator_compare_nodes`**: Compares nodes between the original and quantized graphs, listing changes in node types and identifying whether a quantization transformation has occurred.

2. **`graph_iterator_node_histogram`**: Aggregates nodes by type, generating a histogram that quantifies changes across different node categories.

### Insights from the Analysis

The analysis reveals that the quantization process predominantly affects specific node types, as illustrated by the transformation of ReLU layers in the experimental output. This transformation is indicative of the quantization strategy's targeted nature, aiming to optimize computational efficiency through precision reduction in activation functions.

#### Key Observations

- **Selective Quantization**: Only nodes meeting the quantization criteria, such as ReLU layers in this case, are transformed, underscoring the process's precision.

- **Impact on Node Attributes**: The quantization alters specific node attributes, such as the precision attribute in `meta`, from floating-point representations (`float32`) to integer-based quantization levels (`16, 8`), demonstrating the quantization's effect on data representation.

- **Unchanged Nodes**: Nodes not targeted for quantization remain unchanged, preserving the original graph's structural and functional integrity where quantization is deemed unnecessary.

### Experimental Results

The transformation is visually represented in the provided diagram, which explicitly shows the ReLU layers' modification post-quantization, with other nodes remaining largely unaffected.

```python
INFO     Quantized graph histogram:
INFO     
| Original type   | OP           |   Total |   Changed |   Unchanged |
|-----------------+--------------+---------+-----------+-------------|
| BatchNorm1d     | batch_norm1d |       1 |         0 |           1 |
| Linear          | linear       |       1 |         0 |           1 |
| ReLUInteger     | relu         |       2 |         0 |           2 |
| output          | output       |       1 |         0 |           1 |
| x               | placeholder  |       1 |         0 |           1 |
```

*In this picture, the changes to the ReLU layers are highlighted, showcasing the quantization's selective impact. The rest of the graph remains intact, illustrating the targeted approach of the quantization process.*

## Conclusion

The comparative analysis between the original and quantized computational graphs underscores the nuanced and selective nature of quantization. By focusing on specific nodes, such as activation functions, the process aims to balance computational efficiency with model performance. This analysis, facilitated by `summarize_quantization_analysis_pass` and its supporting functions, provides valuable insights into the quantization's impact, guiding future optimizations and implementations.

## 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the pass_args for your custom network might be different if you have used more than the Linear layer in your network.

### Quantization Analysis of the JSC-Toy Network

In extending the quantization analysis to a more complex network, the JSC-Toy network trained in Lab 1, the flow remains consistent with previous examples, with adjustments made primarily to accommodate the network's architecture and checkpoint loading. This section outlines the process of applying quantization to the JSC-Toy network and analyzes the impact of this transformation on the network's structure.

#### Setting Up the Network for Quantization

To prepare the JSC-Toy network for quantization, the model and its corresponding dataset are loaded with configurations specific to the network's architecture. This setup involves specifying the batch size, model name, and dataset name, followed by loading the model from the checkpoint saved during Lab 1 training. The crucial step is to ensure the correct `CHECKPOINT_PATH` is provided to load the trained model accurately.

#### Code Snippet for Model Loading

```python
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

# Update CHECKPOINT_PATH to your Lab1 trained model's path
CHECKPOINT_PATH = "<Your Model Checkpoint Path>"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)
```

and the output of the function summarize_quantization_analysis_pass is as below:(using the same quantization scheme as above)
```python
INFO     Initialising model 'jsc-toy'...
INFO     Initialising dataset 'jsc'...
INFO     Project will be created at /home/xinyi/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-02-10
INFO     Training model 'jsc-toy'...

  | Name      | Type               | Params
-------------------------------------------------
0 | model     | JSC_Toy            | 327   
1 | loss_fn   | CrossEntropyLoss   | 0     
2 | acc_train | MulticlassAccuracy | 0     
3 | acc_val   | MulticlassAccuracy | 0     
4 | acc_test  | MulticlassAccuracy | 0     
5 | loss_val  | MeanMetric         | 0     
6 | loss_test | MeanMetric         | 0     
-------------------------------------------------
```
After i set the pass_args to modify the linear layer, the output of the function summarize_quantization_analysis_pass is as below:
```python
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 16,
            "data_in_frac_width": 4,
            "weight_width": 16,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 8,
        }
    },
}

finish jsc-toy analysis pass
the results shape [16, 8]
the weights torch type [8, 4]
the bias torch type [8, 4]
the results torch type torch.Size([8, 16])
the weights torch type torch.Size([8, 16])
the results shape [16, 8]
the weights torch type [8, 4]
the bias torch type [8, 4]
the results torch type torch.Size([8, 8])
the weights torch type torch.Size([8, 8])
the results shape [16, 8]
the weights torch type [8, 4]
the bias torch type [8, 4]
the results torch type torch.Size([5, 8])
the weights torch type torch.Size([5, 8])
```

This picture shows the structure of Jsc-toy network. This network is bigger than jsc-tiny network.


#### 6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the Quantized Layers .

this is the pass_args that i wrote previously for quantization:

```python
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}
```

in order to show the weights of the layers, i write a simple code block as below:

```python
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
```

this code block will print out the weights size and format. the original weights can be found in the output of the orignial precision of args. and the quantized weights can be found in the output of the quantized precision of args. the output of the code block is as below:

In order to test the accuracy of this data structure, i then changed the arguments into another set:

```python
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 16,
            "data_in_frac_width": 4,
            "weight_width": 16,
            "weight_frac_width": 8,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}
```

And the results are as follows:

```python
the results shape [16, 4]
the weights torch type [16, 8]
the bias torch type [8, 4]
```

When changing the arguments, the weights and bias are also changed.

```python
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 16,
            "data_in_frac_width": 4,
            "weight_width": 16,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 8,
        }
    },
}
```
When using new arguments, the weights and bias are also changed as follows:
```python
the results shape [16, 4]
the weights torch type [16, 4]
the bias torch type [8, 8]
```

As the picture above, we can see the changes made by arguments. This affects the precision of data_in, weights and bias as quantizes above. So the results are correct.

### 7. Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.

using the .toml file that i wrote previously, i can use the command line interface to perform the quantization. the command line is as below:

```python
./ch transform --config configs/examples/jsc_toy_by_type.toml --task cls --cpu=0
```

and the output of the command line is as below:

```python
INFO     Initialising model 'jsc-toy'...
INFO     Initialising dataset 'jsc'...
INFO     Project will be created at /home/xinyi/mase_xinyi/mase_output/jsc-toy
INFO     Transforming model 'jsc-toy'...
INFO     Loaded pytorch lightning checkpoint from /home/xinyi/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt
INFO     Quantized graph histogram:
INFO     
| Original type   | OP           |   Total |   Changed |   Unchanged |
|-----------------+--------------+---------+-----------+-------------|
| BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
| Linear          | linear       |       3 |         3 |           0 |
| ReLU            | relu         |       4 |         0 |           4 |
| output          | output       |       1 |         0 |           1 |
| x               | placeholder  |       1 |         0 |           1 |
INFO     Saved mase graph to /home/xinyi/mase_xinyi/mase_output/jsc-toy/software/transform/transformed_ckpt
INFO     Transformation is completed
```

## Optional task: FLOPs and BitOPs

I use the funtion defined in the source code to calculate the FLOPs. About BitOPs, i simply calculate the number of bit calculation persuming all the data are multiplied between layers. The results are as below:

```python
store_flops_data {BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): {'total_parameters': 32, 'computations': 512, 'backward_computations': 512, 'input_buffer_size': 128, 'output_buffer_size': 128}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 128, 'backward_computations': 128, 'input_buffer_size': 128, 'output_buffer_size': 128}, Linear(in_features=16, out_features=5, bias=True): {'total_parameters': 80, 'computations': 640.0, 'backward_computations': 1280.0, 'input_buffer_size': 128, 'output_buffer_size': 40}, ReLU(inplace=True): {'total_parameters': 0, 'computations': 40, 'backward_computations': 40, 'input_buffer_size': 40, 'output_buffer_size': 40}}
store_bitops_data {BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True): {'computations': 512, 'bitops': 16384}, ReLU(inplace=True): {'computations': 128, 'bitops': 4096}, Linear(in_features=16, out_features=5, bias=True): {'computations': 640.0, 'bitops': 20480.0}, ReLU(inplace=True): {'computations': 40, 'bitops': 1280}}
flops 1320.0
bitops 42240.0
```

my calculation functions are based on flops calculation and use the conputation value inside the flop calculation function. then using the frount calculation number to calculate the BitOPs number.