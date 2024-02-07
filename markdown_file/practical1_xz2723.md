### lab1 for Advanced Deep Learning Systems (ADLS)

## 1. What is the impact of varying batch sizes and why?
1. batchsize can affect the running time of training. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

2. batchsize can affect the overall accuracy but we can't for sure this will improve the accuracy or not. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.

3. batchsize can affect the generalization of the model. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.

4. batchsize can affect the memory usage. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

5. batchsize can affect the overall running time. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

6. batchsize can affect the convergence speed. because the smaller the batchsize, the more iterations are needed to reach the same accuracy.

7. batchsize can affect the stability of the model. because the smaller batchsize can be seen as a regularization technique, which can prevent the model from overfitting.

   Try different batch sizes of the training and the results are shown as follows:

   ###### different batch sizes

   | epoch | batch_size | learning_rate | training loss | validation loss |
   | :---: | :--------: | :-----------: | :-----------: | :-------------: |
   |  50   |     64     |    0.0001     |    0.7837     |      0.878      |
   |  50   |    128     |    0.0001     |    0.9908     |     0.8581      |
   |  50   |    256     |    0.0001     |    0.9283     |      0.859      |
   |  50   |    512     |    0.0001     |     1.075     |      1.053      |
   |  50   |    1024    |    0.0001     |     1.049     |      1.086      |

   From the above form we can see that, different batch sizes have an influence on learning time. the higher the batch size is, the shorter the training time will be. this is because the training procedure is calculated by batch size and the time this batches were sent to the training network.

   ###### 2. What is the impact of varying maximum epoch number?

1. the larger the maximum epoch number, the more iterations are needed to reach the same accuracy. so the epoch number can affect the running time of training.

2. the epoch number can affect the overall accuracy but we can't for sure this will improve the accuracy or not.

3. the epoch number can affect the generalization of the model. this is because the larger the epoch number, the more likely the model will overfit.

4. the epoch number can affect the memory usage. because the larger the epoch number, the more iterations are needed to reach the same accuracy.

   Try different max epoch number, the results are as follows:

   ###### different max epoch

   | epoch | batch_size | learning_rate | training loss | validation loss |
   | :---: | :--------: | :-----------: | :-----------: | :-------------: |
   |  10   |    256     |    0.0001     |    0.9919     |     0.9922      |
   |  50   |    256     |    0.0001     |    0.9283     |      0.859      |
   |  100  |    256     |    0.0001     |     1.073     |     0.8435      |
   |  150  |    256     |    0.0001     |    0.8844     |      0.84       |
   |  200  |    256     |    0.0001     |    0.6919     |     0.8382      |

   The table show the training loss and validation loss with respect to the change of epochs. When the epoch is small than 50, the training loss decreases with increasing epochs. When the epoch is larger than 150, the training loss is much smaller than validation loss, indicating an overfitting problem.

## 3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
1. with a large learning rate, the model will converge faster but the accuracy will be lower. this is because the large learning rate will make the model jump over the optimal point.

2. with a small learning rate, the model will converge slower but the accuracy will be higher. this is because the small learning rate will make the model converge to the optimal point.

3. the relationship between learning rates and batch sizes is that the smaller the batch size, the smaller the learning rate should be. this is because the smaller batch size can be seen as a regularization technique, which can prevent the model from overfitting. and the smaller learning rate can also be seen as a regularization technique, which can prevent the model from overfitting.

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

### Lab 2 for Advanced Deep Learning Systems (ADLS)

#### 1. Explain the functionality of report_graph_analysis_pass and its printed jargons such as placeholder, get_attr … You might find the doc of torch.fx useful.

the report_graph_analysis_pass function is as below:

```python
def report_graph_analysis_pass(graph, pass_args={"file_name": None}):
    """
    Generates a report for the graph analysis
    and prints out an overview of the model in a table.
    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass can take a string argument named "file_name", defaults to None
    :type pass_args: dict, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, dict)
    """
    file_name = pass_args.get("file_name")
    buff = ""
    buff += str(graph.fx_graph)
    count = {
        "placeholder": 0,
        "get_attr": 0,
        "call_function": 0,
        "call_method": 0,
        "call_module": 0,
        "output": 0,
    }
    layer_types = []

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].module is not None:
            layer_types.append(node.meta["mase"].module)

    for node in graph.fx_graph.nodes:
        count[node.op] += 1
    buff += f"""\nNetwork overview:
{count}
Layer types:
{layer_types}"""
    if file_name is None:
        print(buff)
    else:
        with open(file_name, "w", encoding="utf-8") as outf:
            outf.write(buff)
    return graph, {}
```
this function will print out the graph of the model and the number of each type of nodes in the graph. it will print out the graph in the form of a table and the contents of the table are the number of each type of nodes in the graph. different types of nodes are defined in the printed jargons such as placeholder, get_attr are the types of nodes in the graph, as shown in the code above. the meaning of each type of nodes are as below:

1. placeholder node is the input node of the graph. 
2. get_attr node is the node that get the attribute of the model. 
3. call_function node is the node that call the function of the model. 
4. call_method node is the node that call the method of the model. 
5. call_module node is the node that call the module of the model. 
6. output node is the output of the graph.

in this function, it print out the nodes in the graph and the number of each type of nodes in the graph. 

the output graph of the model is as below:
(add pictures here)

#### 2. What are the functionalities of profile_statistics_analysis_pass and report_node_meta_param_analysis_pass respectively?

##### function profile_statistics_analysis_pass

the function profile_statistics_analysis_pass is as below:
```python
def profile_statistics_analysis_pass(graph, pass_args: dict):
    """
    Perform profile statistics analysis on the given graph.

    :param graph: The graph to perform analysis on.
    :type graph: MaseGraph

    :param pass_args: The arguments for the analysis pass.
    :type pass_args: dict

    :return: The modified graph and an empty dictionary.
    :rtype: tuple(MaseGraph, dict)
    """

    graph = graph_iterator_register_stat_collections(
        graph,
        by=pass_args["by"],
        target_weight_nodes=pass_args["target_weight_nodes"],
        target_act_nodes=pass_args["target_activation_nodes"],
        weight_stats=pass_args["weight_statistics"],
        act_stats=pass_args["activation_statistics"],
        profile_output_act=pass_args.get("profile_output_activation", False),
    )

    graph = graph_iterator_profile_weight(graph)

    graph = graph_iterator_profile_act(
        graph,
        input_generator=pass_args["input_generator"],
        num_samples=pass_args["num_samples"],
    )

    graph = graph_iterator_compute_and_unregister_stats(graph)

    return graph, {}
```
this function will perform profile statistics analysis on the given graph. 
it achieves the following functionalities:
1. register the statistics collections on the graph.
2. profile the weight of the graph.
3. profile the activation of the graph.
4. compute and unregister the statistics of the graph.

to register the statistics collections on the graph, it will iterate through the nodes in the graph and register the statistics collections on the nodes. using the statics collections function as below:

```python
def graph_iterator_register_stat_collections(
    graph,
    by,
    target_weight_nodes,
    target_act_nodes,
    weight_stats,
    act_stats,
    profile_output_act=False,
):
    match by:
        case "name":
            graph = graph_iterator_register_stat_collections_by_name(
                graph,
                target_weight_nodes,
                target_act_nodes,
                weight_stats,
                act_stats,
            )
        case "type":
            graph = graph_iterator_register_stat_collections_by_type(
                graph,
                target_weight_nodes,
                target_act_nodes,
                weight_stats,
                act_stats,
            )
        case _:
            raise ValueError(f"Unknown by: {by}")

    return graph
```
this function will iterate through the nodes in the graph and register the statistics collections on the nodes. it will iterate through the nodes in the graph and register the statistics collections on the nodes by name or by type. so after the iteration in this function, the statistics collections are registered on the nodes in the graph, also means that the statistics collections are added to the nodes in the graph.

after the statistics collections are registered on the nodes in the graph, the weight of the graph will be profiled, which is the function graph_iterator_profile_weight. the function is as below:
```python
def graph_iterator_profile_weight(graph):
    for node in tqdm(
        graph.fx_graph.nodes,
        total=len(list(graph.fx_graph.nodes)),
        desc="Profiling weight statistics",
    ):
        if node.op != "call_module":
            continue

        param_dict = dict(node.meta["mase"].module.named_parameters())
        buffer_dict = dict(node.meta["mase"].module.named_buffers())
        p_b_dict = {**param_dict, **buffer_dict}

        for w_name, s_meta in node.meta["mase"].parameters["software"]["args"].items():
            stat = s_meta["stat"]
            if not isinstance(stat, WeightStatCollection):
                continue

            w = p_b_dict[w_name]
            stat.update(w.data)

    return graph
```
this function will iterate through the nodes in the graph and profile the weight of the graph. it will iterate through the nodes in the graph and get the parameters of the nodes. then it will update the statistics collections of the nodes with the parameters of the nodes. so after the iteration in this function, the statistics collections of the nodes are updated with the parameters of the nodes.

after iterating through the nodes in the graph and profile the weight of the graph, the activation of the graph will be profiled, which is the function graph_iterator_profile_act. the function is as below:
```python
def graph_iterator_profile_act(graph, input_generator, num_samples):
    act_profiler = ActProfiler(graph.model, garbage_collect_values=True)

    max_batches = math.ceil(num_samples / input_generator.batch_size)

    for i in tqdm(range(max_batches), desc="Profiling act statistics"):
        batch = next(input_generator)
        act_profiler.run(*batch.values())

    return graph
```

in this function, it will iterate through the nodes in the graph and profile the activation of the graph. it will iterate through the nodes in the graph and get the input of the nodes. then it will run the activation profiler with the input of the nodes. so after the iteration in this function, the activation profiler is run with the input of the nodes.

after iterating through the nodes in the graph and profile the activation of the graph, the statistics of the graph will be computed and unregistered, which is the function graph_iterator_compute_and_unregister_stats. the function is as below:
```python
def graph_iterator_compute_and_unregister_stats(graph):
    for node in graph.fx_graph.nodes:
        for entry, s_meta in (
            node.meta["mase"].parameters["software"].get("args", {}).items()
        ):
            stat = s_meta["stat"]
            if isinstance(stat, (WeightStatCollection, ActStatCollection)):
                result = stat.compute()
                set_meta_arg_stat(node, entry, result)
        # for entry, s_meta in (
        #     node.meta["mase"].parameters["software"]["results"].items()
        # ):
        #     stat = s_meta["stat"]
        #     if isinstance(stat, ActStatCollection):
        #         result = stat.compute()
        #         set_meta_result_stat(node, entry, result)
    return graph
```

in this function, it will iterate through the nodes in the graph and compute and unregister the statistics of the graph. it will iterate through the nodes in the graph and get the statistics of the nodes. then it will compute and unregister the statistics of the nodes. so after the iteration in this function, the statistics of the nodes are computed and unregistered.

so in conclusion, the function profile_statistics_analysis_pass achieve the following steps:
1. register the statistics collections on the graph.
2. profile the weight of the graph.
3. profile the activation of the graph.
4. compute and unregister the statistics of the graph.

##### function report_node_meta_param_analysis_pass

the full function report_node_meta_param_analysis_pass is as below:
```python
def report_node_meta_param_analysis_pass(graph, pass_args: dict = None):
    """
    Perform meta parameter analysis on the nodes in the graph and generate a report.

    :param graph: The graph to analyze.
    :type graph: MaseGraph
    :param pass_args: Optional arguments for the analysis pass, a dict of arguments for this pass, including
        - "which": str, and a list of options in ["all", "common", "hardware", "software"], default ["all"]
        - "save_path": str, a str of path to save the table, default None
    :type pass_args: dict, default None
    :return: The analyzed graph and an empty dictionary.
    :rtype: tuple(MaseGraph, dict)
    """
    which_param = pass_args.get("which", ("all",))
    assert isinstance(which_param, (list, tuple))
    for param in which_param:
        assert param in [
            "all",
            "common",
            "hardware",
            "software",
        ], f"Invalid which_param {param}, must be a list of options in ['all', 'common', 'hardware', 'software'], got {param}"
    save_path = pass_args.get("save_path", None)

    headers = [
        "Node name",
        "Fx Node op",
        "Mase type",
        "Mase op",
    ]

    if "common" in which_param or "all" in which_param:
        headers.append("Common Param")
    if "hardware" in which_param or "all" in which_param:
        headers.append("Hardware Param")
    if "software" in which_param or "all" in which_param:
        headers.append("Software Param")

    rows = []
    for node in graph.fx_graph.nodes:
        new_row = [
            node.name,
            node.op,
            node.meta["mase"].parameters["common"]["mase_type"],
            node.meta["mase"].parameters["common"]["mase_op"],
        ]

        if "common" in which_param or "all" in which_param:
            new_row.append(pformat(node.meta["mase"].parameters["common"]))
        if "hardware" in which_param or "all" in which_param:
            new_row.append(pformat(node.meta["mase"].parameters["hardware"]))
        if "software" in which_param or "all" in which_param:
            new_row.append(pformat(node.meta["mase"].parameters["software"]))

        rows.append(new_row)

    table_txt = tabulate(rows, headers=headers, tablefmt="grid")
    logger.info("Inspecting graph [add_common_meta_param_analysis_pass]")
    logger.info("\n" + table_txt)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(save_path), "w") as f:
            f.write(table_txt)
            logger.info(f"Node meta param table is saved to {save_path}")
    return graph, {}
```

this function will perform meta parameter analysis on the nodes in the graph and generate a report.

it achieves the following functionalities:
1. generate a report of the meta parameter analysis on the nodes in the graph. (1) inspect the graph. (2) print out the table of the meta parameter analysis on the nodes in the graph. (3) save the table of the meta parameter analysis on the nodes in the graph to the given path.
2. save the report to the given path.


#### 3. Explain why only 1 OP is changed after the quantize_transform_pass .

the function quantize_transform_pass is as below:
```python
def quantize_transform_pass(graph, pass_args=None):
    """
    Apply quantization transformation to the given graph.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation.
    :type pass_args: dict, optional

    :return: The transformed graph.
    :rtype: tuple
    :raises ValueError: If the quantize "by" argument is unsupported.


    - pass_args
        - by -> str : different quantization schemes choose from ["type", "name", "regx_name"]
    """

    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_iterator_quantize_by_type(graph, pass_args)
        case "name":
            graph = graph_iterator_quantize_by_name(graph, pass_args)
        case "regex_name":
            graph = graph_iterator_quantize_by_regex_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
```

this function will apply quantization transformation to the given graph.only 1 OP is changed after this function is because this function can only read one type of quantization scheme. take the quantize by name as an example, it will iterate through the nodes in the graph and quantize the nodes in the graph by name. the function graph_iterator_quantize_by_name is as below:

```python
def graph_iterator_quantize_by_name(graph, config: dict):
    for node in graph.fx_graph.nodes:
        import pdb; pdb.set_trace()
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, node.name)
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        output_layers_names = node_config.get("additional_layers_outputs", [])
        output_layers = [
            get_node_target_by_name(graph, name) for name in output_layers_names
        ]
        input_layers_names = node_config.get("additional_layers_inputs", [])
        input_layers = [
            get_node_target_by_name(graph, name) for name in input_layers_names
        ]
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
                node.meta,
                input_layers=input_layers,
                output_layers=output_layers,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            update_quant_meta_param(node, node_config, get_mase_op(node))
            logger.debug(f"Quantized module: {node.target} with config: {node_config}")
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
            logger.debug(
                f"Quantized function: {node.target} with config: {node_config}"
            )
        else:
            raise ValueError(
                "Unsupported node type for quantisation: {}".format(get_mase_type(node))
            )
    return graph
```

in this function, it will iterate and quantize the nodes, then get the names of them. then it will parse the node config and get the output layers and input layers of the nodes. in this function, only 1 OP is changed is because this function only read the nodes by OP names, so one time running this function can only change 1 OP of the nodes.

#### 4. Write some code to traverse both mg and ori_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize_quantization_analysis_pass useful.

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

(add pictures here)

to reference the function summarize_quantization_analysis_pass, i write a simple code block as below:
```python
def summarize_quantization_analysis_pass(
    ori_graph, graph, save_dir: str = None
) -> None:
    """
    Summarizes the quantization analysis pass.

    Args:
        ori_graph: The original graph.
        graph: The modified graph.
        save_dir (optional): The directory to save the summary files. Defaults to None.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    table_path = os.path.join(save_dir, "quantize_table.csv") if save_dir else None
    histogram_path = (
        os.path.join(save_dir, "quantize_histogram.csv") if save_dir else None
    )
    graph_iterator_compare_nodes(ori_graph, graph, save_path=table_path, silent=False)
    graph_iterator_node_histogram(ori_graph, graph, save_path=histogram_path)
```
in this function, it uses two iteration functions to compares the nodes in two graphs. the first function is graph_iterator_compare_nodes, which is as below:
```python
def graph_iterator_compare_nodes(
    ori_graph, graph, save_path=None, silent=False
) -> pd.DataFrame:
    """List all nodes in the graph and compare the original and quantized nodes."""

    def get_type_str(node):
        if node.op == "call_module":
            return type(get_node_actual_target(node)).__name__
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
            "patched_func",
        ]:
            return get_node_actual_target(node).__name__
        elif get_mase_type(node) in ["implicit_func"]:
            actual_target = get_node_actual_target(node)
            if isinstance(actual_target, str):
                return actual_target
            else:
                return actual_target.__name__
        else:
            return node.target

    headers = [
        "Ori name",
        "New name",
        "MASE_TYPE",
        "Mase_OP",
        "Original type",
        "Quantized type",
        "Changed",
    ]
    rows = []
    for ori_n, n in zip(ori_graph.fx_graph.nodes, graph.fx_graph.nodes):
        rows.append(
            [
                ori_n.name,
                n.name,
                get_mase_type(n),
                get_mase_op(n),
                get_type_str(ori_n),
                get_type_str(n),
                type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)),
            ]
        )
    if not silent:
        logger.debug("Compare nodes:")
        logger.debug("\n" + tabulate(rows, headers=headers, tablefmt="orgtbl"))
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(tabulate(rows, headers=headers))

    df = pd.DataFrame(rows, columns=headers)
    if save_path is not None:
        df.to_csv(save_path)

    return df
```
this function iterate through the nodes and conpare teh types of the nodes in the two graphs. the second function is graph_iterator_node_histogram, which is as below:
```python
def graph_iterator_node_histogram(ori_graph, graph, save_path: str = None):
    """Group nodes by their types and count the number of nodes in each group."""
    df = graph_iterator_compare_nodes(ori_graph, graph, save_path=None, silent=True)
    histogram_df = df.groupby(["Original type"]).agg(
        OP=pd.NamedAgg(column="Mase_OP", aggfunc="first"),
        Total=pd.NamedAgg(column="Changed", aggfunc="count"),
        Changed=pd.NamedAgg(column="Changed", aggfunc=lambda x: np.sum(x)),
        Unchanged=pd.NamedAgg(
            column="Changed", aggfunc=lambda x: np.sum(1 - np.array(x))
        ),
    )
    logger.info("Quantized graph histogram:")
    logger.info("\n" + tabulate(histogram_df, headers="keys", tablefmt="orgtbl"))
    if save_path is not None:
        histogram_df.to_csv(save_path)


# def graph_iterator_compare_nodes(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass


# def graph_iterator_node_histogram(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass
```
this function will build a histogram of the nodes in the two graphs. it will iterate through the nodes and group the nodes by their types and count the number of nodes in each group. so after the iteration in this function, the histogram of the nodes in the two graphs are built.

to traverse two graphs, i also use the function summarize_quantization_analysis_pass. the output of the function is as below:

![](lab2pic\lab2-diff.png)

In this picture, i changed the ReLU layers so the output here shows the changes of the ReLU only. The rest stays the same.

#### 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the pass_args for your custom network might be different if you have used more than the Linear layer in your network.

the quantisation flow to the bigger JSC network is the same as above. only need to change the model loading part. the code is as below:
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

# 📝️ change this CHECKPOINT_PATH to the one you trained in Lab1
CHECKPOINT_PATH = "/home/xinyi/ADL/mase_xinyi/mase_output/jsc-toy_classification_jsc_2024-01-25/software/training_ckpts/best.ckpt"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

```

and the output of the function summarize_quantization_analysis_pass is as below:(using the same quantization scheme as above)

![](lab2pic\lab2-jsc-q3.jpg)

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
        args_weights_precision = node.meta['mase'].parameters['common']['args']['weight']['precision']
        print('the weights torch type', args_weights_precision)
        args_bias_precision = node.meta['mase'].parameters['common']['args']['bias']['precision']
        print('the bias torch type', args_bias_precision)
```

this code block will print out the weights size and format. the original weights can be found in the output of the orignial precision of args. and the quantized weights can be found in the output of the quantized precision of args. the output of the code block is as below:

![(add pictures here)](lab2pic\lab2-quantize1.png)

In order to test the accuracy of this data structure, i then changed the arguments into another set:

```python
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 16,
            "data_in_frac_width": 8,
            "weight_width": 8,
            "weight_frac_width": 8,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}
```

And the results are as follows:

![](E:\master-2\ADLS-coursework\lab2pic\lab2-quantize2.png)

As the picture above, we can see the changes made by arguments. This affects the precision of data_in, weights and bias as quantizes above. So the results are correct.