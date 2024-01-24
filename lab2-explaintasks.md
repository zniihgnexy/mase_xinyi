1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` ... You might find the doc of [torch.fx](https://pytorch.org/docs/stable/fx.html) useful.

report_graph_analysis_pass: 用来打印出graph的信息，包括graph中的node的信息，以及graph的输入输出信息。
summary of the function report_graph_analysis_pass is:
1. generate a report for the graph analysis
2. print out an overview of the model in a table
3. return a tuple of a MaseGraph and an empty dict (no additional info to return)
4. the function takes a MaseGraph and a dict as input
5. the dict can have a string argument named "file_name"
6. if the "file_name" argument is not None, the report will be written to the file
7. if the "file_name" argument is None, the report will be printed to the console
8. the report includes the following information:
   1. the number of each type of node in the graph
   2. the types of layers in the graph
----------------------------------------------------------------------------------------------------


2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?

summary of the function profile_statistics_analysis_pass is:
1. perform profile statistics analysis on the given graph
2. return a tuple of the modified graph and an empty dict
3. the function takes a MaseGraph and a dict as input
4. the dict can have the following arguments:
   1. "by": "name" or "type"
   2. "target_weight_nodes": a list of node names or types
   3. "target_activation_nodes": a list of node names or types
   4. "weight_statistics": a dict of statistics to collect for weights
   5. "activation_statistics": a dict of statistics to collect for activations
   6. "profile_output_activation": whether to profile the output activation
   7. "input_generator": a generator of input data
   8. "num_samples": the number of samples to profile
5. the function performs the following steps:
   1. register stat collections for weights and activations
   2. profile weights
   3. profile activations
   4. compute and unregister stats
   5. return the modified graph and an empty dict
----------------------------------------------------------------------------------------------------

summary of the function report_node_meta_param_analysis_pass is:
summary of the function report_node_meta_param_analysis_pass is:
1. perform meta parameter analysis on the nodes in the graph
2. generate a report
3. print out the report
4. return a tuple of a MaseGraph and an empty dict (no additional info to return)
5. the function takes a MaseGraph and a dict as input
6. the dict can have two arguments:
   1. "which": str, and a list of options in ["all", "common", "hardware", "software"], default ["all"]
   2. "save_path": str, a str of path to save the table, default None
7. the report includes the following information:
   1. the name of each node
   2. the Fx node op
   3. the Mase type
   4. the Mase op
   5. the common parameters
   6. the hardware parameters
   7. the software parameters
----------------------------------------------------------------------------------------------------


3. Explain why only 1 OP is changed after the `quantize_transform_pass` .
because the quantization is done by name(or type, regex_name), and the name of the OP is not changed
so only the OP with the specific name (or type, regex_name) is changed

----------------------------------------------------------------------------------------------------


4. Write some code to traverse both `mg` and `ori_mg`, check and comment on the nodes in these two graphs. You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear` layer in your network.

6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the [Quantized Layers](../../machop/chop/passes/transforms/quantize/quantized_modules/linear.py) .

7. Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.