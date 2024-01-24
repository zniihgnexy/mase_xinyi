import logging

# summary of the function report_graph_analysis_pass is:
# 1. generate a report for the graph analysis
# 2. print out an overview of the model in a table
# 3. return a tuple of a MaseGraph and an empty dict (no additional info to return)
# 4. the function takes a MaseGraph and a dict as input
# 5. the dict can have a string argument named "file_name"
# 6. if the "file_name" argument is not None, the report will be written to the file
# 7. if the "file_name" argument is None, the report will be printed to the console
# 8. the report includes the following information:
#    1. the number of each type of node in the graph
#    2. the types of layers in the graph


logger = logging.getLogger(__name__)


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
