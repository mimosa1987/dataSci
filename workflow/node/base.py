import pandas as pd


class BaseNode(object):
    def __init__(self, node_name, next_nodes, input_data, node_class_params=None, run_params=None):
        self.node_name = node_name
        self.next_nodes = next_nodes
        self.node_class_params = node_class_params
        self.run_params = run_params
        self.output_data = None
        self.is_finished = False
        self.input_data = input_data

    def add_input(self, input):
        if input is not None and len(input) > 0:
            if self.input_data is None:
                self.input_data = list()
            self.input_data.append(input)

    def input_merge(self, axis=0):
        if isinstance(self.input_data, list):
            if len(self.input_data) == 0:
                return None
            elif len(self.input_data) == 1:
                return self.input_data[0]
            else:
                if isinstance(self.input_data[0], pd.DataFrame):
                    ret_value = pd.concat(self.input_data, axis=axis)
                    if axis == 0:
                        ret_value.reset_index(drop=True, inplace=True)
                    return ret_value
                elif isinstance(self.input_data[0], dict):
                    result = dict()
                    for item in self.input_data:
                        result.update(item)
                    return result
                elif isinstance(self.input_data[0], str):
                    return ",".join(self.input_data)
        else:
            return self.input_data

    def run(self):
        print("Run the node %s" % self.node_name)
