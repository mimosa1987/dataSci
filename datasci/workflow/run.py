import os
import time
import json
from queue import Queue
import pandas as pd
from datasci.workflow.node.common import EndNode
from datasci.utils.reflection import Reflection


def _init_node(nodename, config):
    node_class = config.get(nodename).get("node_class")
    idx = node_class.rfind(".")
    module_path = node_class[0: idx]
    class_name = node_class[idx + 1: len(node_class)]
    if module_path is None:
        print("Module path is None!")
        exit(-1)
    if class_name is None:
        print("Class name is None!")
        exit(-1)

    params = config.get(nodename).get("params", None)
    if len(params) == 0:
        params = None

    node_class_params = None
    run_params = None
    if params is not None:
        node_class_params = params.get("node_class_params", None)
        run_params = params.get("run_params", None)

    next_nodes = config.get(nodename).get("next", None)
    if len(next_nodes) == 0 or next_nodes == "" or next_nodes == []:
        next_nodes = None
    input_data_file = config.get(nodename).get("input")

    if input_data_file is not None and os.path.exists(input_data_file):
        input_data = pd.read_csv(input_data_file)
    else:
        input_data = None

    init_params = {
        "node_name": nodename,
        "next_nodes": next_nodes,
        "node_class_params": node_class_params,
        "run_params": run_params,
        "input_data": input_data
    }
    cls_obj = Reflection.reflect_obj(module_path=module_path, class_name=class_name, params=init_params)

    return cls_obj


def run(config=None):
    logo = '''

    ________         _____         ________       _____        ___       __               ______  ________________                  
    ___  __ \______ ___  /_______ ___  ___/__________(_)       __ |     / /______ ___________  /_____  ____/___  /______ ___      __
    __  / / /_  __ `/_  __/_  __ `/_____ \ _  ___/__  /        __ | /| / / _  __ \__  ___/__  //_/__  /_    __  / _  __ \__ | /| / /
    _  /_/ / / /_/ / / /_  / /_/ / ____/ / / /__  _  /         __ |/ |/ /  / /_/ /_  /    _  ,<   _  __/    _  /  / /_/ /__ |/ |/ / 
    /_____/  \__,_/  \__/  \__,_/  /____/  \___/  /_/          ____/|__/   \____/ /_/     /_/|_|  /_/       /_/   \____/ ____/|__/  


            '''
    print(logo)

    result = {}
    if config is None:
        from datasci.workflow.config.global_config import global_config
        config = global_config.get("workflow")
    with open(config) as f:
        conf = f.read()
    run_dag_config = json.loads(conf)

    run_queue = Queue(maxsize=0)
    node_map = dict()
    ready_list = list()
    start_node = _init_node(nodename='start', config=run_dag_config)
    node_map['start'] = start_node
    run_queue.put("start")
    ready_list.append('start')
    i = 1
    print("------------------------ START ------------------------")
    while run_queue.qsize() != 0:
        print(">>> Execute Queue size [ %s ]" % run_queue.qsize())
        print(">>> Ready to Execute Nodes [ %s ]" % ", ".join(ready_list))
        node_name = run_queue.get()
        node = node_map.get(node_name)
        if node_name in ready_list:
            ready_list.remove(node_name)
        if node.is_finished:
            print(">>> Node [ %s ] is finished, SKIP this Node" % node.node_name)
            continue
        print("\n")
        print("------------------------ STEP %s ------------------------" % i)
        print("NODE NAME :[ %s ] , START TIME : [ %s ] " % (
            node.node_name, time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
        print(">>> [ %s ]  Node Running ... ... " % node.node_name)
        ret = node.run()
        if isinstance(node, EndNode):
            result[node.node_name] = ret
        print(">>> [ %s ]  Node Finished ! " % node.node_name)
        print("NODE NAME :[ %s ] , FINISHED TIME : [ %s ]  " % (
            node.node_name, time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
        i += 1
        if node.next_nodes is not None:
            for n_name in node.next_nodes:
                if n_name not in node_map:
                    node_map[n_name] = _init_node(nodename=n_name, config=run_dag_config)
                sub_node = node_map[n_name]
                if sub_node.input_data is None or isinstance(sub_node.input_data, list):
                    sub_node.add_input(node.output_data)
                ready_list.append(n_name)
                run_queue.put(n_name)
        else:
            continue
    print("\n")
    print("------------------------ FINISHED ------------------------")
    return result
