import time

from datasci.workflow.evaluate.evaluate import EvaluateProcesser

from datasci.workflow.node.base import BaseNode
from datasci.workflow.train.train import TrainProcesser
from datasci.workflow.predict.predict import PredictProcesser
from datasci.workflow.output.join import JoinProcesser
from datasci.workflow.output.save import SaveProcesser


class TrainNode(BaseNode):

    def run(self):
        if self.input_data is not None:
            self.input_data = self.input_merge(axis=0)
        train_class = TrainProcesser(
            **self.node_class_params) if self.node_class_params is not None else TrainProcesser()
        multi_process = self.run_params.get('multi_process', False) if self.run_params is not None else False
        train_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = None
        self.is_finished = True
        return self.output_data


class PredictNode(BaseNode):

    def run(self):
        if self.input_data is not None:
            self.input_data = self.input_merge(axis=0)
        predict_class = PredictProcesser(
            **self.node_class_params) if self.node_class_params is not None else PredictProcesser()
        multi_process = self.run_params.get('multi_process', False) if self.run_params is not None else False
        result = predict_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class EvaluateNode(BaseNode):

    def run(self):
        if self.input_data is not None:
            self.input_data = self.input_merge(axis=0)
        evaluate_class = EvaluateProcesser(
            **self.node_class_params) if self.node_class_params is not None else EvaluateProcesser()
        multi_process = self.run_params.get('multi_process', False) if self.run_params is not None else False
        result = evaluate_class.run(data=self.input_data, multi_process=multi_process)
        self.output_data = result
        self.is_finished = True
        return self.output_data


class SaveNode(BaseNode):

    def run(self):
        self.input_data = self.input_merge()
        save_class = SaveProcesser(
            **self.node_class_params) if self.node_class_params is not None else SaveProcesser()

        output_config = self.run_params.get('output_config', None) if self.run_params is not None else None
        pagesize = self.run_params.get('pagesize', 10000) if self.run_params is not None else 10000
        model_name = self.run_params.get('model_name', 'default') if self.run_params is not None else 'default'

        ex_col = {
            'model_version': model_name,
            'dt': "%s" % time.strftime("%Y%m%d", time.localtime())
        }
        save_class.run(data=self.input_data, extend_columns=ex_col, output_config=output_config, pagesize=pagesize)
        self.is_finished = True
        return self.output_data


class JoinNode(BaseNode):

    def run(self):
        join_key = self.run_params.get('join_key', None) if self.run_params is not None else None
        if self.input_data is not None:
            if isinstance(self.input_data[0], dict):
                self.input_data = self.input_merge()
            join_class = JoinProcesser(
                **self.node_class_params) if self.node_class_params is not None else JoinProcesser()
            result = join_class.run(data=self.input_data, join_key=join_key)
        else:
            result = None
        self.output_data = result
        self.is_finished = True
        return self.output_data
