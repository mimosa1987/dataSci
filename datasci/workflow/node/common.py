import time
from datasci.workflow.node.base import BaseNode
from datasci.workflow.start.init import InitProcesser


class StartNode(BaseNode):

    def run(self):
        from datasci.utils.mylog import get_stream_logger
        from datasci.workflow.config.log_config import log_level
        log = get_stream_logger("START NODE", level=log_level)
        log.info("Job start at %s " % time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        start_class = InitProcesser(
            **self.node_class_params) if self.node_class_params is not None else InitProcesser()
        start_class.run()
        self.output_data = self.input_data
        self.is_finished = True
        return self.output_data


class EndNode(BaseNode):

    def run(self):
        is_merge = self.run_params.get('is_merge', None) if self.run_params is not None else False
        axis = self.run_params.get('axis', None) if self.run_params is not None else 0
        if is_merge:
            self.output_data = self.input_merge(axis=axis)
        else:
            self.output_data = self.input_data
        self.is_finished = True
        from datasci.utils.mylog import get_stream_logger
        from datasci.workflow.config.log_config import log_level
        log = get_stream_logger("END NODE", level=log_level)
        log.info("Job finished at %s " % time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        return self.output_data


class DebugNode(BaseNode):

    def run(self):
        merge = self.input_merge()
        arg_input = self.run_params.get('input', None) if self.run_params is not None else merge
        self.output_data = merge + " " + arg_input
        self.is_finished = True
        return self.output_data


class PlaceholderNode(BaseNode):

    def run(self):
        self.output_data = self.input_merge()
        self.is_finished = True
        return self.output_data
