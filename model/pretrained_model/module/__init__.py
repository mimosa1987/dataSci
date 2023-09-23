# coding:utf8

import torch
from transformers import *
from torch import nn
from torch.optim import *

model_lib = {
    'cls': BertForSequenceClassification,
    'tag': BertForTokenClassification
}


def bert_downstream_model(pretrained_model, num_cls, model_type='cls', return_dict=True):
    """

    Args:
        pretrained_model: str value, pretrained model's name or local directory
        num_cls: int value, number of categories
        model_type: str value, downstream task type, default 'cls', current only support 'cls'、'tag'
        return_dict: bool value, whether return dict type value, default True

    Returns:
        bert model
        bert tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model_class = model_lib.get(model_type)
    model = model_class.from_pretrained(pretrained_model, num_labels=num_cls, return_dict=return_dict)
    cudas = torch.cuda.device_count()
    if cudas == 0:
        print('load bert cls model to CPU')
        return model, tokenizer
    else:
        if cudas >= 1:
            print('load bert cls model to GPU')
            model = nn.DataParallel(model)
        return model.cuda(), tokenizer


opti_dict = {'Adam': Adam, 'SGD': SGD, 'Adadelta': Adadelta, 'Adagrad': Adagrad, 'RMSprop': RMSprop, 'LBFGS': LBFGS}


def get_model_outputs(data_batch, model, mode='train', label_field_name='label'):
    """

    Args:
        data_batch:
        model:
        mode:
        label_field_name:

    Returns:

    """
    cudas = torch.cuda.device_count()
    if cudas > 0:
        torch.cuda.empty_cache()

    if isinstance(data_batch, dict):
        items = {var_name: var_value.cuda() if cudas > 0 else var_value for var_name, var_value in
                 data_batch.items()}
        # items = {var_name: var_value for var_name, var_value in data_batch.items()}

        outputs = model(**items)
        logits = outputs.logits
        if mode == 'infer':
            return logits
        elif mode == 'test':
            labels = data_batch[label_field_name]
            return logits, labels

        loss = outputs.loss
        labels = data_batch[label_field_name]
    else:
        items = [t.cuda() if cudas > 0 else t for t in data_batch]
        outputs = model(*items)
        logits, loss = outputs
        labels = data_batch[-1]

    return logits, loss, labels
