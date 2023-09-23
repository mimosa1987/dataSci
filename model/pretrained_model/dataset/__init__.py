# coding:utf8

from torch.utils.data import DataLoader
from functools import partial
from .dataset import create_mini_batch


def get_data_loader(dataset_class, tokenizer, data, mode='train', batch_size=32, max_seq_len=512, test_data=None,
                    return_dict=True, label_field_name='y', text_field_name='x'):
    """

    Args:
        dataset_class:
        tokenizer:
        data:
        batch_size:
        max_seq_len:
        test_data:
        return_dict:
        label_field_name:
        text_field_name:

    Returns:

    """
    train_data = dataset_class(data, tokenizer, max_seq_len=max_seq_len, mode=mode,
                               label_field_name=label_field_name,
                               text_field_name=text_field_name)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              collate_fn=partial(create_mini_batch, return_dict=return_dict, mode=mode))

    if test_data is not None:
        test_data = dataset_class(test_data, tokenizer, max_seq_len=max_seq_len, mode=mode,
                                  label_field_name=label_field_name,
                                  text_field_name=text_field_name)
        test_loader = DataLoader(test_data, batch_size=batch_size,
                                 collate_fn=partial(create_mini_batch, return_dict=return_dict, mode=mode))
        return train_loader, test_loader
    return train_loader
