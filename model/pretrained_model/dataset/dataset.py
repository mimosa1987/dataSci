# coding:utf8

import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset


class TextSequenceDataset(Dataset):
    def __init__(self, df, tokenizer, mode='train', max_seq_len=512,
                 text_field_name='x', label_field_name='y', x_post_func=None, y_post_func=None):
        self.mode = mode
        self.max_seq_len = max_seq_len

        self.tokenizer = tokenizer

        self.origin_data = df
        self.X = self.origin_data[text_field_name].values
        if mode == 'train':
            self.y = self.origin_data[label_field_name].values.reshape(-1, 1)
        del self.origin_data

        self.x_post_func = x_post_func
        self.y_post_func = y_post_func

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = dict()

        x = str(self.X[idx])

        if self.x_post_func is not None:
            x = self.x_post_func(x)
        if len(x) > self.max_seq_len:
            x = x[:self.max_seq_len]

        if self.mode == 'train':
            y = self.y[idx]
            if self.y_post_func is not None:
                y = self.y_post_func(y)
            sample['label'] = torch.LongTensor(y)

        tokens = self.tokenizer.tokenize(x)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        sample['input_ids'] = torch.LongTensor(input_ids)

        return sample


def create_mini_batch(samples, return_dict=True, mode='train'):
    """

    Args:
        samples:

    Returns:

    """
    input_ids = [s['input_ids'] for s in samples]

    input_ids = pad_sequence(input_ids, batch_first=True)

    attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
    attention_mask = attention_mask.masked_fill(input_ids != 0, 1)

    if mode == 'infer':
        if return_dict:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        return input_ids, attention_mask

    label = [s['label'] for s in samples]
    if return_dict:
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': torch.tensor(label)}
    return input_ids, attention_mask, torch.tensor(label)
