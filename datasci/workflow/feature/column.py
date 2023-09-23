from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

def column_change(encoder, input_columns, change_feature_map):
    """
    Get change of the encoder which changed the columns
    :param encoder: encoder
    :param input_columns: the columns before chenged
    :param change_feature_map: changed map of features
    :return: the changed cloumns ,and feature map
    """

    final_cols = list()
    if isinstance(encoder, OneHotEncoder):
        for i in range(len(input_columns)):
            tmp_cols = ['%s_%s' % (input_columns[i], x) for x in encoder.categories_[i]]
            change_feature_map[input_columns[i]] = tmp_cols
            final_cols.extend(tmp_cols)
    elif isinstance(encoder, KBinsDiscretizer) and (encoder.encode in ['onehot', 'onehot-dense']):
        for i in range(len(input_columns)):
            col = input_columns[i]
            if isinstance(encoder.n_bins, int):
                k = encoder.n_bins
            else:
                k = encoder.n_bins[i]
            tmp_cols = list()
            for j in range(k):
                tmp_cols.append('%s_%s' % (col, j))
            change_feature_map[col] = tmp_cols
            final_cols.extend(tmp_cols)
    else:
        final_cols = input_columns
    return final_cols, change_feature_map