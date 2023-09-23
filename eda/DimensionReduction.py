# coding: utf8

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def dim_reduce(X, y=None, method='pca', n_components=2., pretrain_data=None, return_model=False, n_iter=8,
               perplexity=30):
    """
    Args:
        X: 需要降维的数据
        y: 标签数据
        method: 降维使用的方法
            可选方法有：pca、truncated_svd、lda、tsne
        n_components: 目标维度
        pretrain_data: 预训练模型所使用的数据（非TSNE时使用）
        return_model: 是否返回模型（非TSNE时使用）
        n_iter: 迭代次数（仅PCA时使用）
        perplexity: 困惑度（仅TSNE时使用）

    Returns:
        X：降维后的数据
        model：降维算法被训练后的模型
    """
    model_dict = {
        'pca': PCA(n_components=n_components, copy=True),
        'truncated_svd': TruncatedSVD(n_components=n_components, n_iter=n_iter),
        'lda': LinearDiscriminantAnalysis(n_components=n_components),
        'tsne': TSNE(n_components=n_components, perplexity=perplexity)
    }
    model = model_dict[method]

    if method == 'tsne':
        return model.fit_transform(X)

    if pretrain_data is not None:
        model.fit(pretrain_data, y) if method == 'lda' else model.fit(pretrain_data)
    else:
        model.fit(X, y) if method == 'lda' else model.fit(X)
    X = model.transform(X)

    return X, model if return_model else X
