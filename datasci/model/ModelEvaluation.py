# coding:utf8

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.model_selection import learning_curve, ShuffleSplit, validation_curve
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import shap

shap.initjs()

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

warnings.filterwarnings("ignore")


def feature_importances(estimator=None, X=None, thresholds=0.01, palette=None):
    """
      预估器输出特征重要性
    Args:
      estimator: 预估器实例
      X: 样本集
      thresholds: 特征重要性阈值，默认是0.01
      palette: 使用不同的调色板，默认是None

    Returns:
       返回特征重要性，按照降序排序

    Owner:wangyue29
    """
    importance_feature = list(zip(X.columns, estimator.feature_importances_))
    importance_feature = pd.DataFrame(importance_feature, columns=['feature_name', 'importances'])

    importance_feature = importance_feature.loc[importance_feature['importances'] >= thresholds]
    importance_feature = importance_feature.sort_values(by='importances', ascending=False)
    sns.set_style("darkgrid")
    sns.barplot(y='feature_name', x='importances', data=importance_feature, orient='h', palette=palette)
    plt.show()

    return importance_feature


def feature_importances_shap(estimator=None, X=None, plot_type="bar"):
    """
      预估器通过SHAP框架输出特征重要性
    Args:
      estimator: 预估器实例
      X: 样本集
      plot_type: 绘制图表类型，默认是条形图

    Returns:
       返回特征重要性，按照降序排序

    Owner:wangyue29
    """
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type=plot_type)


def feature_shap_value(estimator=None, X=None, feature_names=[]):
    """
      每个特征的SHAP值
    Args:
      estimator: 预估器实例
      X: 样本集
      feature_names: 特征名称列表

    Returns:
       返回每个特征的SHAP值，按照降序排序

    Owner:wangyue29
    """
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)

    if 0 == len(feature_names):
        feature_names = X.columns

    shap.summary_plot(shap_values, pd.DataFrame(X, columns=feature_names))


def single_feature_explainer(estimator=None, X=None):
    """
     单特征的各自有其贡献
   Args:
     estimator: 预估器实例
     X: 样本集

   Returns:
      返回单特征的贡献度

   Owner:wangyue29
   """
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True)


def feature_interaction_value(estimator=None, X=None):
    """
     组合特征区分度
   Args:
     estimator: 预估器实例
     X: 样本集

   Returns:
      返回组合特征可视化呈现

   Owner:wangyue29
   """
    explainer = shap.TreeExplainer(estimator)
    shap_interaction_values = explainer.shap_interaction_values(X)
    shap.summary_plot(shap_interaction_values, X)


def evaluate(estimator=None, X=None, y=None, logits=None, average='macro'):
    """

    Args:
        estimator: 预估器实例
        X: 样本集
        y: 目标变量
        logits: 模型输出的概率
        average: average类型，macro和micro，默认macro

    Returns:
        返回预估器评估指标字典

    Owner:wangyue29
    """
    assert estimator is not None or logits is not None, ValueError('estimator和logits不能同时为空')
    assert y is not None, ValueError('y不能为空')
    assert logits is not None or X is not None, ValueError('当logits为空时X不能为空')

    if estimator is not None:
        pred_proba = estimator.predict_proba(X)[:, 1]
        pred = estimator.predict(X)
    else:
        pred_proba = logits[:, 1]
        pred = np.argmax(logits, axis=-1)

    evaluate_dict = {}
    evaluate_dict['auc'] = '%.3f' % metrics.roc_auc_score(y, pred_proba, average=average)
    evaluate_dict['f1'] = '%.3f' % metrics.f1_score(y, pred, average=average)
    evaluate_dict['recall'] = '%.3f' % metrics.recall_score(y, pred, average=average)
    evaluate_dict['precision'] = '%.3f' % metrics.precision_score(y, pred, average=average)
    evaluate_dict['accuracy'] = '%.3f' % metrics.accuracy_score(y, pred)
    evaluate_dict['ture_rate'] = '%.3f' % (pred.sum() / pred.shape[0])
    evaluate_dict['positive'] = '%s' % pred.sum()

    return evaluate_dict


def evaluate_seq(labels, logits, mask=None, average='macro', return_dict=True):
    """
        序列token级别分类模型评估
        Args:
            labels: 标签列表
            logits: 模型输出概率列表
            mask: 序列的mask
            average: 评估方式，包含'micro'和'macro'，默认为'macro'
            return_dict: 是否返回dict格式，默认为True

        Returns:
            返回各指标的评估结果
        """
    all_labels = [x for x in range(logits.shape[-1])]
    preds = np.argmax(logits, axis=-1)

    if mask is not None:
        acc = np.sum((preds == labels) * mask.numpy()) / np.sum(mask.numpy())
    else:
        acc = np.mean(preds == labels)

    seq_p, seq_r, seq_auc, seq_f1 = 0, 0, 0, 0
    for idx, seq_logits in enumerate(logits):
        seq_preds = preds[idx]
        seq_labels = labels[idx]

        seq_p += precision_score(seq_labels, seq_preds, labels=all_labels[1:], average=average, zero_division=0)
        seq_r += recall_score(seq_labels, seq_preds, labels=all_labels[1:], average=average, zero_division=0)

        seq_labels_onehot = label_binarize(seq_labels, all_labels)
        try:
            seq_auc += roc_auc_score(y_true=seq_labels_onehot, y_score=seq_logits, average=average)
        except ValueError as e:
            seq_auc += 0
        seq_f1 += f1_score(seq_labels, seq_preds, average=average)

    p, r, auc, f1 = \
        seq_p / logits.shape[0], seq_r / logits.shape[0], seq_auc / logits.shape[0], seq_f1 / logits.shape[0]

    if return_dict:
        return {'acc': acc, 'p': p, 'r': r, 'auc': auc, 'f1': f1}
    else:
        return acc, p, r, auc, f1


def roc(estimator=None, X=None, y=None):
    """
      roc 曲线
    Args:
      estimator: 预估器实例
      X: 样本集
      y: 目标变量
    Returns:
       返回roc曲线

    Owner:wangyue29
    """
    pred_proba = estimator.predict_proba(X)[:, 1]

    fpr, tpr, threshold = roc_curve(y, pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_learning_curve(estimator=None, X=None, y=None, ylim=(0.5, 1.01), cv=None,
                        n_splits=10, test_size=0.2, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
      学习曲线
    Args:
      estimator: 预估器实例
      X: 样本集
      y: 目标变量
      ylim: y轴坐标范围，默认从0.5到1.01
      cv: 交叉样本量，默认为空后，利用ShuffleSplit方法获取
      n_splits: 划分训练集、测试集的次数，默认为10，ShuffleSplit方法参数
      test_size: 测试集比例或样本数量，默认为0.2，ShuffleSplit方法参数
      n_jobs: CPU并行核数，默认为1，-1的时候，表示cpu里的所有core进行工作
      train_sizes: 训练样本比例，默认[0.1,0.325,0.55,0.775,1.]

    Returns:
       返回绘制学习曲线

    Owner:wangyue29
    """
    plt.figure()

    title = r"Learning Curve ({0})".format(estimator.__class__.__name__)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if cv is None:
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_validation_curve(estimator=None, X=None, y=None, param_name="gamma",
                          param_range=np.logspace(-6, -1, 5), scoring="accuracy",
                          n_jobs=1, ylim=(0.0, 1.01), verbose=0):
    """
      验证曲线
    Args:
      estimator: 预估器实例
      X: 样本集
      y: 目标变量
      param_name: 参数名称，默认gamma
      param_range: 训练样本比例，默认从0.0到1.01
      scoring: 评分函数，默认accuracy
      n_jobs: CPU并行核数，默认为1，-1的时候，表示cpu里的所有core进行工作
      ylim: y轴坐标范围，默认从0.5到1.01
      verbose: 是否打印调试信息，默认不打印

    Returns:
       返回绘制验证曲线

    Owner:wangyue29
    """

    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name,
                                                 param_range=param_range,
                                                 scoring=scoring, n_jobs=n_jobs, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    title = r"Validation Curve ({0})".format(estimator.__class__.__name__)
    plt.title(title)

    plt.xlabel(r"$\{0}$".format(param_name))
    plt.ylabel("Score")

    if ylim is not None:
        plt.ylim(*ylim)

    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


def plot_train_test_dataset_feature_dist(X_train=None, X_test=None, feature_names=[], f_rows=1, f_cols=2):
    """
       训练集、测试集的特征分布可视化
     Args:
       X_train: 训练集
       X_test: 测试集
       feature_names: 特征名称,默认可自动识别连续特征
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = X_test.columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)

        ax = sns.kdeplot(X_train[feat_name], color='Red', shade=True)
        ax = sns.kdeplot(X_test[feat_name], color='Green', shade=True)

        ax.set_xlabel(feat_name)
        ax.set_ylabel('Frequency')
        ax = ax.legend(['train', 'test'])

    plt.show()
