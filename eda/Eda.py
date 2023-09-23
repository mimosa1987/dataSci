# coding:utf8

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from copy import deepcopy

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

warnings.filterwarnings("ignore")


def data_dist_info(dataset, feature_names=None, show_only=False):
    """
       查看数据分布
       Owner:wangyue29
     Args:
       dataset: 用来被统计的数据集
        feature_names: 特征名称，默认为None，即使用所有特征
        show_only: 是否只展示结果

    Returns:
        dict变量，统计的结果
        若show_only=True，则不返回数据
     """
    # 判断是否给定了列名称
    if feature_names is not None:
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        dataset = deepcopy(dataset)
        dataset = dataset[feature_names]

    categorical_feature_names = list(dataset.select_dtypes(include=['object', 'category', 'bool', 'string']).columns)
    numberical_feature_names = list(dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns)
    if not show_only:
        cate_dict = {}
        numb_dict = {}
        # 统计离散特征的值分布情况
        for cat_fea in categorical_feature_names:
            cate_dict[cat_fea] = {k: v for k, v in zip(dataset[cat_fea].value_counts().index.to_list(),
                                                       dataset[cat_fea].value_counts().to_list())}
        # 统计数值特征的值分布情况
        for num_fea in numberical_feature_names:
            skew = dataset[num_fea].skew()
            kurt = dataset[num_fea].skew()
            numb_dict[num_fea] = {'skew': skew, 'kurt': kurt}

        # 将离散、连续特征存储到一个变量
        data_dist_dict = dict()
        data_dist_dict['categorical'] = cate_dict
        data_dist_dict['numberical'] = numb_dict
        return data_dist_dict
    else:
        print('--- 离散特征分布--- ')
        for cat_fea in categorical_feature_names:
            print(cat_fea + '的特征分布如下')
            print('{}特征有个{}不同值'.format(cat_fea, dataset[cat_fea].nunique()))
            print(dataset[cat_fea].value_counts() + '\n')

        print('\n')

        print('--- 连续特征分布--- ')
        for numberical_fea in numberical_feature_names:
            print(numberical_fea + '的特征分布如下')
            print('{:15}'.format(numberical_fea),
                  'Skewness:{:05.2f}'.format(dataset[numberical_fea].skew()),
                  '',
                  'Kurtosis:{:06.2f}'.format(dataset[numberical_fea].kurt())
                  )

        print('\n')
        print('--- dataset describe ---')
        print(dataset.describe())
        print('\n')
        print('--- dataset info ---')
        print(dataset.info(), '\n')


def null_value_info(dataset, feature_names=None, show_only=False):
    """
       空值数、率信息统计
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认为False，即使用所有特征名称
       show_only: 是否只展示数据，默认False

     Returns:
        返回空值统计信息结果
        若show_only=True，则不返回数据
     Owner:wangyue29
     """

    if feature_names is None:
        feature_names = dataset.columns
    elif isinstance(feature_names, str):
        feature_names = [feature_names]

    null_value_info_dict = dict()
    for feat_name in feature_names:
        null_value = dataset[feat_name].isnull().sum()
        null_value_ratio = null_value * 1.0 / dataset[feat_name].shape[0]
        if not show_only:
            null_value_info_dict[feat_name] = {'count': null_value, 'ratio': null_value_ratio}
        else:
            print('特征名称:{},空值数:{},空值率:{:0.2f}'.format(feat_name, null_value, null_value_ratio))
    if len(null_value_info_dict) > 0:
        return null_value_info_dict


def target_dist_info(dataset, target='label', is_bin_cls=True, show_only=False):
    """
       label分布信息统计
     Args:
       dataset: dataframe数据集
       target: 目标变量target值,默认'label'
       is_bin_cls: 是否为二分类问题，如果为True则label为0或1，默认为False。
       show_only: 是否只展示数据，默认False

     Returns:
        返回正、负样本分布信息结果

     Owner:wangyue29
     """
    if is_bin_cls:
        pos_size = dataset[target].sum()
        neg_size = dataset[target].shape[0] - pos_size
        if not show_only:
            return {0: neg_size, 1: pos_size}
        else:
            print('正样本数:{},负样本数:{},负样本数/正样本数:{:0.2f}'.format(pos_size, neg_size, pos_size / neg_size))
    else:
        label_dict = dict()
        labels = dataset[target].nunique()
        for label in labels:
            label_dict[label] = np.sum(dataset[target] == label)

        return label_dict


def plot_categorical_feature_bar_chart(dataset, feature_names=[], hue=None, f_rows=1, f_cols=2, palette=None):
    """
       离散特征条形图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认可自动识别离散特征
       hue: 在x或y标签划分的同时，再以hue标签划分统计个数
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
       palette: 使用不同的调色板，默认是None
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['object', 'category', 'bool', 'string']).columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.countplot(x=feat_name, hue=hue, data=dataset, palette=palette)
        plt.title('variable={}'.format(feat_name))
        plt.xlabel('')

    plt.tight_layout()
    plt.show()


def plot_numberical_feature_hist(dataset, feature_names=[], f_rows=1, f_cols=2, kde=True, rotation=30):
    """
       连续特征直方图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表,默认可自动识别连续特征
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
       kde:KDE分布，默认值True
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """
    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.distplot(dataset[feat_name], fit=stats.norm, kde=kde)
        plt.title('variable={}'.format(feat_name))
        plt.xlabel('')

        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        res = stats.probplot(dataset[feat_name], plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(dataset[feat_name])))

    plt.tight_layout()
    plt.xticks(rotation=rotation)
    plt.show()


def plot_numberical_feature_hist_without_qq_chart(dataset, kde=False, feature_names=[], rotation=0):
    """
       连续特征直方图可视化【无Q-Q图】
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认可自动识别连续特征
       rotation:横坐标值旋转角度，默认是0
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    def dist_plot(x, **kwargs):
        sns.distplot(x, kde=kde)
        plt.xticks(rotation=0)

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    f = pd.melt(dataset, value_vars=feature_names)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
    g.map(dist_plot, "value")


def plot_numberical_feature_corr_heatmap(dataset, feature_names=[]):
    """
       连续特征相关热力图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称列表，默认可自动识别连续特征
     Returns:
        可视化呈现结果

     Owner:wangyue29
    """
    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    corr = dataset[feature_names].corr()

    f, ax = plt.subplots(figsize=(7, 7))
    plt.title('Correlation of Numberical Features', y=1, size=16)
    sns.heatmap(corr, annot=True, square=True, vmax=1.0, vmin=-1.0,
                linewidths=.5, annot_kws={'size': 12, 'weight': 'bold', 'color': 'blue'})


def plot_linear_reg_corr(dataset, feature_names=[], target='label', f_rows=1, f_cols=2, is_display_distplot=False):
    """
       线性回归关系图可视化
     Args:
       dataset: dataframe数据集
       feature_names: 特征名称,默认可自动识别连续特征
       target: 目标变量target值,默认'label'
       f_rows: 图行数，默认值1
       f_cols: 图列数，默认值2
     Returns:
        可视化呈现结果

     Owner:wangyue29
     """

    if 0 == len(feature_names):
        feature_names = dataset.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns

    if 1 == f_rows and 0 != len(feature_names):
        f_rows = len(feature_names)

    plt.figure(figsize=(6 * f_cols, 6 * f_rows))

    idx = 0
    for feat_name in feature_names:
        idx += 1
        ax = plt.subplot(f_rows, f_cols, idx)
        sns.regplot(x=feat_name, y=target, data=dataset, ax=ax,
                    scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                    line_kws={'color': 'k'})

        plt.title('variable=' + '{}'.format(feat_name))
        plt.xlabel('')
        plt.ylabel(target)

        if is_display_distplot:
            idx += 1
            ax = plt.subplot(f_rows, f_cols, idx)
            sns.distplot(dataset[feat_name].dropna())
            plt.xlabel(feat_name)


def plot_figure_combination_chart(x, y_list, y_name, color='pink'):
    """
       绘制柱状图与折线图工具
     Args:
       x: x轴列表
       y_list: y轴列表
       y_name: y轴名称
       color: 柱状图颜色 默认'pink'
     Returns:
        图像

     Owner:baijiaqi1
     """
    y1 = y_list[0]
    plt.figure(figsize=(14, 7))
    plt.bar(x, y1, color=color, alpha=0.8)
    for a, b in zip(x, y1):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=14)
    plt.twinx()
    for i in range(1, len(y_list)):
        y = y_list[i]
        label_name = y_name[i - 1]
        plt.plot(x, y, alpha=1, marker='.', linewidth=2, label=label_name)
        for a, b in zip(x, y):
            plt.text(a, b, '%.1f' % (b * 100) + '%', ha='center', va='bottom', fontsize=14)
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    plt.title('特征用户量与转化率')
    plt.xlabel("特征分段")
    plt.ylabel("转化率")
    plt.show()


def plot_feature_target_relationship_chart(dataset, feature_name, target='label', feature_type='categorical', bins=None,
                                           qcut=5, ascending=False, plot_color='pink'):
    """
       单特征与目标值关系图表
     Args:
       dataset: 数据集
       feature_name: 列名
       target: 目标值列名 默认 'label'
       feature_type: categorical 离散 numberical 连续 默认'categorical'
       bins: 自定义分箱方式 默认'None'
       qcut: 等频分箱个数
       ascending: 离散排序方式， 默认转化率降序
       plot_color: 柱状图颜色，默认'pink'
     Returns:
        单特征与目标值关系图表,与关系详情表

     Owner:baijiaqi1
     """
    if feature_type == 'categorical':
        df_all_group = dataset.groupby(feature_name).agg(
            {target: [('用户量', lambda x: len(x)), ('转化量', lambda x: sum(x))]})
        df_all_group['转化率'] = df_all_group[target, '转化量'] / df_all_group[target, '用户量']
        df_all_group = df_all_group.sort_values(by='转化率', ascending=ascending)
    elif feature_type == 'numberical':
        if bins != None:
            df_tag_cut = pd.cut(dataset[feature_name], bins=bins)
            df_all_group = dataset.groupby([df_tag_cut]).agg(
                {target: [('用户量', lambda x: len(x)), ('转化量', lambda x: sum(x))]})
            df_all_group['转化率'] = df_all_group[target, '转化量'] / df_all_group[target, '用户量']
        elif qcut != None:
            df_tag_cut = pd.qcut(dataset[feature_name], q=qcut, duplicates="drop")
            df_all_group = dataset.groupby([df_tag_cut]).agg(
                {target: [('用户量', lambda x: len(x)), ('转化量', lambda x: sum(x))]})
            df_all_group['转化率'] = df_all_group[target, '转化量'] / df_all_group[target, '用户量']

    plot_figure_combination_chart(df_all_group.index.astype(str), [df_all_group[target, '用户量'], df_all_group['转化率']],
                                  ['转化率'], plot_color)
    return df_all_group


def plot_numberical_feature_violin(dataset, target='label', feature_names=None, is_label=1, x=None, y=None,
                                   scale="area", palette=None, gridsize=100):
    """
       连续值绘制小提琴图
     Args:
       dataset: 数据集
       target: 目标值
       feature_names: 特征名列表
       is_label: 是否基于label值绘制 1/0, 默认1
       x:自定义小提琴x
       y:自定义小提琴y
       scale: 测度小提琴图的宽度 默认'area' area-面积相同,count-按照样本数量决定宽度,width-宽度一样
       palette: 设置调色板 默认None
       gridsize: 设置小提琴图的平滑度，越高越平滑 默认100
     Returns:
        小提琴图可视化呈现结果

     Owner:baijiaqi1
     """
    if isinstance(feature_names, list) and is_label == 1:
        plot_num = int(len(feature_names))
        fig, axes = plt.subplots(plot_num, 1)
        index = 0
        for i in feature_names:
            sns.violinplot(y=i, x=target, data=dataset, scale=scale, palette=palette, gridsize=gridsize, ax=axes[index])
            index += 1
    elif isinstance(feature_names, list) and is_label != 1:
        plot_num = int(len(feature_names))
        fig, axes = plt.subplots(plot_num, 1)
        index = 0
        for i in feature_names:
            sns.violinplot(y=i, data=dataset, scale=scale, palette=palette, gridsize=gridsize, ax=axes[index])
            index += 1
    elif x and y:
        sns.violinplot(x=x, y=y, data=dataset, scale=scale, palette=palette, gridsize=gridsize)

    else:
        return "请输入特征"


def plot_numberical_feature_box(dataset, feature_names=None, target=None, width=5, height=5, box_width=0.2):
    """
       绘制箱线图
     Args:
       dataset: 数据集
       feature_names: 特征列表or单特征
       target: 类别变量or因变量
       width: 图像宽度
       height: 图像高度
       box_width: 箱宽
     Returns:
        箱线图可视化呈现结果

     Owner:yujie5
     """
    plt.figure(figsize=(width, height))
    if isinstance(feature_names, list) and isinstance(target, str):
        n = 0
        for i in feature_names:
            fig = plt.figure(num=n, figsize=(width, height))
            sns.boxplot(x=target, y=i, data=dataset, width=box_width)
            n += 1
    elif isinstance(feature_names, list):
        sns.boxplot(data=dataset[feature_names], width=box_width)
    else:
        sns.boxplot(x=target, y=feature_names, data=dataset, width=box_width)
