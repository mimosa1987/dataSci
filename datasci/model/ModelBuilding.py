# coding:utf8

import warnings

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV  # pip install scikit-optimize
from xgboost import XGBClassifier
import xgboost as xgb
import os
import copy
import numpy as np
from sklearn import metrics
from tqdm import tqdm, tqdm_notebook

warnings.filterwarnings("ignore")

# 默认预估器列表
default_estimators = [('LR', {}), ('XGB', {})]

# 预估器简称与预估器实体映射关系
estimator_name_mapping = {
    'NB': MultinomialNB(alpha=0.01),
    'DT': DecisionTreeClassifier(random_state=42),
    'LR': LogisticRegression(penalty='l2'),
    'KNN': KNeighborsClassifier(),
    'RFC': RandomForestClassifier(random_state=42),
    'SVC': SVC(kernel='rbf', gamma='scale'),
    'ADA': AdaBoostClassifier(),
    'GBDT': GradientBoostingClassifier(),
    "XGB": XGBClassifier(),
    "LGB": LGBMClassifier()
}


def select_best_estimator(estimators=[], X=None, y=None, scoring='roc_auc', cv=5, verbose=0):
    """
       选择最佳预估器，基于scoring评分排名
     Args:
       estimators: 候选预估器列表
       X: 样本集
       y: 目标变量
       scoring: 评分函数，默认roc_auc
       cv: 交叉验证，默认是5
       verbose: 是否打印调试信息，默认不打印

     Returns:
        返回最好的预估器，预估器评分结果集

     Owner:wangyue29
     """
    estimator_result = dict()
    best_estimator = None
    best_score = 0.0

    if 0 == len(estimators):
        estimators = default_estimators

    for estimator in estimators:
        estimator_name = estimator[0]
        estimator_params = estimator[1]
        estimator = estimator_name_mapping[estimator_name]

        if estimator is None:
            print('wrong estimator name!')

        if 0 != len(estimator_params):
            print(estimator_params)
            estimator.set_params(**estimator_params)

        estimator.fit(X, y)

        estimator_full_name = estimator.__class__.__name__
        scores = cross_val_score(estimator, X, y, verbose=0, cv=cv, scoring=scoring)
        score_avg = scores.mean()
        estimator_result[estimator_full_name] = score_avg

        if 1 == verbose:
            print('Cross-validation of : {0}'.format(estimator_full_name))
            print('{0} {1:0.2f} (+/- {2:0.2f})'.format(scoring, score_avg, scores.std()))

        if score_avg >= best_score:
            best_estimator = estimator
            best_score = score_avg

    print('Best Model: {0},Score:{1:0.2f}'.format(best_estimator.__class__.__name__, best_score))

    result = pd.DataFrame({
        'Model': [i for i in estimator_result.keys()],
        'Score': [i for i in estimator_result.values()]})

    result.sort_values(by='Score', ascending=False)
    return best_estimator, result


def grid_search_optimization(estimator, param_grid={}, X=None, y=None, scoring='roc_auc', cv=5, verbose=0):
    """
       预估器优化-网格搜索
     Args:
       estimator: 预估器实例
       param_grid: 预估器参数
       X: 样本集
       y: 目标变量
       scoring: 评分函数，默认roc_auc
       cv: 交叉验证，默认是5
       verbose: 是否打印调试信息，默认不打印

     Returns:
        返回最好的预估器，预估器评分结果集

     Owner:wangyue29
     """

    cross_validation = StratifiedKFold(n_splits=cv)

    grid_search = GridSearchCV(
        estimator,
        scoring=scoring,
        param_grid=param_grid,
        cv=cross_validation,
        n_jobs=-1,
        verbose=verbose)

    grid_search.fit(X, y)
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    return parameters


def randomized_search_optimization(estimator, param_grid={}, X=None, y=None, n_iter=30, verbose=0):
    """
       预估器优化-随机搜索
     Args:
       estimator: 预估器实例
       param_grid: 预估器参数
       X: 样本集
       y: 目标变量
       n_iter: 迭代次数，默认30次
       verbose: 是否打印调试信息，默认不打印

     Returns:
        返回最好的预估器，预估器评分结果集

     Owner:wangyue29
     """

    randomized_search = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter, random_state=42, verbose=verbose)
    randomized_search.fit(X, y)
    parameters = randomized_search.best_params_

    print('Best score: {}'.format(randomized_search.best_score_))
    print('Best parameters: {}'.format(randomized_search.best_params_))

    return parameters


def bayesian_search_optimization(estimator, param_grid={}, X=None, y=None, n_iter=30, verbose=0):
    """
       预估器优化-贝叶斯搜索
     Args:
       estimator: 预估器实例
       param_grid: 预估器参数
       X: 样本集
       y: 目标变量
       n_iter: 迭代次数，默认30次
       verbose: 是否打印调试信息，默认不打印

     Returns:
        返回最好的预估器，预估器评分结果集

     Owner:wangyue29
     """

    bayes_search = BayesSearchCV(estimator, param_grid, n_iter=n_iter, random_state=42, verbose=verbose)
    bayes_search.fit(X, y)
    parameters = bayes_search.best_params_

    print('Best score: {}'.format(bayes_search.best_score_))
    print('Best parameters: {}'.format(bayes_search.best_params_))

    return parameters


def train(estimator_name='XGB', estimator_params={}, X=None, y=None):
    """
       训练预估器【支持分类、回归、聚类等】
     Args:
       estimator_name: 预估器名称，默认是XGB
       estimator_params: 预估器参数
       X: 样本集
       y: 目标变量

     Returns:
        返回训练后的预估器实例

     Owner:wangyue29
     """

    estimator = estimator_name_mapping[estimator_name]

    if estimator is None:
        print('wrong estimator name!')

    if 0 != len(estimator_params):
        estimator.set_params(**estimator_params)

    estimator.fit(X, y)

    return estimator


def train_random_neg_sample(X, y, X_test=None, y_test=None, neg_lbl_value=0, estimator_name='XGB',
                            estimator_params=None, eps=4, params_post_process_func=None,
                            num_steps_per_epoch=128, metrics_weight=None, checkpoint=False,
                            saved_model_name=None):
    """
        随机负采样模型训练，按比例对负样本进行随机采样，与正样本一起加入模型训练。
        训练过程中，每个step均会去采样正例样本数*eps量的负样本，同时将模型的正例权重设为eps。
        评估指标为各个标准指标的加权平均，即auc、f1-score、recall、precision的加权结果，权重由用户指定。
            如果想使用单个指标或者某些非全部指标的组合，则在指标权重字典参数中设置想参与计算的指标的权重，其它的不设置。例如：
                metrics_weight={'auc':1.1,'recall':1}
    Args:
        X:  ndarray对象，训练集特征
        y:  ndarray对象，训练集标签
        X_test:  ndarray对象，测试集特征，可以为空
        y_test:  ndarray对象，测试集标签，可以为空
        neg_lbl_value:  负标签值，默认为0
        estimator_name:  str值，模型名称
        estimator_params:  dict值，模型参数
        eps:  采样比率，默认为4
        params_post_process_func:  参数的后处理函数
        num_steps_per_epoch:  int值，训练步数
        metrics_weight: dict对象，各个指标的自定义权重，用以评估最优模型
        checkpoint: bool值，是否保存最优模型，默认False
        saved_model_name: str值，保存模型文件的名称，默认为None

    Returns:
        训练好的模型
    """

    def calc_score(y_, X_, estimator):
        predict = estimator.predict_proba(X_)
        pred_cate = np.argmax(predict, axis=1)

        current_step_metrics = {'auc': metrics.roc_auc_score(y_, predict[:, 1]),
                                'f1-score': metrics.f1_score(y_, pred_cate),
                                'precision': metrics.precision_score(y_, pred_cate),
                                'recall': metrics.recall_score(y_, pred_cate)}

        if metrics_weight is None:
            score = sum(current_step_metrics.values())
        else:
            score = 0
            for k, v in current_step_metrics.items():
                score += v * metrics_weight.get(k, 0)
        return score, current_step_metrics

    if params_post_process_func is not None:
        estimator_params = params_post_process_func(estimator_params)

    if estimator_name != 'XGB0.7':
        estimator = estimator_name_mapping[estimator_name]
        if estimator_params is not None:
            estimator.set_params(**estimator_params)
    else:
        estimator = xgb
        if estimator_params is not None:
            estimator.Booster.set_param(**estimator_params)

    X_pos = X[(y != neg_lbl_value).reshape(-1)]
    X_neg = X[(y == neg_lbl_value).reshape(-1)]

    y_pos = y[(y != neg_lbl_value).reshape(-1)]
    y_neg = y[(y == neg_lbl_value).reshape(-1)]

    if "JPY_PARENT_PID" in os.environ:
        bar = tqdm_notebook(range(num_steps_per_epoch), ncols=700)
    else:
        bar = tqdm(range(num_steps_per_epoch), ncols=700)

    best_score = 0
    best_metrics = None

    for _ in bar:
        idxs = np.random.choice(range(X_neg.shape[0]), int(X_pos.shape[0] * eps), replace=False)
        X_neg_ = X_neg[idxs]
        y_neg_ = y_neg[idxs]
        X_ = np.concatenate([X_pos, X_neg_], axis=0)
        y_ = np.concatenate([y_pos, y_neg_], axis=0)
        Xy = np.concatenate([X_, y_], axis=1)
        np.random.shuffle(Xy)
        X_ = Xy[:, :-1]
        y_ = Xy[:, -1:]

        if estimator_name != 'XGB0.7':
            estimator.fit(X_, y_)

            if X_test is not None and y_test is not None:
                score, current_step_metrics = calc_score(y_test, X_test, estimator)
            else:
                score, current_step_metrics = calc_score(y, X, estimator)

            if score > best_score:
                best_score = score
                best_metrics = current_step_metrics
                if checkpoint:
                    if saved_model_name is None:
                        saved_model_name = estimator_name + '.bin'
                    estimator.save_model(saved_model_name)
        else:
            model = estimator.train(estimator_params, xgb.DMatrix(data=X_, label=y_),
                                    evals=(xgb.DMatrix(data=X_test, label=y_test)))

            score, current_step_metrics = calc_score(y_test, X_test, model)

            if score > best_score:
                best_score = score
                best_metrics = current_step_metrics
                if checkpoint:
                    if saved_model_name is None:
                        saved_model_name = estimator_name + '.bin'
                    model.save_model(saved_model_name)

        bar.set_description(
            'f1-score:%.3f | auc:%.3f | r:%.3f | p:%.3f' % (best_metrics['f1-score'], best_metrics['auc'],
                                                            best_metrics['recall'], best_metrics['precision']))

    return estimator


def save(estimmator=None, filename=None, compress=3, protocol=None):
    """
       模型保存
     Args:
       estimator_name: 预估器名称
       filename: 模型保存路径，文件格式支持‘.z’, ‘.gz’, ‘.bz2’, ‘.xz’ or ‘.lzma’
       compress: 压缩数据等级，0-9，值越大，压缩效果越好，但会降低读写效率，默认建议是3
       protocol: pickle protocol,

     Returns:
        返回存储数据文件列表
    """
    filenames = joblib.dump(estimmator, filename, compress, protocol)

    return filenames
