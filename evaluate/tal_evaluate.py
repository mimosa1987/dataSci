from sklearn import metrics


class CommonEvaluate(object):
    def __init__(self, ):
        pass

    def evaluate(self, data):
        metrics_dict = dict()
        label = data['label']
        cls_data = data['cls']
        proba_data = data.drop(['label', 'cls'], axis=1)
        metrics_dict['auc'] = metrics.roc_auc_score(label, proba_data.iloc[:, 1])
        metrics_dict['recall'] = metrics.recall_score(label, cls_data)
        metrics_dict['precision'] = metrics.precision_score(label, cls_data)
        metrics_dict['f1_score'] = metrics.f1_score(label, cls_data)
        return metrics_dict
