import xgboost as xgb
import joblib


class TalXGBClassifier(xgb.XGBClassifier):

    def tal_load_model(self, model_file):
        print(model_file)

    def tal_save_model(self, model_file):
        joblib.dump(self, model_file)

    def tal_train(self, X_train, y_train, X_val, y_val, pre_model, **params):
        self.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)], xgb_model=pre_model, **params)
        return self

    def tal_predict(self, data, **params):
        return self.predict(data, **params)

    def tal_predict_proba(self, data, **params):
        return self.predict_proba(data, **params)


class TalXgboost(xgb.Booster):
    def __init__(self):
        super(TalXgboost, self).__init__()
        self.model = None

    def tal_load_model(self, model_file):
        self.model = self.load_model(model_file)
        return self.model

    def tal_save_model(self, model_file):
        self.save_model(model_file)

    def tal_train(self, X_train, y_train, X_val, y_val, pre_model, **params):
        train_data = xgb.DMatrix(X_train, y_train)
        val_data = xgb.DMatrix(X_val, y_val)
        self.model = xgb.train(params=params, dtrain=train_data, evals=[(val_data, 'eval'), (train_data, 'train')],
                               xgb_model=pre_model)
        return self

    def tal_predict(self, data, **params):
        return self.model.predict(data, **params)
