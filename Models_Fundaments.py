import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import numpy as np
from sklearn.model_selection import StratifiedKFold


class Models_Fundaments(object):
    '''Класс, в который закладывается фундамент для модели: параметры, число cv и вспомогательные функции'''
    
    def __init__(self, path_to_file, NFOLDS = 5, xgb_params=None, lgb_params=None, cat_params=None):
        
        self.features_train = pd.read_csv(path_to_file)
        self.folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        if xgb_params:
            self.xgb_params = xgb_params
        else:
            self.xgb_params = {'booster':'gbtree',
                                'objective': "binary:logistic", 
                               'early_stopping_rounds':100,
                                'eval_metric':'rmse',
                                'colsample_bytree': 0.34131772233697516,
                                 'gamma': 0.7413397793676475,
                                 'l1': 1.1853720119095321,
                                 'l2': 0.6289283526816473,
                                 'learning_rate': 0.07840692470132621,
                                 'max_depth': 5,
                                 'min_child_weight': 2.6569914098925027,
                                 'min_gain_to_split': 0.12486151508017662,
                                 'num_leaves': 850,
                                 'subsample': 0.6800110130757757,
                                  'n_estimators': 2000}

        if lgb_params:
            self.lgb_params = lgb_params
        else:
            self.lgb_params = {
                            'n_jobs': -1,
                            'seed': 42,
                            'boosting_type': 'gbdt',
                            'objective': 'binary',
                            'metric': 'auc',
                            'learning_rate': 0.005,
                            'min_child_weight': 18.86380721048759,

                             'feature_fraction':0.9128088855957803,
                            'bagging_fraction': 0.8926918914437931,
                            'learning_rate': 0.025,
                            'max_depth': 13,
                            'num_leaves': 924,
                            'min_gain_to_split': 0.09270624702126036,
                            'min_child_weight': 1.8886380721048759,
                            'lambda_l1': 1.6448761550547633,
                            'lambda_l2':1.1863869490262469,
                            'subsample': 0.937203344164974}
            
        if cat_params:
            self.cat_params = cat_params
        else:
            self.cat_params = {'eval_metric': 'AUC',
                                'random_seed': 42,
                              'n_estimators': 2000}
            
    def preproccesing(self, i=0):
        X = self.features_train.iloc[:-5000][self.features_train['treatment_flg']==i].iloc[:, :-2].drop('client_id', axis=1)
        y = self.features_train.iloc[:-5000][self.features_train['treatment_flg']==i].iloc[:,-1].values
        X_val = self.features_train.iloc[-5000:, :-2].drop('client_id', axis=1)
        y_val = self.features_train.iloc[-5000:,-1].values
        y_tr = self.features_train.iloc[-5000:,-2].values
        
        return X,y, X_val, y_val, y_tr
    
    def uplift_score(self, prediction, treatment, target, rate=0.3):
        """
        Подсчет Uplift Score
        """
        order = np.argsort(-prediction)
        treatment_n = int((treatment == 1).sum() * rate)
        treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
        control_n = int((treatment == 0).sum() * rate)
        control_p = target[order][treatment[order] == 0][:control_n].mean()
        score = treatment_p - control_p
        return score
    
    
    
    