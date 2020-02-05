import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class Models_Fundaments(object):
    
    '''Класс, в который закладывается фундамент для модели: параметры, число cv и вспомогательные функции.
    Если указан путь, то с него подгружаются файлы. Имена файлов должны быть features_train.csv, features_val.csv, 
    features_test.csv для train, val, test соответсвенно.
    Если путь не указан, то файлы надо задать на прямую для self.train, self.val и self.test'''
    
    def __init__(self, cols_to_drop, NFOLDS = 5,  path_to_files=None, xgb_params=None, lgb_params=None, cat_params=None, mode='val',):
        
        self.cols_to_drop = cols_to_drop
        self.mode=mode
        
        if path_to_files:
            self.train = pd.read_csv(path_to_files+'features_train.csv')            
            self.val = pd.read_csv(path_to_files+'features_val.csv')            
            if self.mode=='test':
                self.test = pd.read_csv(path_to_files+'features_test.csv')
                self.train = pd.concat([self.val, self.train], sort=False)
        
        if self.mode == 'test':
            try:
                test_columns = self.test.columns.to_list()
                self.drop_columns_test = [col for col in cols_to_drop if col in test_columns]
            except:
                print('Тестовый датасет не загружен и drop_columns_test не определены. Установите напрямую.')
        self.folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        
        if xgb_params:
            self.xgb_params = xgb_params
        else:
            self.xgb_params = {'booster':'gbtree',
                                'objective': "binary:logistic", 
                               
                                'eval_metric':'rmse',
                                'colsample_bytree': 0.34131772233697516,
                                 'gamma': 0.7413397793676475,
                                 'l1': 1.1853720119095321,
                                 'l2': 1.6289283526816473,
                                 'learning_rate': 0.07840692470132621,
                                 'max_depth': 5,
                                 'min_child_weight': 50,
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
                            'learning_rate': 0.02,
                            'min_child_weight': 50,
                        
                             'feature_fraction':0.9128088855957803,
                            'bagging_fraction': 0.8926918914437931,
                            #'learning_rate': 0.025,
                            'max_depth': 13,
                            'num_leaves': 924,
                            'min_gain_to_split': 0.09270624702126036,
                            'n_estimators': 2000,
                            'lambda_l1': 1.6448761550547633,
                            'lambda_l2':1.4,
                            'subsample': 0.937203344164974}
            
        if cat_params:
            self.cat_params = cat_params
        else:
            self.cat_params = {'eval_metric': 'AUC',
                                'random_seed': 42,
                              'n_estimators': 2000,
                              'eta': 0.02}
            
    def create_classes(self):
        
        '''Функция для создания дополнительно таргета за счет классификации клиентов в зависимости от treatment_flg и target'''
        
        def classses(treatment_flg, target):
            if (treatment_flg==0) & (target==0):
                n_class=0
            elif (treatment_flg==0) &( target==1):
                n_class=1
            elif (treatment_flg==1) & (target==0):
                n_class=2       
            else:
                n_class=3
            return n_class
        
        self.train['classes']=self.train.apply(lambda x: classses(x['treatment_flg'], x['target']), axis=1)
        
        if self.mode=='val':
            self.val['classes']=self.val.apply(lambda x: classses(x['treatment_flg'], x['target']), axis=1)
            
        if self.mode=='train':
            self.train['classes']=self.train.apply(lambda x: classses(x['treatment_flg'], x['target']), axis=1)
   
        self.cols_to_drop.append('classes')
        
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
    
    
    
    