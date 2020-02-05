import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import numpy as np
from IPython.core.display import clear_output
from Models_Fundaments import Models_Fundaments

class All_Models(Models_Fundaments):
    '''Основной класс, в котором тренируются модели для uplitf моделирования. Основан на классе Models_Fundaments'''
    
    def start_training_xgb(self, target):
        
        ''' Бинарный классификатор основанный на xgb. Делает предсказание на train, test или val в зависимости от выбранного режима.
            Важный момент, что предсказание на train set делается для всех данных, за исключением тех, на которых модель обучалась, 
            т.е. train set может служишь ориентиром в подсчете uplift score'''
        
        
        '''Сначала модели генерируются для данных, где treatment_flg==0'''
        X = self.train[self.train['treatment_flg']==0].drop(self.cols_to_drop, axis=1)
        y = self.train[self.train['treatment_flg']==0][target].values
        xgb_models_control=[]
        predict_control=[]
        predict_train_0 = []
        pred_temp = []

        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            print(f'{fold+1} fold в процессе...')
            print('------------------------')
            dtrain = xgb.DMatrix(X.iloc[train_ids], y[train_ids])
            dval = xgb.DMatrix(X.iloc[val_ids], y[val_ids])
            model = xgb.train(params=self.xgb_params,
                                          dtrain=dtrain,
                                          evals=[(dtrain, 'train'), (dval, 'val')],
                                          verbose_eval=200
                                         )
            pred = model.predict(xgb.DMatrix(X.iloc[val_ids]))
            predict_train_0 = np.concatenate([pred, predict_train_0]) 
            pred_temp.append(model.predict(xgb.DMatrix(self.train[self.train['treatment_flg']==1].drop(self.cols_to_drop, axis=1))))                                    
            if self.mode=='val':
                predict_control.append(model.predict(xgb.DMatrix(self.val.drop(self.cols_to_drop, axis=1)))) 
            elif self.mode == 'test':
                predict_control.append(model.predict(xgb.DMatrix(self.test.drop(self.drop_columns_test, axis=1))))
            
            xgb_models_control.append(model)
        predict_train_0 = np.concatenate([predict_train_0, np.asarray(pred_temp).mean(axis=0)])      
        clear_output()
        
        '''Во второй части модели берут данные, где treatment_flg==1'''
        X = self.train[self.train['treatment_flg']==1].drop(self.cols_to_drop, axis=1)
        y = self.train[self.train['treatment_flg']==1][target].values
        xgb_models_treatment=[]
        predict_treatment=[]
        predict_train_1 = []
        pred_temp = []
        
        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            print(f'{fold+1} fold в процессе...')
            print('------------------------')
            dtrain = xgb.DMatrix(X.iloc[train_ids], y[train_ids])
            dval = xgb.DMatrix(X.iloc[val_ids], y[val_ids])
            model = xgb.train(params=self.xgb_params,
                                          dtrain=dtrain,
                                          evals=[(dtrain, 'train'), (dval, 'val')],
                                          verbose_eval=200,
                                          early_stopping_rounds = 100
                                         )
            pred = model.predict(xgb.DMatrix(X.iloc[val_ids]))
            predict_train_1 = np.concatenate([pred, predict_train_1])
            pred_temp.appen(model.predict(xgb.DMatrix(self.train[self.train['treatment_flg']==0].drop(self.cols_to_drop, axis=1))))                 
            if self.mode=='val':
                predict_treatment.append(model.predict(xgb.DMatrix(self.val.drop(self.cols_to_drop, axis=1)))) 
            elif self.mode == 'test':
                predict_treatment.append(model.predict(xgb.DMatrix(self.test.drop(self.drop_columns_test, axis=1))))
            xgb_models_treatment.append(model)
        predict_train_1 = np.concatenate([predict_train_1, np.asarray(pred_temp).mean(axis=0)])           
        clear_output()
            
        if self.mode == 'val':
            treatment = self.val['treatment_flg']
            target = self.val['target']
            xgb_predict_uplift = (np.asarray(predict_treatment) - np.asarray(predict_control)).mean(axis=0)
            xgb_score = self.uplift_score(xgb_predict_uplift,  treatment, target)
            print(f'uplift score---{xgb_score}')
            
            return predict_control, predict_treatment, predict_train_0, predict_train_1
        
        elif self.mode == 'test':
            xgb_predict_uplift = (np.asarray(predict_treatment) - np.asarray(predict_control)).mean(axis=0)
            
            return predict_control, predict_treatment, predict_train_0, predict_train_1, xgb_predict_uplift
        

    def start_training_lgb(self, target):   

        ''' Бинарный классификатор основанный на xgb. Делает предсказание на train, test или val в зависимости от выбранного режима.
            Важный момент, что предсказание на train set делается для всех данных, за исключением тех, на которых модель обучалась, 
            т.е. train set может служишь ориентиром в подсчете uplift score'''
        
        
        '''Сначала модели генерируются для данных, где treatment_flg==0'''                           
        X = self.train[self.train['treatment_flg']==0].drop(self.cols_to_drop, axis=1)
        y = self.train[self.train['treatment_flg']==0][target].values


        lgb_models_control = []
        predict_control = []
        predict_train_0 = []
        pred_temp = []

        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            print(f'{fold+1} fold в процессе...')
            print('------------------------')
            train_set = lgb.Dataset(X.iloc[train_ids], y[train_ids])
            val_set = lgb.Dataset(X.iloc[val_ids], y[val_ids])
            model = lgb.train(params=self.lgb_params,
                                              train_set=train_set,
                                              valid_sets=[train_set, val_set],
                                              verbose_eval=200,
                                              early_stopping_rounds = 100
                                             )
            pred = model.predict((X.iloc[val_ids]), num_iteration=model.best_iteration)
            predict_train_0 = np.concatenate([pred, predict_train_0])
            pred_temp.append(model.predict((self.train[self.train['treatment_flg']==1].drop(self.cols_to_drop, axis=1)), num_iteration=model.best_iteration))     
            if self.mode=='val':
                predict_control.append(model.predict(self.val.drop(self.cols_to_drop, axis=1), num_iteration=model.best_iteration)) 
            elif self.mode == 'test':
                predict_control.append(model.predict(self.test.drop(self.drop_columns_test, axis=1), num_iteration=model.best_iteration))
            lgb_models_control.append(model)       
        predict_train_0 = np.concatenate([predict_train_0, np.asarray(pred_temp).mean(axis=0)])  
        clear_output()
        
        
        '''Во второй части модели берут данные, где treatment_flg==1'''
        X = self.train[self.train['treatment_flg']==1].drop(self.cols_to_drop, axis=1)
        y = self.train[self.train['treatment_flg']==1][target].values

        lgb_models_treatment = []
        predict_treatment = []
        predict_train_1 = []
        pred_temp = []

        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            print(f'{fold+1} fold в процессе...')
            print('------------------------')
            train_set = lgb.Dataset(X.iloc[train_ids], y[train_ids])
            val_set = lgb.Dataset(X.iloc[val_ids], y[val_ids])
            model = lgb.train(params=self.lgb_params,
                                              train_set=train_set,
                                              valid_sets=[train_set, val_set],
                                              verbose_eval=200,
                                              early_stopping_rounds = 100
                                             )
            pred = model.predict((X.iloc[val_ids]), num_iteration=model.best_iteration)
            predict_train_1 = np.concatenate([pred, predict_train_1])
            pred_temp.append(model.predict((self.train[self.train['treatment_flg']==0].drop(self.cols_to_drop, axis=1)), num_iteration=model.best_iteration))     
            if self.mode=='val':
                predict_treatment.append(model.predict(self.val.drop(self.cols_to_drop, axis=1), num_iteration=model.best_iteration)) 
            elif self.mode == 'test':
                predict_treatment.append(model.predict(self.test.drop(self.drop_columns_test, axis=1), num_iteration=model.best_iteration))
                  
            lgb_models_treatment.append(model)
        predict_train_1 = np.concatenate([predict_train_1, np.asarray(pred_temp).mean(axis=0)])  
        clear_output()
        
        if self.mode == 'val':
            treatment = self.val['treatment_flg']
            target = self.val['target']
            lgb_predict_uplift = (np.asarray(predict_treatment) - np.asarray(predict_control)).mean(axis=0)
            lgb_score = self.uplift_score(lgb_predict_uplift,  treatment, target)
            print(f'uplift score---{lgb_score}')
            
            return predict_control, predict_treatment, predict_train_0, predict_train_1
        
        elif self.mode == 'test':
            lgb_predict_uplift = (np.asarray(predict_treatment) - np.asarray(predict_control)).mean(axis=0)
            
            return predict_control, predict_treatment, predict_train_0, predict_train_1, lgb_predict_uplift

     
            
    def start_training_cat(self, target):
        
        ''' Бинарный классификатор основанный на xgb. Делает предсказание на train, test или val в зависимости от выбранного режима.
            Важный момент, что предсказание на train set делается для всех данных, за исключением тех, на которых модель обучалась, 
            т.е. train set может служишь ориентиром в подсчете uplift score'''
        
        
        
        '''Сначала модели генерируются для данных, где treatment_flg==0'''
        X = self.train[self.train['treatment_flg']==0].drop(self.cols_to_drop, axis=1)
        y = self.train[self.train['treatment_flg']==0][target].values


        cat_models_control = []
        predict_control = []
        predict_train_0 = []
        pred_temp = []
        
        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            print(f'{fold+1} fold в процессе...')
            print('------------------------')
            dtrain = cat.Pool(X.iloc[train_ids], y[train_ids])
            dval = cat.Pool(X.iloc[val_ids], y[val_ids])
            model = cat.train(params=self.cat_params,
                                  dtrain=dtrain,
                                  eval_set=dval,
                                  verbose=200,
                                  early_stopping_rounds=100
                                 )
            pred = model.predict(cat.Pool(X.iloc[val_ids]))
            predict_train_0 = np.concatenate([pred, predict_train_0])
            pred_temp.append(model.predict(cat.Pool(self.train[self.train['treatment_flg']==1].drop(self.cols_to_drop, axis=1))))                 
            if self.mode=='val':
                predict_control.append(model.predict(cat.Pool(self.val.drop(self.cols_to_drop, axis=1)))) 
            elif self.mode == 'test':
                predict_control.append(model.predict(cat.Pool(self.test.drop(self.drop_columns_test, axis=1))))    
                             
            cat_models_control.append(model)
        predict_train_0 = np.concatenate([predict_train_0, np.asarray(pred_temp).mean(axis=0)])        
        clear_output()

        '''Во второй части модели берут данные, где treatment_flg==1'''
        X = self.train[self.train['treatment_flg']==1].drop(self.cols_to_drop, axis=1)
        y = self.train[self.train['treatment_flg']==1][target].values

        cat_models_treatment = []
        predict_treatment = []
        predict_train_1 = []
        pred_temp = []
        
        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            print(f'{fold+1} fold в процессе...')
            print('------------------------')
            train_set = cat.Pool(X.iloc[train_ids], y[train_ids])
            val_set = cat.Pool(X.iloc[val_ids], y[val_ids])
            model = cat.train(params=self.cat_params,
                                              dtrain=train_set,
                                              eval_set=val_set,
                                              verbose=200,
                                              early_stopping_rounds=100

                                             )
            pred = model.predict(cat.Pool(X.iloc[val_ids]))
            predict_train_1 = np.concatenate([pred, predict_train_1])    
            pred_temp.append(model.predict(cat.Pool(self.train[self.train['treatment_flg']==0].drop(self.cols_to_drop, axis=1))))        
            if self.mode=='val':
                predict_treatment.append(model.predict(cat.Pool(self.val.drop(self.cols_to_drop, axis=1)))) 
            elif self.mode == 'test':
                predict_treatment.append(model.predict(cat.Pool(self.test.drop(self.drop_columns_test, axis=1))))        
                      
            cat_models_treatment.append(model)
       
        predict_train_1 = np.concatenate([predict_train_1, np.asarray(pred_temp).mean(axis=0)])  
        clear_output()

        if self.mode == 'val':
            treatment = self.val['treatment_flg']
            target = self.val['target']
            cat_predict_uplift = (np.asarray(predict_treatment) - np.asarray(predict_control)).mean(axis=0)
            cat_score = self.uplift_score(cat_predict_uplift,  treatment, target)
            print(f'uplift score---{cat_score}')
            
            return predict_control, predict_treatment, predict_train_0, predict_train_1
        
        elif self.mode == 'test':
            cat_predict_uplift = (np.asarray(predict_treatment) - np.asarray(predict_control)).mean(axis=0)
            
            return predict_control, predict_treatment, predict_train_0, predict_train_1, cat_predict_uplift

      
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                                     