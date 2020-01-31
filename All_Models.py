import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import numpy as np
from Models import Models_Fundaments
class All_Models(Models_Fundaments):
    '''Основной класс, в котором тренируются модели. Основан на классе Models_Fundaments'''
    
    def start_training_xgb(self):
        X,y, X_val, y_val, y_tr = self.preproccesing()


        xgb_models_control=[]
        predict_control=[]

        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            dtrain = xgb.DMatrix(X.iloc[train_ids], y[train_ids])
            dval = xgb.DMatrix(X.iloc[val_ids], y[val_ids])
            model = xgb.train(params=self.xgb_params,
                                          dtrain=dtrain,
                                          evals=[(dtrain, 'train'), (dval, 'val')],
                                          verbose_eval=200
                                         )
            predict_control.append(model.predict(xgb.DMatrix(X_val)))
        xgb_models_control.append(model)

        X,y, X_val, y_val, y_tr = self.preproccesing(i=1)

        xgb_models_treatment=[]
        predict_treatment=[]

        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            dtrain = xgb.DMatrix(X.iloc[train_ids], y[train_ids])
            dval = xgb.DMatrix(X.iloc[val_ids], y[val_ids])
            model = xgb.train(params=self.xgb_params,
                                          dtrain=dtrain,
                                          evals=[(dtrain, 'train'), (dval, 'val')],
                                          verbose_eval=200

                                         )
            predict_treatment.append(model.predict(xgb.DMatrix(X_val)))               
        xgb_models_treatment.append(model)

        xgb_predict_uplift = np.asarray(predict_treatment) - np.asarray(predict_control)
        xgb_score = self.uplift_score(xgb_predict_uplift,  y_tr, y_val)

        return xgb_predict_uplift
                                    
    def start_training_lgb(self):                              
        X,y, X_val, y_val, y_tr = self.preproccesing()


        lgb_models_control=[]
        predict_control=[]

        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            train_set = lgb.Dataset(X.iloc[train_ids], y[train_ids])
            val_set = lgb.Dataset(X.iloc[val_ids], y[val_ids])
            model = lgb.train(params=self.lgb_params,
                                              train_set=train_set,
                                              valid_sets=[train_set, val_set],
                                              verbose_eval=200
                                             )
            predict_control.append(model.predict((X_val), num_iteration=model.best_iteration))
        lgb_models_control.append(model)

        X,y, X_val, y_val, y_tr = self.preproccesing(i=1)

        lgb_models_treatment=[]
        predict_treatment=[]

        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            train_set = lgb.Dataset(X.iloc[train_ids], y[train_ids])
            val_set = lgb.Dataset(X.iloc[val_ids], y[val_ids])
            model = lgb.train(params=self.lgb_params,
                                              train_set=train_set,
                                              valid_sets=[train_set, val_set],
                                              verbose_eval=200

                                             )
            predict_treatment.append(model.predict((X_val), num_iteration=model.best_iteration))             
        lgb_models_treatment.append(model)

        lgb_predict_uplift = np.asarray(predict_treatment) - np.asarray(predict_control)
        lgb_score = self.uplift_score(lgb_predict_uplift,  y_tr, y_val)
        return lgb_predict_uplift  
            
    def start_training_cat(self):
        X,y, X_val, y_val, y_tr = self.preproccesing()


        cat_models_control=[]
        predict_control=[]
        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            dtrain = cat.Pool(X.iloc[train_ids], y[train_ids])
            dval = cat.Pool(X.iloc[val_ids], y[val_ids])
            model = cat.train(params=self.cat_params,
                                  dtrain=dtrain,
                                  eval_set=dval,
                                  verbose=200,
                                  early_stopping_rounds=100
                                 )
                                             
            predict_control.append(model.predict(cat.Pool(X_val)))
        cat_models_control.append(model)

        X,y, X_val, y_val, y_tr = self.preproccesing(i=1)

        cat_models_treatment=[]
        predict_treatment=[]
        for fold, (train_ids, val_ids) in enumerate(self.folds.split(X,y)):
            train_set = cat.Pool(X.iloc[train_ids], y[train_ids])
            val_set = cat.Pool(X.iloc[val_ids], y[val_ids])
            model = cat.train(params=self.cat_params,
                                              dtrain=train_set,
                                              eval_set=dval,
                                              verbose=200,
                                              early_stopping_rounds=100

                                             )
            predict_treatment.append(model.predict(cat.Pool(X_val)))             
        cat_models_treatment.append(model)

        cat_predict_uplift = np.asarray(predict_treatment) - np.asarray(predict_control)
        cat_score = self.uplift_score(cat_predict_uplift,  y_tr, y_val)
        return cat_predict_uplift 
    
    def uplift_score_calculation(self):
        X,y, X_val, y_val, y_tr = self.preproccesing(i=1)
        xgb_score = self.start_training_xgb()
        lgb_score = self.start_training_lgb()
        cat_score = self.start_training_cat()
        
        uplift_score = (xgb_score+lgb_score+cat_score)/3
        uplift_score_2 = (xgb_score*0.2+lgb_score*0.5+cat_score*0.3)
        score = self.uplift_score(uplift_score,  y_tr, y_val)
        score_2 = self.uplift_score(uplift_score_2,  y_tr, y_val)
        return score, score_2
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                                     