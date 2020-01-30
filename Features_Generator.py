class Features_Generator(object):
    
    '''Класс для преобразования имеющихся данных в два датасета признаков.
     Нужно лишь задать путь (path) к папке, где лежат данные и запустить финальную функцию.'''
    
    
    def __init__(self, path):
        self.products_path = path+'products.csv'
        self.purchases_path = path+'purchases.csv'
        self.clients_path = path+'clients.csv'
        self.train_path = path+'uplift_train.csv'
        self.test_path = path+'uplift_test.csv'  
  

    def features_dict(self, df_products):
        
        '''Функция создания словаря для создания признаков на основе купленных продуктов'''
        
        features_product=dict(set(zip(df_products['level_1'].unique(), np.zeros(len(df_products['level_1'].unique())))))
        for col in df_products.columns[2:6]:
            if col=='segment_id':
                features_product.update(dict(set(zip('segment_id_'+(df_products[col].astype(str).unique()), np.zeros(len(df_products[col].unique()))))))
            else:
                features_product.update(dict(set(zip(df_products[col].unique(), np.zeros(len(df_products[col].unique()))))))
        
        return features_product
   

    def transaction_max_delay(self, client):
        
        '''Функция для подсчета дней между первой и последней покупкой, где:
        client - данные по одному клиенту'''
        
        first_transaction = pd.to_datetime(client['transaction_datetime'].iloc[0])
        last_transatction = pd.to_datetime(client['transaction_datetime'].iloc[-1])
        day_delay = (last_transatction-first_transaction).days
        return day_delay
    
   

    def features_processing_from_purchases(self, purchases):
        
        '''Это функция извлечения признаков из данных о покупках. Данные групируются вокруг клиента'''
        
        all_client_n = purchases['client_id'].nunique()
        start_time_for_print = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        start_time = time.time()
        dict_product = dict(zip(df_products['product_id'].values, df_products.iloc[:, 1:6].values))
        dict_product_number = dict(zip(df_products['product_id'].values, df_products.iloc[:, -3:].values))
        print(f'Начало обработки клиентов {start_time_for_print}')
        features_product=self.features_dict(df_products)
        features_product['netto'] = 0
        features_product['trademark'] = 0
        features_product['alhocol'] = 0
        features_product['regular_points_received']=0
        features_product['express_points_received']=0
        features_product['regular_points_spent']=0
        features_product['express_points_spent']=0
        points_list = ['regular_points_received', 'express_points_received',\
                       'regular_points_spent', 'express_points_spent']
        data = []
        count =0
        for i, client in (purchases.groupby('client_id', sort=False)):
            count+=1
            if count%5000==0:
                clear_output()
                print(f'Начало обработки клиентов {start_time_for_print}')
                print(f'Обработано {count} клиентов из {all_client_n}...')
                print(f'C начало прошло {int(time.time()-start_time)} секунд')
                
            features = features_product.copy()
            n_trans = client['transaction_id'].nunique()
            features['transactions'] = n_trans
            features['sum'] = client['trn_sum_from_iss'].sum() 
            features['trn_sum_from_red'] = client['trn_sum_from_red'].sum()
            features['n_store'] = client['store_id'].nunique()
            features['n_product'] = client['product_id'].nunique()
            features['max_price'] = client['trn_sum_from_iss'].max()
            features['min_price'] = client['trn_sum_from_iss'].min()
            features['quantity'] = client['product_quantity'].sum()            
            features['first_buy_sum'] = client['purchase_sum'].iloc[0]
            features['last_buy_sum'] = client['purchase_sum'].iloc[-1]
            try:
                features['almost_last_buy'] = client['purchase_sum'].unique()[-2]
            except:
                features['almost_last_buy'] = client['purchase_sum'].unique()[0]

            features['client_id'] = client['client_id'].iloc[0]
            features['transaction_max_delay'] = self.transaction_max_delay(client)

            #Features from products
            count_products = Counter(client['product_id'])
            for product in count_products.keys():
                values=dict_product[product]
                for value in values:
                    if type(value)!=str:
                        features_product['segment_id_'+str(value)]=count_products[product]
                    else:
                        features_product[value]=count_products[product]

            temp_dict_quantity = dict(zip(client['product_id'], client['product_quantity']))
            for product, quantity in temp_dict_quantity.items():
                
                features['netto']+=quantity*dict_product_number[product][0] 
                features['trademark']+=quantity*dict_product_number[product][1] 
                features['alhocol']+=quantity*dict_product_number[product][2] 
                
            #Features from points
            points_dict = dict(zip(client['transaction_id'].values, client[points_list].values))
            for point in points_dict.values():
                features_product['regular_points_received']+=point[0]
                features_product['express_points_received']+=point[1]
                features_product['regular_points_spent']+=point[2]
                features_product['express_points_spent']+=point[3]
            
                
                
            #Average features
            features_product['avg_regular_points_received']=features_product['regular_points_received']/n_trans 
            features_product['avg_express_points_received']=features_product['express_points_received']/n_trans  
            features_product['avg_regular_points_spent']=features_product['regular_points_spent']/n_trans 
            features_product['avg_express_points_spent']=features_product['express_points_spent']/n_trans  
            features['avg_sum_from_red'] = features['trn_sum_from_red']/n_trans  
            features['avg_price_product'] = features['sum']/n_trans
            features['avg_delay_beetwen_transc'] = features['transaction_max_delay']/n_trans   
            features['avg_sum'] = features['sum']/n_trans
            features['avg_quantity'] = features['quantity']/n_trans
            features['avg_netto'] = features['netto']/n_trans
            features['avg_trademark'] = features['trademark']/n_trans
            features['avg_alhocol'] = features['alhocol']/n_trans
            
            data.append(features)
        features_set = pd.DataFrame(data)    
        clear_output()
        print(f'Начало обработки клиентов {start_time_for_print}')
        print(f'Обработка данных клиентов завершена за {int(time.time()-start_time)} секунд')
        
        return features_set
    
    def features_processing_from_clients_df(self, df_clients):
        df_clients['first_issue_unixtime'] = pd.to_datetime(df_clients['first_issue_date']).astype(np.int64)/10**9
        df_clients['first_redeem_unixtime'] = pd.to_datetime(df_clients['first_redeem_date']).astype(np.int64)/10**9
        df_features = pd.DataFrame({
                            'client_id': df_clients['client_id'],
                            'gender_M': (df_clients['gender'] == 'M').astype(int),
                            'gender_F': (df_clients['gender'] == 'F').astype(int),
                            'gender_U': (df_clients['gender'] == 'U').astype(int),
                            'age': df_clients['age'],
                            'first_issue_time': df_clients['first_issue_unixtime'],
                            'first_redeem_time': df_clients['first_redeem_unixtime'],
                            'issue_redeem_delay': df_clients['first_redeem_unixtime'] - df_clients['first_issue_unixtime'],
                        }).fillna(0)
        return df_features
    
    def predict_broke_age(self, features_set):
        '''Это функция тренирует модель на предсказание возраста и делает предсказание ошибочного возраста.
        Под ошибочным понимается тот, который выбивается из интервала 14-95 лет'''
        print('Запущена модель для корректировки неправильного возраста клиентов')
        broke_age_index = features_set[(features_set['age']<14) | (features_set['age']>95)].index
        
        X=features_set[~features_set.index.isin(broke_age_index)].drop(['age', 'client_id'], axis=1)
        y=features_set[~features_set.index.isin(broke_age_index)]['age']
        params = {
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1}
        train_set = lgb.Dataset(X.iloc[:-5000], y[:-5000])
        val_set = lgb.Dataset(X.iloc[-5000:], y[-5000:])
        model = lgb.train(params=params,
                             train_set = train_set,
                             valid_sets = [train_set, val_set],
                             num_boost_round=5000,
                             early_stopping_rounds=100,
                             verbose_eval=False
                             )
        age_predict = model.predict(features_set.loc[broke_age_index, :].drop(['age', 'client_id'], axis=1),\
                                    num_iteration = model.best_iteration)
        
        features_set.loc[broke_age_index, 'age']=age_predict.astype(int)
        clear_output()
        return features_set
    
    
    def create_features(self, low_memory=True, save_files=None):
        
        '''Финальная функция, которая запускает все остальные. Подгружает датасеты и перерабатывает их в
        train and test DataFrames. Low_memory предназначена для ПК с малым количеством оперативной памяти.
        Информация из датасета по покупкам подгружается постепенно.'''
        
        start_time = time.time()
        df_products=pd.read_csv(self.products_path).fillna('None')
        df_products.loc[df_products['netto']=='None', 'netto']=0
        df_clients = pd.read_csv(self.clients_path) 
        train = pd.read_csv(self.train_path) 
        test = pd.read_csv(self.test_path) 
        df_features = self.features_processing_from_clients_df(df_clients)
        if low_memory:
            purchases_chunks = pd.read_csv(self.purchases_path, chunksize=10**6)
            features_set = pd.DataFrame()
            for chunk in purchases_chunks:
                features_set = features_set.append(self.features_processing_from_purchases(chunk))
        else:
            features_set = self.features_processing_from_purchases(pd.read_csv(self.purchases_path))
            
        features_set = pd.merge(features_set, df_features, how='inner')
        features_set = self.predict_broke_age(features_set)
        
        features_set_train = pd.merge(features_set, train, how='inner')
        features_set_test = pd.merge(features_set, test, how='inner') 
        
        if save_files:
            features_set_train.to_csv(path+'features_set_train', index=False)
            features_set_test.to_csv(path+'features_set_test', index=False)
            print(f'Файлы успешно сохранены в директорию {path}')
            
        print(f'Весь процесс обработки данных занял {int(time.time()-start_time)} секунд')
        print(f'Тренировочный датасет состоит из обработанных данных {features_set_train.shape[0]} клиентов и {features_set_train.shape[1]-2} сгенерированных признаков')
        print(f'Тестовый датасет состоит из обработанных данных {features_set_test.shape[0]} клиентов и {features_set_test.shape[1]} сгенерированных признаков')
        return features_set_train, features_set_test
        
