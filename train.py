import pandas as pd
import numpy as np
import lightgbm as lgb
from feature_select import train_val_split
from dataload import dataload
import gc
from sklearn.model_selection import GridSearchCV
import pickle
train, questions, lectures = dataload()

X_train, y_train, X_test, y_test = train_val_split(train, questions, lectures)



######################################################################
parameters = {
              'max_depth': [15, 20, 25, 30, 35],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'cat_smooth': [1, 10, 15, 20, 35]
}

model_lgb = lgb.LGBMClassifier(max_depth=3,
                    learning_rate=0.1,
                    n_estimators=200, # 使用多少个弱分类器
                    num_class=2,
                    booster='gbtree',
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0,
                    reg_lambda=1,
                    seed=0 # 随机数种子
                )
###################################################################

#####################################################################
params = {
    'num_leaves': [13,14,15],
#     'max_depth': [-1,4,6,8],
#     'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
#     'n_estimators':[10,15,20],
#     'min_child_samples':[15,20,25],
#     'subsample':[0.4,0.5,0.6,0.7],
#     'colsample_bytree':[0.4,0.5,0.6,0.7],
#     'reg_alpha':[0,1,2,3,5,8],
#     'reg_lambda':[7,8,9,10],
#     'num_iterations':[30,40,50],
#     'min_data_in_leaf': [30, 50, 100, 300, 400],
#     'cat_smooth':[150,160,170,180,190]
}
other_params = {
    'max_depth' : 4,
    'num_leaves': 15,
    'learning_rate': 0.07,
    'cat_smooth':180,
    'num_iterations':100,
    'colsample_bytree': 0.7,
    'subsample': 0.4,
    'reg_alpha':3,
    'reg_lambda':9,
}
model_lgb = lgb.LGBMRegressor(**other_params)

################################################################



optimized_lgb = GridSearchCV(estimator=model_lgb, param_grid=params, scoring='roc_auc', cv=5, verbose=1, n_jobs=2)
optimized_lgb.fit(X_train, y_train, categorical_feature=category_feature)
print('参数的最佳取值：{0}'.format(optimized_lgb.best_params_))
print('最佳模型得分:{0}'.format(optimized_lgb.best_score_))
print(optimized_lgb.cv_results_['mean_test_score'])
print(optimized_lgb.cv_results_['params'])

##########################################################
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'num_leaves': 9,
#     'learning_rate': 0.3,
#     'feature_fraction_seed': 2,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'min_data': 20,
#     'min_hessian': 1,
#     'verbose': -1,
#     'silent': 0
# }
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# gbm = lgb.train(
#     params,
#     lgb_train,
#     num_boost_round=10000,
#     valid_sets=lgb_eval,
#     early_stopping_rounds=20
# )
# with open('myfeature/lgb.pickle', 'wb') as fw:
#     pickle.dump(gbm, fw)

##################################################################
