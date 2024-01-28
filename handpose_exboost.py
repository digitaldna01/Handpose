""" Import python package """
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib as plt
import pandas as pd

""" Data Interface """
input_array = np.genfromtxt('./Handpose/x.csv', delimiter=',')
output_array = np.genfromtxt('./Handpose/y.csv', delimiter=',')
test_array = np.genfromtxt('./Handpose/test.csv', delimiter=',')

""" Build XGBOOST DMatrix """
x_train, x_test, y_train, y_test = train_test_split(input_array, output_array, test_size=0.2)
# x_test, x_val, y_test, y_val  = train_test_split(x_test, y_test, test_size=0.15, random_state=42)

dtrain = xgb.DMatrix(x_train, y_train)
# dval = xgb.DMatrix(x_val, y_val)
dtest = xgb.DMatrix(x_test, y_test)

print("Train data Quantitiy: ", dtrain.num_row())
# print("Validation data Quantitiy: ", dval.num_row())
print("Test data Quantitiy: ",  dtest.num_row())

df = pd.DataFrame(y_test)
df.to_csv("y_test.csv", header=False, index=False)

df = pd.DataFrame(x_test)
df.to_csv("x_test.csv", header=False, index=False)

""" USE XGBOOSTREGRESSOR MODEL """
""" Setting Parameters and Create Model """
from xgboost.callback import EarlyStopping

learningRate = 0.05
booster = 'gbtree'
numBoostRound = 400
maxDepth = 2
minChildWeight = 5
gamma = 0
subsample = 0.95
colsampleByTree = 0.75
scalePosWeight = 1
earlyStoppingRounds = 100
evalMetric = mean_squared_error
regAlpha = 0.01

xgbR = XGBRegressor(n_estimators = numBoostRound,
                    booster = booster,
                    learning_rate = learningRate,
                    max_depth = maxDepth,
                    gamma = gamma,
                    subsample = subsample,
                    min_child_weight = minChildWeight,
                    colsample_bytree = colsampleByTree,
                    scale_pos_weight = scalePosWeight,
                    reg_alpha = regAlpha,
                    eval_metric = evalMetric)

""" Train Model """
##########
xgbR.fit(x_train, y_train)
score = xgbR.score(x_train, y_train)
print("Training score: ", score)
# Prediction with train data
testScore = xgbR.score(x_test, y_test)
print("Testing score: ", testScore)

""" Prediction with test data """
prediction =xgbR.predict(x_test)

mse = mean_squared_error(y_test, prediction)
rmse = mean_squared_error(y_test, prediction, squared=False)
mae = mean_absolute_error(y_test, prediction )
print(f"MSE of the base model: {mse}")
print(f"RMSE of the base model: {rmse}")
print(f"MAE of the base model: {mae}")

###########
# test_predict = xgbR.predict(test_array)
###########

###########
# df = pd.DataFrame(prediction)
# df.to_csv("test_prediction.csv", header=False, index=False)
###########

###########
# df = pd.DataFrame(test_predict)
# df.to_csv("test_predict.csv", header=False, index=False)
###########

""" Parameter Test 2 - Max Depth & Min Child Weight """
# ParamTest2 = {
#     'max_depth': range(3, 10, 2),
#     'min_child_weight': range(1, 6, 2)
# }

# xgbR2 = XGBRegressor(learning_rate = learningRate,
#                      n_estimators = numBoostRound,
#                      max_depth = maxDepth,
#                      min_child_weight= minChildWeight,
#                      gamma = gamma,
#                      subsample = subsample,
#                      colsample_bytree = colsampleByTree,
#                      scale_pos_weight = scalePosWeight,
#                      seed=27)

# gsearch2 = GridSearchCV(estimator = xgbR2,
#                         param_grid = ParamTest2,
#                         scoring='neg_root_mean_squared_error',
#                         n_jobs=4,
#                         cv=5)

# gsearch2.fit(x_train,y_train)

# print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)

""" Parameter Test 2 - extend """
# ParamTest2 = {
#     'max_depth': [2, 3, 4],
#     'min_child_weight': [4, 5, 6]
# }

# xgbR2 = XGBRegressor(learning_rate = learningRate,
#                      n_estimators = numBoostRound,
#                      max_depth = maxDepth,
#                      min_child_weight= minChildWeight,
#                      gamma = gamma,
#                      subsample = subsample,
#                      colsample_bytree = colsampleByTree,
#                      scale_pos_weight = scalePosWeight,
#                      seed=27)

# gsearch2 = GridSearchCV(estimator = xgbR2,
#                         param_grid = ParamTest2,
#                         scoring='neg_root_mean_squared_error',
#                         n_jobs=4,
#                         cv=5)

# gsearch2.fit(x_train,y_train)

# print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)

""" Parameter Test 3 - GAMMA """
# ParamTest3 = {
#  'gamma': [i/10.0 for i in range(0,5)]
# }

# xgbR3 = XGBRegressor(learning_rate = learningRate,
#                      n_estimators = numBoostRound,
#                      max_depth = maxDepth,
#                      min_child_weight= minChildWeight,
#                      gamma = gamma,
#                      subsample = subsample,
#                      colsample_bytree = colsampleByTree,
#                      scale_pos_weight = scalePosWeight,
#                      seed=27)

# gsearch3 = GridSearchCV(estimator = xgbR3,
#                         param_grid = ParamTest3,
#                         scoring='neg_root_mean_squared_error',
#                         n_jobs=4,
#                         cv=5)

# gsearch3.fit(x_train, y_train)
# print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)

""" Parameter Test 4 - Subsample & Colsample By Tree """
# ParamTest4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }

# xgbR4 = XGBRegressor(learning_rate = learningRate,
#                      n_estimators = numBoostRound,
#                      max_depth = maxDepth,
#                      min_child_weight= minChildWeight,
#                      gamma = gamma,
#                      subsample = subsample,
#                      colsample_bytree = colsampleByTree,
#                      scale_pos_weight = scalePosWeight,
#                      seed=27)

# gsearch4 = GridSearchCV(estimator = xgbR4,
#                         param_grid = ParamTest4,
#                         scoring='neg_root_mean_squared_error',
#                         n_jobs=4,
#                         cv=5)

# gsearch4.fit(x_train, y_train)
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)

""" Parameter Test 4 - Extend """
# ParamTest4 = {
#  'subsample':[i/100.0 for i in range(85,100,5)],
#  'colsample_bytree':[i/100.0 for i in range(75,90,5)]
# }

# xgbR4 = XGBRegressor(learning_rate = learningRate,
#                      n_estimators = numBoostRound,
#                      max_depth = maxDepth,
#                      min_child_weight= minChildWeight,
#                      gamma = 0,
#                      subsample = subsample,
#                      colsample_bytree = colsampleByTree,
#                      scale_pos_weight = scalePosWeight,
#                      seed=27)

# gsearch4 = GridSearchCV(estimator = xgbR4,
#                         param_grid = ParamTest4,
#                         scoring='neg_root_mean_squared_error',
#                         n_jobs=4,
#                         cv=5)

# gsearch4.fit(x_train, y_train)
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)

""" Step 5 Tuning regularization parameters """
# ParamTest5 = {
#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# }

# xgbR5 = XGBRegressor(learning_rate = learningRate,
#                      n_estimators = numBoostRound,
#                      max_depth = maxDepth,
#                      min_child_weight= minChildWeight,
#                      gamma = gamma,
#                      subsample = subsample,
#                      colsample_bytree = colsampleByTree,
#                      scale_pos_weight = scalePosWeight,
#                      seed=27)

# gsearch5 = GridSearchCV(estimator = xgbR5,
#                         param_grid = ParamTest5,
#                         scoring='neg_root_mean_squared_error',
#                         n_jobs=4,
#                         cv=5)

# gsearch5.fit(x_train, y_train)
# print(gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_)