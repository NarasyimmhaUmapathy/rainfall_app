#import gridsearch

from core import config
import joblib
from sklearn.model_selection import GridSearchCV
import json
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score,precision_score,recall_score,make_scorer
from sklearn.model_selection import KFold
import pickle,numpy as np


X_train = pd.read_csv("./processed/X_train_scaled.csv")
y_train = pd.read_csv("./processed/y_train.csv")
params = config.b_config.grid_params
max_iter = config.b_config.max_iter

ridge = Ridge(max_iter=max_iter)
print(params)
f1 = make_scorer(f1_score)
precision = make_scorer(precision_score)
recall = make_scorer(recall_score)
scorer = {"f1":f1,"precision":precision,"recall":recall}

param_grid = {params[0]:[0.1,1],params[2]:[0.0001,0.001],
              params[1]:("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs")}

cv = KFold()
grid = GridSearchCV(estimator=ridge,param_grid=param_grid,cv=cv,scoring=scorer,refit="r2")

grid.fit(X_train,y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_
best_score = np.mean(grid.best_score_)

print(best_params)

with open("./models/grid_params.pkl", "wb") as fp:
    pickle.dump(best_params,fp) 
    pickle.dummp(best_score,fp)

