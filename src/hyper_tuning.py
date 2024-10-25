from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
import os
import numpy as np

# Define the parameters to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'alpha': [0, 0.1],
    'lambda': [1, 10],
    'min_child_weight': [1, 5]
}
params=yaml.safe_load(open('params.yaml'))['hyper_tuning']
train_params=yaml.safe_load(open('params.yaml'))['train']


def hyper_tuning(data_path):
    df= pd.read_csv(data_path)
    target=['pickup_counts']

    X= df.drop(target,axis=1)
    y= df[target]

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    }
    model= XGBRegressor()
    grid_search=GridSearchCV(model,param_grid,cv=3)
    grid_search.fit(x_train,y_train)

    best_model=grid_search.best_estimator_

    y_pred_test=best_model.predict(x_test)

    mae=mean_absolute_error(y_test,y_pred_test)
    r2= r2_score(y_test,y_pred_test)

    best_params = grid_search.best_params_
    train_params.update(best_params)

    print(f'mae : {mae:.3f}')
    print(f'r2 score : {r2:.3f}')
    print('best params : ')
    print(grid_search.best_params_)


if __name__=='__main__':
    hyper_tuning(params['data'])


