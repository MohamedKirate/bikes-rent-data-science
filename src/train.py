from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor

import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import yaml

params=yaml.safe_load(open('params.yaml'))['train']

def train(data_path,model_path,n_estimators,learning_rate,max_depth,subsample):
    df= pd.read_csv(data_path)

    target=['pickup_counts']
    x=df.drop(target,axis=1)
    y=df[target]

    y_log= np.log1p(y)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y_log,test_size=0.2,random_state=42)
    params={
        'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        'max_depth':max_depth,
        'subsample':subsample,
    }
    model= XGBRegressor(**params)

    model.fit(x_train,y_train)

    prediction_log=model.predict(x_test)

    r2_Score_xgb=r2_score(y_test,prediction_log)
    mae_xgb=mean_absolute_error(y_test,prediction_log)

    print(f' xgb the mean absolute error : {mae_xgb:.2f}')
    print(f' xgb R2 Score : {r2_Score_xgb}')


    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    pickle.dump(model,open(model_path,'wb'))

    path_pred_data='model/predict_data.csv'
    os.makedirs(os.path.dirname(path_pred_data),exist_ok=True)

    prediction= np.expm1(y_log)

    predict_data = pd.concat([df[target].iloc[x_test.index], pd.DataFrame(prediction, columns=['predicted_counts'])], axis=1)

    predict_data.to_csv(path_pred_data, index=False)


if __name__=="__main__":
    train(params['data'],params['model'],
          params['n_estimators'],params['learning_rate'],params['max_depth'],
          params['subsample'])
    


