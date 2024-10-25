from sklearn.metrics import r2_score,mean_absolute_error
import pandas as pd
import numpy as np
import pickle
import yaml
import os

params=yaml.safe_load(open('params.yaml'))['evaluate']
def evaluate(data_path,model_path):

    df=pd.read_csv(data_path)
    target=['pickup_counts']

    x=df.drop(target , axis=1)
    y=df[target]

    model= pickle.load(open(model_path,'rb'))

    prediction=model.predict(x)

    r2_Score=r2_score(y,prediction)
    mae=mean_absolute_error(y,prediction)

    print(f'the mean absolute error : {mae:.2f}')
    print(f'R2 Score : {r2_Score:.2f}')

if __name__=="__main__":
    evaluate(params['data'],params['model'])