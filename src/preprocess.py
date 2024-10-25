import pandas as pd
import numpy as np
import os
import sys
import yaml
import pickle

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

params=yaml.safe_load(open("params.yaml"))['preprocess']


def preprocess(input_path,output_path):
    df= pd.read_csv(input_path)
    cat_columns='description'
    num_columns=['temp', 'precip', 'snow', 'windspeed']
    target=['pickup_counts']

    preprocess=ColumnTransformer(
    [('TfidfVectorizer',TfidfVectorizer(),cat_columns),
    ('StandardScaler',StandardScaler(),num_columns)]
    )

    trans_data= df[num_columns + [cat_columns]]

    data_model= preprocess.fit_transform(trans_data)

    tfidf=preprocess.named_transformers_['TfidfVectorizer']
    tfidf_encoded_columns= tfidf.get_feature_names_out()

    new_columns = num_columns + list(tfidf_encoded_columns)
    data_model_set = pd.DataFrame(data_model, columns=new_columns)

    pca= PCA(n_components=0.95)
    data_model_pca= pca.fit_transform(data_model_set)

    data_model_set_pca=pd.DataFrame(data_model_pca)

    data_model_set = pd.concat([data_model_set_pca, df[target]], axis=1)


    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    data_model_set.to_csv(output_path,index=False)


if __name__=="__main__":
    preprocess(params['input'],params['output'])





