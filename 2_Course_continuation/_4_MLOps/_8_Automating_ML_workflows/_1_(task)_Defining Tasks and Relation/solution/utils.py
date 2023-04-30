'''
Utils.py contains all utility functions
used during the inference process
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from heart_disease.constants import *
import joblib


def get_inference_data():
    '''
    Method for loading inference data
    Input: No input
    Output: Returns inference data features and labels
    Example usage: inference_data, labels = get_inference_data()
    '''
    # Live connection to the database
    data = pd.read_csv(f"{DATA_DIR}/heart.csv")
    data.drop_duplicates(subset=None, inplace=True)
    data.duplicated().any()
    inerence_data = data[data.columns.drop('target')]
    actual_lables = data['target']
    
    inerence_data.to_csv(f"{DATA_DIR}/heart_processed.csv", index = False)
    actual_lables.to_csv(f"{DATA_DIR}/heart_labels.csv", index = False)


# apply same pre-processing and feature engineering techniques as applied during the training process
def encode_features():
    '''
    Method for one-hot encoding all selected categorical fields
    Input: The method takes pandas dataframe and
    list of the feature names as input
    Output: Returns a dataframe with one-hot encoded features
    Example usage:
    one_hot_encoded_df = encode_features(dataframe, list_features_to_encode)
    '''
    # Implement these steps to prevent dimension mismatch during inference
    df = pd.read_csv(f"{DATA_DIR}/heart_processed.csv")
    encoded_df = pd.DataFrame(columns= ONE_HOT_ENCODED_FEATURES) # from constants.py
    placeholder_df = pd.DataFrame()
    
    # One-Hot Encoding using get_dummies for the specified categorical features
    for f in FEATURES_TO_ENCODE:
        if(f in df.columns):
            encoded = pd.get_dummies(df[f])
            encoded = encoded.add_prefix(f + '_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')
            return df
    
    # Implement these steps to prevent dimension mismatch during inference
    for feature in encoded_df.columns:
        if feature in df.columns:
            encoded_df[feature] = df[feature]
        if feature in placeholder_df.columns:
            encoded_df[feature] = placeholder_df[feature]
    # fill all null values
    encoded_df.fillna(0, inplace=True)
    
    encoded_df.to_csv(f"{DATA_DIR}/heart_encoded.csv", index = False)
    

def normalize_data():
    '''
    Normalize data using Min-Max Scaler
    Input: The method takes pandas dataframe as input
    Output: Returns a dataframe with normalized features
    Example usage:
    normalized_df = normalize_data(df)
    '''
    df = pd.read_csv(f"{DATA_DIR}/heart_encoded.csv")
    values = df.values 
    min_max_normalizer = preprocessing.MinMaxScaler()
    norm_val = min_max_normalizer.fit_transform(values)
    norm_df = pd.DataFrame(norm_val)
    
    norm_df.to_csv(f"{DATA_DIR}/heart_disease_normalized.csv", index=False)


def predict_data():
    df = pd.read_csv(f"{DATA_DIR}/heart_disease_normalized.csv")
    model = joblib.load(f"{MODEL_DIR}/aditya_model1_adaboost.joblib")
    predictions = model.predict(df)
    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv(f"{DATA_DIR}/heart_predictions.csv", index = False)
