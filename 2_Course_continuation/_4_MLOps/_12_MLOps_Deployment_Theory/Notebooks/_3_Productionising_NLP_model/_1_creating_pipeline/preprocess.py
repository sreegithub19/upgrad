"""Feature engineering for complaints dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    # call the internal preprocessing files
    # Comment this during execution as it will take more time 
    #execfile('complaints.py')
    
    # X_test= pd.read_csv('https://sagemaker-project-p-ahzzm2wop323.s3.amazonaws.com/dataset/X_test_features.csv')
    # X_train= pd.read_csv('https://sagemaker-project-p-ahzzm2wop323.s3.amazonaws.com/dataset/X_train_features.csv')
    # Y_test= pd.read_csv('https://sagemaker-project-p-ahzzm2wop323.s3.amazonaws.com/dataset/y_test_features.csv')
    # Y_train= pd.read_csv('https://sagemaker-project-p-ahzzm2wop323.s3.amazonaws.com/dataset/y_train_features.csv')    
    X_test= pd.read_csv('https://sagemaker-project-p-2tz8o8pkejbu.s3.us-west-1.amazonaws.com/DataForModels/X_test_features.csv')
    X_train= pd.read_csv('https://sagemaker-project-p-2tz8o8pkejbu.s3.us-west-1.amazonaws.com/DataForModels/X_train_features.csv')
    Y_test= pd.read_csv('https://sagemaker-project-p-2tz8o8pkejbu.s3.us-west-1.amazonaws.com/DataForModels/y_test_features.csv')
    Y_train= pd.read_csv('https://sagemaker-project-p-2tz8o8pkejbu.s3.us-west-1.amazonaws.com/DataForModels/y_train_features.csv')
    
    
    train=pd.concat([Y_train, X_train], axis=1)
    test=pd.concat([Y_test, X_test], axis=1)
    validation=pd.concat([Y_train, X_train], axis=1)

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)