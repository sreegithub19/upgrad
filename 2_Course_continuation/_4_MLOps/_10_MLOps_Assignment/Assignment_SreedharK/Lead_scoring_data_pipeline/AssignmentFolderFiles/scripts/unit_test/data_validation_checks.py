"""
Import necessary modules
############################################################################## 
"""

import pandas as pd

import os
import sqlite3
from sqlite3 import Error
import collections

 
    
def load_data(file_path_list):
    data = []
    for eachfile in file_path_list:
        data.append(pd.read_csv(eachfile, index_col=0))
    return data

###############################################################################
# Define function to validate raw data's schema
# ############################################################################## 

def raw_data_schema_check(data_directory,raw_data_schema):
    '''
    This function check if all the columns mentioned in schema.py are present in
    leadscoring.csv file or not.

   
    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be   
        raw_data_schema : schema of raw data in the form oa list/tuple as present 
                          in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Raw datas schema is in line with the schema present in schema.py' 
        else prints
        'Raw datas schema is NOT in line with the schema present in schema.py'

    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    leadscoring = load_data( [f"{data_directory}leadscoring.csv",])[0]
    source_cols = leadscoring.columns.to_list()
    
    if collections.Counter(source_cols) == collections.Counter(raw_data_schema):
        print('Raw datas schema is in line with the schema present in schema.py')
    else:
        print('Raw datas schema is NOT in line with the schema present in schema.py')
    


###############################################################################
# Define function to validate model's input schema
# ############################################################################## 

def model_input_schema_check(db_path, db_file_name, model_input_schema):
    '''
    This function check if all the columns mentioned in model_input_schema in 
    schema.py are present in table named in 'model_input' in db file.

   
    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be   
        raw_data_schema : schema of models input data in the form oa list/tuple
                          present as in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Models input schema is in line with the schema present in schema.py'
        else prints
        'Models input schema is NOT in line with the schema present in schema.py'
    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    df = pd.read_sql('select * from interactions_mapped', cnx)
    df.drop(columns=['index'], inplace=True, axis=1, errors='ignore')
    source_columns = df.columns.to_list()
    result =  all(elem in source_columns for elem in model_input_schema)
    if result:
        print('Models input schema is in line with the schema present in schema.py')
    else:
        print('Models input schema is NOT in line with the schema present in schema.py')    

    
    
