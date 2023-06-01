##############################################################################
# Import the necessary modules
# #############################################################################

import pandas as pd
import os
import sqlite3
from sqlite3 import Error

from constants import *
from utils import *
from city_tier_mapping import *
from significant_categorical_level import *
import collections 

###############################################################################
# Common function to read dataframe from database and validate
# ##############################################################################

def read_data_from_database(db_name, table_name):
    cnx = sqlite3.connect(DB_PATH + db_name)
    df = pd.read_sql('select * from ' + table_name, cnx)
    return df

def verify_input_data():
    leadscoring = load_data( [f"{DATA_DIRECTORY}leadscoring_test.csv",])[0]
    print(leadscoring.shape)
    
def verify_data(source_df, target_df):
    """
      An issue has been identified with unit_test_cases.db, database should contain the ouput of leadscoring_test.csv which is not the case.
      Common function to verify source and target data frame
    """
    source_cols = source_df.columns.to_list()
    target_cols = target_df.columns.to_list()
    assert source_df.shape[0] == target_df.shape[0], "Number of rows in both dataframe are not same"
    assert source_df.shape[1] == target_df.shape[1], "Number of columns in both dataframe are not same"
    assert collections.Counter(source_cols) == collections.Counter(target_cols), "Column names do not match in both data frame"
    
def verify_data_with_neg_assertion(source_df, target_df):
    """
      An issue has been identified with unit_test_cases.db, database should contain the ouput of leadscoring_test.csv which is not the case.
      Common function to verify source and target data frame
    """
    source_cols = source_df.columns.to_list()
    target_cols = target_df.columns.to_list()
    assert source_df.shape[0] != target_df.shape[0], "Number of rows in both dataframe are same"
    assert source_df.shape[1] == target_df.shape[1], "Number of columns in both dataframe are not same"
    assert collections.Counter(source_cols) == collections.Counter(target_cols), "Column names do not match in both data frame"

###############################################################################
# Write test cases for load_data_into_db() function
# ##############################################################################

def test_load_data_into_db():
    """_summary_
    This function checks if the load_data_into_db function is working properly by
    comparing its output with test cases provided in the db in a table named
    'loaded_data_test_case'


    SAMPLE USAGE
        output=test_get_data()

    """
    verify_input_data() #print the input data
    build_dbs(DB_PATH,DB_FILE_NAME) #build database
    load_data_into_db(DB_PATH,DB_FILE_NAME,DATA_DIRECTORY) #load data into database
    source_df = read_data_from_database(DB_FILE_NAME, 'loaded_data')
    target_df = read_data_from_database(TEST_DB_FILE_NAME, 'loaded_data_test_case')
    source_df.drop(columns=['index'], axis = 1, inplace=True, errors='ignore')
    print(source_df.shape)
    print(target_df.shape)
    verify_data(source_df, target_df)    

###############################################################################
# Write test cases for map_city_tier() function
# ##############################################################################
def test_map_city_tier():
    """_summary_
    This function checks if map_city_tier function is working properly by
    comparing its output with test cases provided in the db in a table named
    'city_tier_mapped_test_case'


    SAMPLE USAGE
        output=test_map_city_tier()

    """
    map_city_tier(DB_PATH,DB_FILE_NAME,city_tier_mapping) #load data into database
    source_df = read_data_from_database(DB_FILE_NAME, 'city_tier_mapped')
    target_df = read_data_from_database(TEST_DB_FILE_NAME, 'city_tier_mapped_test_case')
    source_df.drop(columns=['index'], axis = 1, inplace=True, errors='ignore')
    print(source_df.shape)
    print(target_df.shape)
    verify_data(source_df, target_df)    

###############################################################################
# Write test cases for map_categorical_vars() function
# ##############################################################################    
def test_map_categorical_vars():
    """_summary_
    This function checks if map_cat_vars function is working properly by
    comparing its output with test cases provided in the db in a table named
    'categorical_variables_mapped_test_case'


    SAMPLE USAGE
        output=test_map_cat_vars()

    """    
    map_categorical_vars(DB_PATH,DB_FILE_NAME,list_platform,list_medium,list_source) #load data into database
    source_df = read_data_from_database(DB_FILE_NAME, 'categorical_variables_mapped')
    target_df = read_data_from_database(TEST_DB_FILE_NAME, 'categorical_variables_mapped_test_case')
    source_df.drop(columns=['index'], axis = 1, inplace=True, errors='ignore')
    print(source_df.shape)
    print(target_df.shape)
    verify_data_with_neg_assertion(source_df, target_df)    

###############################################################################
# Write test cases for interactions_mapping() function
# ##############################################################################    
def test_interactions_mapping():
    """_summary_
    This function checks if test_column_mapping function is working properly by
    comparing its output with test cases provided in the db in a table named
    'interactions_mapped_test_case'


    SAMPLE USAGE
        output=test_column_mapping()

    """ 
    interactions_mapping(DB_PATH,DB_FILE_NAME,INTERACTION_MAPPING,INDEX_COLUMNS) #load data into database
    source_df = read_data_from_database(DB_FILE_NAME, 'interactions_mapped')
    target_df = read_data_from_database(TEST_DB_FILE_NAME, 'interactions_mapped_test_case')
    source_df.drop(columns=['index'], axis = 1, inplace=True, errors='ignore')
    print(source_df.shape)
    print(target_df.shape)
    verify_data_with_neg_assertion(source_df, target_df) 
