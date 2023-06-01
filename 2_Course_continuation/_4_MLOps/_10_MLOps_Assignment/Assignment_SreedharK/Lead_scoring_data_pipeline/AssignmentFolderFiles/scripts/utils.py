##############################################################################
# Import necessary modules and files
##############################################################################


import pandas as pd
import os
import sqlite3
from sqlite3 import Error

def load_data(file_path_list):
    data = []
    for eachfile in file_path_list:
        data.append(pd.read_csv(eachfile, index_col=0))
    return data


###############################################################################
# Define the function to build database
###############################################################################

def build_dbs(db_path, db_file_name):
    '''
    This function checks if the db file with specified name is present 
    in the /Assignment/01_data_pipeline/scripts folder. If it is not present it creates 
    the db file with the given name at the given path. 


    INPUTS
        db_file_name : Name of the database file 'utils_output.db'
        db_path : path where the db file should be '   


    OUTPUT
    The function returns the following under the conditions:
        1. If the file exsists at the specified path
                prints 'DB Already Exsists' and returns 'DB Exsists'

        2. If the db file is not present at the specified loction
                prints 'Creating Database' and creates the sqlite db 
                file at the specified path with the specified name and 
                once the db file is created prints 'New DB Created' and 
                returns 'DB created'


    SAMPLE USAGE
        build_dbs()
    '''
    
    if os.path.isfile(db_path+db_file_name):
        print( "DB Already Exist")
        print(os.getcwd())
        return "DB Exist"
    else:
        print ("Creating Database")
        """ create a database connection to a SQLite database """
        conn = None
        try:
            
            conn = sqlite3.connect(db_path+db_file_name)
            print("New DB Created")
        except Error as e:
            print(e)
            return "Error"
        finally:
            if conn:
                conn.close()
                return "DB Created"
    

###############################################################################
# Define function to load the csv file to the database
###############################################################################

def load_data_into_db(db_path, db_file_name,data_directory):
    '''
    Thie function loads the data present in datadirectiry into the db
    which was created previously.
    It also replaces any null values present in 'toal_leads_dropped' and
    'referred_lead' with 0.


    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        data_directory : path of the directory where 'leadscoring.csv' 
                        file is present
        

    OUTPUT
        Saves the processed dataframe in the db in a table named 'loaded_data'.
        If the table with the same name already exsists then the function 
        replaces it.


    SAMPLE USAGE
        load_data_into_db()
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    leadscoring = load_data( [f"{data_directory}leadscoring.csv",])[0]
    leadscoring.reset_index(drop=True)
    leadscoring.to_sql(name='loaded_data', con=cnx, if_exists='replace')
    cnx.close()
    return "Writing to DataBase loaded_data Done or Data Already was in Table. Check Logs."


###############################################################################
# Define function to map cities to their respective tiers
###############################################################################

    
def map_city_tier(db_path, db_file_name,city_tier_mapping):
    '''
    This function maps all the cities to their respective tier as per the
    mappings provided in /mappings/city_tier_mapping.py file. If a
    particular city's tier isn't mapped in the city_tier_mapping.py then
    the function maps that particular city to 3.0 which represents
    tier-3.


    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        city_tier_mapping : a dictionary that maps the cities to their tier

    
    OUTPUT
        Saves the processed dataframe in the db in a table named
        'city_tier_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_city_tier()

    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    map_df = pd.read_sql('select * from loaded_data', cnx)
    map_df["city_tier"] = map_df["city_mapped"].map(city_tier_mapping)
    map_df["city_tier"] = map_df["city_tier"].fillna(3.0)
    map_df.drop(columns=['city_mapped','index'],axis=1,inplace=True,errors='ignore')
    map_df.to_sql(name='city_tier_mapped',con=cnx,if_exists='replace')
    cnx.close()
    return "Writing to DataBase city_tier_mapped Done or Data Already was in Table. Check Logs."

###############################################################################
# Define function to map insignificant categorial variables to "others"
###############################################################################


def map_categorical_vars(db_path,db_file_name,list_platform,list_medium,list_source):
    '''
    This function maps all the unsugnificant variables present in 'first_platform_c'
    'first_utm_medium_c' and 'first_utm_source_c'. The list of significant variables
    should be stored in a python file in the 'significant_categorical_level.py' 
    so that it can be imported as a variable in utils file.
    

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        list_platform : list of all the significant platform.
        list_medium : list of all the significat medium
        list_source : list of all rhe significant source

        **NOTE : list_platform, list_medium & list_source are all constants and
                 must be stored in 'significant_categorical_level.py'
                 file. The significant levels are calculated by taking top 90
                 percentils of all the levels. For more information refer
                 'data_cleaning.ipynb' notebook.
  

    OUTPUT
        Saves the processed dataframe in the db in a table named
        'categorical_variables_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_categorical_vars()
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    cat_df = pd.read_sql('select * from city_tier_mapped', cnx)
    
    cat_df.drop(columns=['level_0','index'],axis=1,inplace=True,errors='ignore')

    # all the levels below 90 percentage are assgined to a single level called others
    new_df = cat_df[~cat_df['first_platform_c'].isin(list_platform)] 
    new_df['first_platform_c'] = "others"
    old_df = cat_df[cat_df['first_platform_c'].isin(list_platform)] 
    cat_df = pd.concat([new_df, old_df])
    
    new_df = cat_df[~cat_df['first_utm_medium_c'].isin(list_platform)] 
    new_df['first_utm_medium_c'] = "others"
    old_df = cat_df[cat_df['first_utm_medium_c'].isin(list_platform)] 
    cat_df = pd.concat([new_df, old_df])
    
    new_df = cat_df[~cat_df['first_utm_source_c'].isin(list_platform)] 
    new_df['first_utm_source_c'] = "others"
    old_df = cat_df[cat_df['first_utm_source_c'].isin(list_platform)] 
    cat_df = pd.concat([new_df, old_df])

    cat_df.to_sql(name='categorical_variables_mapped',con=cnx,if_exists='replace')
    cnx.close()
    return "Writing to DataBase categorical_variables_mapped Done Check Logs."

##############################################################################
# Define function that maps interaction columns into 4 types of interactions
##############################################################################
def interactions_mapping(db_path,db_file_name,interaction_mapping_file,index_columns):
    '''
    This function maps the interaction columns into 4 unique interaction columns
    These mappings are present in 'interaction_mapping.csv' file. 


    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        interaction_mapping_file : path to the csv file containing interaction's
                                   mappings
        index_columns : list of columns to be used as index while pivoting and
                        unpivoting
        NOTE : Since while inference we will not have 'app_complete_flag' which is
        our label, we will have to exculde it from our index_columns. It is recommended 
        that you use an if loop and check if 'app_complete_flag' is present in 
        'categorical_variables_mapped' table and if it is present pass a list with 
        'app_complete_flag' in it as index_column else pass a list without 'app_complete_flag'
        in it.

    
    OUTPUT
        Saves the processed dataframe in the db in a table named 
        'interactions_mapped'. If the table with the same name already exsists then 
        the function replaces it.
        
        It also drops all the features that are not requried for training model and 
        writes it in a table named 'model_input'

    
    SAMPLE USAGE
        interactions_mapping()
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    df = pd.read_sql('select * from categorical_variables_mapped', cnx)
    
    df.drop(columns=['index'],axis=1,inplace=True,errors='ignore')
    df = df.drop_duplicates()
    
    # read the interaction mapping file
    df_event_mapping = load_data( [f"{interaction_mapping_file}interaction_mapping.csv",])[0]
    
    # unpivot the interaction columns and put the values in rows
    df_unpivot = pd.melt(df, id_vars=index_columns, var_name='interaction_type', value_name='interaction_value')
    
    # handle the nulls in the interaction value column
    df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)
    
    # map interaction type column with the mapping file to get interaction mapping
    df = pd.merge(df_unpivot, df_event_mapping, on='interaction_type', how='left')
    
    #dropping the interaction type column as it is not needed
    df = df.drop(['interaction_type'], axis=1)
    
    # pivoting the interaction mapping column values to individual columns in the dataset
    df_pivot = df.pivot_table(values='interaction_value', index=index_columns, columns='interaction_mapping', aggfunc='sum')
    df_pivot = df_pivot.reset_index()
    
    df_pivot.to_sql(name='interactions_mapped',con=cnx,if_exists='replace')
    df_pivot.drop(columns=['index'],axis=1,inplace=True,errors='ignore')
    cnx.close()
    return "Writing to DataBase- interactions_mapped Done . Check Logs."
   