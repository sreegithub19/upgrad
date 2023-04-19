import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pandas_profiling import ProfileReport
import sqlite3
from sqlite3 import Error
# from pycaret.classification import *
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV # run pip install scikit-optimize
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from datetime import datetime
from datetime import date


def load_data(file_path_list):
    data = []
    for eachfile in file_path_list:
        data.append(pd.read_csv(eachfile))
    return data

def compress_dataframes(list_of_dfs):
    final_df = []
    for eachdf in list_of_dfs:
        original_size = (eachdf.memory_usage(index=True).sum())/ 1024**2
        int_cols = list(eachdf.select_dtypes(include=['int']).columns)
        float_cols = list(eachdf.select_dtypes(include=['float']).columns)
        for col in int_cols:
            if ((np.max(eachdf[col]) <= 127) and(np.min(eachdf[col] >= -128))):
                eachdf[col] = eachdf[col].astype(np.int8)
            elif ((np.max(eachdf[col]) <= 32767) and(np.min(eachdf[col] >= -32768))):
                eachdf[col] = eachdf[col].astype(np.int16)
            elif ((np.max(eachdf[col]) <= 2147483647) and(np.min(eachdf[col] >= -2147483648))):
                eachdf[col] = eachdf[col].astype(np.int32)
            else:
                eachdf[col] = eachdf[col].astype(np.int64)
    
        for col in float_cols:
            eachdf[col] = eachdf[col].astype(np.float16)
        compressed_size = (eachdf.memory_usage(index=True).sum())/ 1024**2
        
        final_df.append((eachdf,original_size,compressed_size))
        
    return final_df

def count_plot(dataframe, list_of_columns):
    for eachcol in list_of_columns:
        plt.figure(figsize=(15,5))
        unique_features = dataframe[eachcol].unique()
        if dataframe[eachcol].dtype =='int64':
            unique_features=sorted(unique_features)
        sns.countplot(x=eachcol, data=dataframe , order = unique_features)
        plt.xlabel(eachcol)
        plt.ylabel('Count')
        plt.title("Frequency plot of {} Count".format(eachcol))
        plt.show()

def fix_time_in_df(dataframe, column_name, expand=False):
    if not expand:
        dataframe[column_name] = dataframe[column_name].astype('str')
        return pd.to_datetime(dataframe[column_name])
    else:
        dataframe_new = dataframe.copy()
        dataframe_new[column_name] = dataframe_new[column_name].astype('str')
        dataframe_new[column_name] = pd.to_datetime(dataframe_new[column_name])
        #Extracting the date time year component
        dataframe_new[f"{column_name}_year"] = pd.DatetimeIndex(dataframe_new[column_name]).year
        #Extracting the date time year component
        dataframe_new[f"{column_name}_month"] = pd.DatetimeIndex(dataframe_new[column_name]).month
        #Extracting the date time year component
        dataframe_new[f"{column_name}_day"] = pd.DatetimeIndex(dataframe_new[column_name]).day_name()
      
        return dataframe_new
    


def get_data_profile(dataframe,html_save_path, 
                     embed_in_cell=True,take_sample=False, sample_frac=0.5, dataframe_name="data"):
    if take_sample:
        dataframe = dataframe.sample(frac=sample_frac)
    if embed_in_cell:
        profile = ProfileReport(dataframe, title=f"{dataframe_name} Data Summary Report")
        return profile.to_notebook_iframe()
    else:
        profile = ProfileReport(dataframe, title=f"{dataframe_name} Data Summary Report")
        timestamp = str(int(time.time()))
        filename = f"{dataframe_name}_data_profile_{timestamp}"
        profile.to_file(html_save_path+filename+".html")
        return "Your Data Profile has been saved at .. ",html_save_path+filename+".html"   
    
def get_data_describe(dataframe,round_num=2):
    return round(dataframe.describe(),round_num)

def get_data_na_values(dataframe, round_num=2):
    return pd.DataFrame({'%missing_values':round(dataframe.isna().sum()/dataframe.shape[0],round_num)})

def get_fill_na_dataframe(dataframe, column_name, value='mean'):
    if value != 'mean' and value !='mode':
        return dataframe[column_name].fillna(value)
    elif value == 'mean':
        value = dataframe[column_name].mean()
        return dataframe[column_name].fillna(value)
    elif value == 'mode':
        value = dataframe[column_name].mode()
        return dataframe[column_name].fillna(value)

def get_convert_column_dtype(dataframe, column_name, data_type='str'):
    if data_type == 'str':
        return dataframe[column_name].astype('str')
    elif data_type == 'int':
        return dataframe[column_name].astype('int')
    elif data_type == 'float':
        return dataframe[column_name].astype('float')
    
def get_groupby(dataframe, by_column, agg_dict=None, agg_func = 'mean', simple_agg_flag=True, reset_index=True):
    if reset_index:
        if simple_agg_flag:
            return dataframe.groupby(by_column).agg(agg_func).reset_index()
        else:
            return dataframe.groupby(by_column).agg(agg_dict).reset_index()
    else:
        if simple_agg_flag:
            return dataframe.groupby(by_column).agg(agg_func)
        else:
            return dataframe.groupby(by_column).agg(agg_dict)
        
def get_merge(dataframe1, dataframe2, on, axis=1,how='inner'):
    return dataframe1.merge(dataframe2, on=on,how=how)

def get_fix_skew_with_log(dataframe, columns, replace_inf = True, replace_inf_with = 0):
    if replace_inf:
        dataframe_log = np.log(dataframe[columns]).replace([np.inf, -np.inf], replace_inf_with)
        return pd.concat([dataframe_log, dataframe.drop(columns,axis=1)], axis=1)
    else:
        dataframe_log = np.log(dataframe[columns])
        return pd.concat([dataframe_log, dataframe.drop(columns,axis=1)], axis=1)
        
def get_save_intermediate_data(dataframe, path, filename="data_interim"):
    filename = filename+"_"+str(int(time.time()))+".csv"
    dataframe.to_csv(path+filename,index=False)
    return "Data Saved Here :",path+filename

def get_label_encoding_dataframe(dataframe, column_name, mapping_dict):
    return dataframe[column_name].map(mapping_dict) 
# #average_age if (x <=0 or x >100) else x

def get_apply_condiiton_on_column(dataframe, column_name, condition):
    return dataframe[column_name].apply(lambda x :eval(condition))


def get_two_column_operations(dataframe, columns_1, columns_2, operator):
    if operator == "+":
        return dataframe[columns_1]+dataframe[columns_2]
    elif operator == "-":
        return dataframe[columns_1]-dataframe[columns_2]
    elif operator == "/":
        return dataframe[columns_1]/dataframe[columns_2]
    elif operator == "*":
        return dataframe[columns_1]*dataframe[columns_2]
    
def get_timedelta_division(dataframe, column, td_type='D'):
    return dataframe[column] /np.timedelta64(1,td_type)

def get_replace_value_in_df(dataframe, column, value, replace_with):
    return dataframe[column].replace(value,replace_with) 

def get_validation_unseen_set(dataframe, validation_frac=0.05, sample=False, sample_frac=0.1):
    if not sample:
        dataset = dataframe.copy()
    else:
        dataset = dataframe.sample(frac=sample_frac)
    data = dataset.sample(frac=(1-validation_frac), random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(inplace=True, drop=True)
    data_unseen.reset_index(inplace=True, drop=True)
    return data, data_unseen

def create_sqlit_connection(db_path,db_file):
    """ create a database connection to a SQLite database """
    conn = None
    # opening the conncetion for creating the sqlite db
    try:
        conn = sqlite3.connect(db_path+db_file)
        print(sqlite3.version)
    # return an error if connection not established
    except Error as e:
        print(e)
    # closing the connection once the database is created
    finally:
        if conn:
            conn.close()
            
            
#{'pycaret_globals', '_all_models_internal', 'y', 'iterative_imputation_iters_param', 
# #'seed', 'imputation_regressor', 'display_container', 'logging_param', '_available_plots',
# 'prep_pipe', 'imputation_classifier', 'X_test', '_ml_usecase', '_internal_pipeline',
# 'X_train', 'n_jobs_param', 'X', 'data_before_preprocess', 'target_param', 
# 'master_model_container', 'USI', '_gpu_n_jobs_param', 'exp_name_log', 
# 'y_test', 'fold_groups_param_full', 'log_plots_param', 
# 'fix_imbalance_param', 'fold_groups_param', 'fold_param', 
# 'experiment__', 'create_model_container', 'gpu_param', '_all_models', 
# 'stratify_param', 'fold_shuffle_param', 'y_train', 'html_param', 
# 'fix_imbalance_method_param', 'transform_target_method_param', 
# 'transform_target_param', 'dashboard_logger', 'fold_generator', '_all_metrics'}

def get_train_test_set_from_setup():
    return get_config(variable="X_train"),\
            get_config(variable="y_train"),\
            get_config(variable="X_test"),\
            get_config(variable="y_test")

def get_x_y_from_setup():
    return get_config(variable="X"),\
            get_config(variable="y")

def get_transformation_pipeline_from_setup():
    return get_config(variable="prep_pipe")

# Pipeline Functions 

def check_if_table_has_value(cnx,table_name):
    # cnx = sqlite3.connect(db_path+db_file_name)
    check_table = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", cnx).shape[0]
    if check_table == 1:
        return True
    else:
        return False
    
def build_dbs(db_path,db_file_name):
    if os.path.isfile(db_path+db_file_name):
        print( "DB Already Exsist")
        print(os.getcwd())
        return "DB Exsist"
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

def get_new_data_appended(old_data_directory,new_data_directory, start_data, end_date,append=False):
    user_logs_n, transactions_n  = load_data( [f"{new_data_directory}user_logs_new.csv",
                                               f"{new_data_directory}transactions_logs_new.csv",
                                             ]
                                            )
    
    members, user_logs, transactions, train  = load_data( [
                                                            f"{old_data_directory}members_profile.csv",
                                                            f"{old_data_directory}userlogs.csv",
                                                            f"{old_data_directory}transactions_logs.csv",
                                                            f"{old_data_directory}churn_logs.csv"
                                                            ]
                                                          )
    
    members_list = np.unique(list(members['msno']))
    train_members_list = np.unique(list(train['msno']))
    
    #get the list of memebers fron historical data. This assumes, no new user has been added in the system. Shouldn't be done, when new users are adde
    #Some Date Filters are manual at this point for sanity check 
    
    user_logs_n['date'] = fix_time_in_df(user_logs_n, 'date', expand=False)
    march_user_logs = user_logs_n[(user_logs_n['date']>start_data) & 
            (user_logs_n['date']<end_date) &
            (user_logs_n['msno'].isin(members_list)) & 
            (user_logs_n['msno'].isin(train_members_list))]

    transactions_n['transaction_date'] = fix_time_in_df(transactions_n, 'transaction_date', expand=False)
    transactions_n['membership_expire_date'] = fix_time_in_df(transactions_n, 'membership_expire_date', expand=False)
    march_transactions = transactions_n[(transactions_n['transaction_date']>start_data) & 
               (transactions_n['transaction_date']<end_date) & 
               (transactions_n['membership_expire_date']<'2017-12-31') & 
               (transactions_n['msno'].isin(members_list)) & 
               (transactions_n['msno'].isin(train_members_list))]

    if not append:
        return march_user_logs, march_transactions
    else:
        user_logs_combined = user_logs.append(march_user_logs)
        transactions_combined = transactions.append(march_transactions)
        return user_logs_combined, transactions_combined

def load_data_from_source(db_path,db_file_name,drfit_db_name, 
                          old_data_directory,new_data_directory,
                          run_on='old',start_data='2017-03-01', end_date='2017-03-31',
                         append=True):
    
    #get process flag df
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    # print(process_flags)
    # print(process_flags['load_data'])
    if process_flags['load_data'][0] == 1:
        if run_on=='old':
            print("Running on OLD Data") 
            cnx = sqlite3.connect(db_path+db_file_name)

            if not check_if_table_has_value(cnx,'train'):
                print("Table Doesn't Exsist - train, Building")
                train = load_data( [f"{old_data_directory}churn_logs.csv",
                           ]
                         )[0]
                train.to_sql(name='train', con=cnx,if_exists='replace',index=False)

            if not check_if_table_has_value(cnx,'user_logs'):
                print("Table Doesn't Exsist - user_logs, Building")
                user_logs = load_data( [f"{old_data_directory}userlogs.csv",
                           ]
                         )[0]
                user_logs, pre_size, post_size = compress_dataframes([user_logs])[0]
                print("user_logs DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                user_logs.to_sql(name='user_logs', con=cnx, if_exists='replace',index=False)

            if not check_if_table_has_value(cnx,'transactions'):
                print("Table Doesn't Exsist - transactions, Building")
                transactions = load_data( [f"{old_data_directory}transactions_logs.csv",
                           ]
                         )[0]
                transactions, pre_size, post_size = compress_dataframes([transactions])[0]
                print("transactions DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                transactions.to_sql(name='transactions', con=cnx,if_exists='replace',index=False)

            if not check_if_table_has_value(cnx,'members'):
                print("Table Doesn't Exsist - members, Building")
                members = load_data( [f"{old_data_directory}members_profile.csv",
                           ]
                         )[0]
                members, pre_size, post_size = compress_dataframes([members])[0]
                print("members DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                members.to_sql(name='members', con=cnx,if_exists='replace',index=False)

            cnx.close()
            return "Writing to DataBase Done or Data Already was in Table. Check Logs."

        elif run_on=='new':
            if append:
                print("Running on New Data") 
                cnx = sqlite3.connect(db_path+db_file_name)

                #Appending new Data to exsisting data
                march_user_logs, march_transactions = get_new_data_appended(old_data_directory,new_data_directory, start_data, end_date)


                if not check_if_table_has_value(cnx,'train'):
                    print("Table Doesn't Exsist - train, Building")
                    train = load_data( [f"{old_data_directory}churn_logs.csv",
                               ]
                             )[0]
                    train.to_sql(name='train', con=cnx,if_exists='replace',index=False)

                if not check_if_table_has_value(cnx,'user_logs'):
                    print("Table Doesn't Exsist - user_logs, Building")
                    user_logs = load_data( [f"{old_data_directory}userlogs.csv",
                               ]
                             )[0]
                    user_logs['date'] = fix_time_in_df(user_logs, 'date', expand=False)
                    user_logs_appended = user_logs.append(march_user_logs)
                    user_logs, pre_size, post_size = compress_dataframes([user_logs_appended])[0]

                    print(user_logs.head())
                    print("user_logs DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                    user_logs.to_sql(name='user_logs', con=cnx, if_exists='replace',index=False)

                if not check_if_table_has_value(cnx,'transactions'):
                    print("Table Doesn't Exsist - transactions, Building")
                    transactions = load_data( [f"{old_data_directory}transactions_logs.csv",
                               ]
                             )[0]
                    transactions['transaction_date'] = fix_time_in_df(transactions, 'transaction_date', expand=False)
                    transactions['membership_expire_date'] = fix_time_in_df(transactions, 'membership_expire_date', expand=False)
                    transactions_appended = transactions.append(march_transactions)
                    transactions, pre_size, post_size = compress_dataframes([transactions_appended])[0]

                    print("transactions DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                    transactions.to_sql(name='transactions', con=cnx,if_exists='replace',index=False)

                if not check_if_table_has_value(cnx,'members'):
                    print("Table Doesn't Exsist - members, Building")
                    members = load_data( [f"{old_data_directory}members_profile.csv",
                               ]
                             )[0]
                    members, pre_size, post_size = compress_dataframes([members])[0]
                    print("members DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                    members.to_sql(name='members', con=cnx,if_exists='replace',index=False)

                cnx.close()
                return "Writing to DataBase Done or Data Already was in Table. Check Logs."
            
            else:
                print("Running on New Data without Append.") 
                cnx = sqlite3.connect(db_path+db_file_name)

                if not check_if_table_has_value(cnx,'train'):
                    print("Table Doesn't Exsist - train, Building")
                    train = load_data( [f"{new_data_directory}churn_logs_new.csv",
                               ]
                             )[0]
                    train.to_sql(name='train', con=cnx,if_exists='replace',index=False)

                if not check_if_table_has_value(cnx,'user_logs'):
                    print("Table Doesn't Exsist - user_logs, Building")
                    user_logs = load_data( [f"{new_data_directory}user_logs_new.csv",
                               ]
                             )[0]
                    user_logs, pre_size, post_size = compress_dataframes([user_logs])[0]
                    print("user_logs DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                    user_logs.to_sql(name='user_logs', con=cnx, if_exists='replace',index=False)

                if not check_if_table_has_value(cnx,'transactions'):
                    print("Table Doesn't Exsist - transactions, Building")
                    transactions = load_data( [f"{new_data_directory}transactions_logs_new.csv",
                               ]
                             )[0]
                    transactions, pre_size, post_size = compress_dataframes([transactions])[0]
                    print("transactions DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                    transactions.to_sql(name='transactions', con=cnx,if_exists='replace',index=False)

                if not check_if_table_has_value(cnx,'members'):
                    print("Table Doesn't Exsist - members, Building")
                    members = load_data( [f"{new_data_directory}members_profile_new.csv",
                               ]
                             )[0]
                    members, pre_size, post_size = compress_dataframes([members])[0]
                    print("members DF before compress was in MB ,",pre_size, "and after compress , ", post_size)
                    members.to_sql(name='members', con=cnx,if_exists='replace',index=False)

                cnx.close()
                return "Writing to DataBase Done or Data Already was in Table. Check Logs."
                
    else:
        print("Skipping.....Not required")

def get_membership_data_transform(db_path,db_file_name,drfit_db_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['process_members'][0] == 1:
        if not check_if_table_has_value(cnx,'members_final'):

            members = pd.read_sql('select * from members', cnx)

            members['gender'] = get_fill_na_dataframe(members, 'gender', value="others")
            gender_mapping = {'male':0,'female':1,'others':2}
            members['gender'] = get_label_encoding_dataframe(members, 'gender',gender_mapping)
            members['registered_via'] = get_convert_column_dtype(members, 'registered_via', data_type='str')
            members['city'] = get_convert_column_dtype(members, 'city', data_type='str')
            members['registration_init_time'] = fix_time_in_df(members, 'registration_init_time', expand=False)
            average_age = round(members['bd'].mean(),2)
            condition = f"{average_age} if (x <=0 or x >100) else x"
            members['bd'] = get_apply_condiiton_on_column(members, 'bd', condition)

            members.to_sql(name='members_final', con=cnx,if_exists='replace',index=False)

            return "Membership Data is Transformed and Saved into members_final"
        return "Membership Data is already Transformed and Saved into members_final"
    else:
        print("Not Required......Skipping") 

def get_transaction_data_transform(db_path,db_file_name,drfit_db_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['process_transactions'][0] == 1:
    
        if not check_if_table_has_value(cnx,'transactions_features_final'):
            transactions = pd.read_sql('select * from transactions', cnx)

            transactions['transaction_date'] = fix_time_in_df(transactions, 'transaction_date', expand=False)
            transactions['membership_expire_date'] = fix_time_in_df(transactions, 'membership_expire_date', expand=False)

            transactions['discount'] =  get_two_column_operations(transactions, 'plan_list_price', 'actual_amount_paid', "-")

            condition = f"1 if x > 0 else 0"
            transactions['is_discount'] = get_apply_condiiton_on_column(transactions, 'discount', condition)


            transactions['amt_per_day'] = get_two_column_operations(transactions, 'actual_amount_paid', 'payment_plan_days', "/")
            transactions['amt_per_day'] = get_replace_value_in_df(transactions, 'amt_per_day', [np.inf, -np.inf], replace_with=0)


            transactions['membership_duration'] = get_two_column_operations(transactions, 'membership_expire_date', 'transaction_date', "-")
            transactions['membership_duration'] = get_timedelta_division(transactions, "membership_duration", td_type='D')
            transactions['membership_duration'] = get_convert_column_dtype(transactions, 'membership_duration', data_type='int')

            condition = f"1 if x>30 else 0"
            transactions['more_than_30'] = get_apply_condiiton_on_column(transactions, 'membership_duration', condition)

            agg = {'payment_method_id':['count','nunique'], # How many transactions user had done in past, captures if payment method is changed
           'payment_plan_days':['mean', 'nunique'] , #Average plan of customer in days, captures how many times plan is changed
           'plan_list_price':'mean', # Average amount charged on user
           'actual_amount_paid':'mean', # Average amount paid by user
           'is_auto_renew':['mean','max'], # Captures if user changed its auto_renew state
           'transaction_date':['min','max','count'], # First and the last transaction of a user
           'membership_expire_date':'max' , # Membership exipry date of the user's last subscription
           'is_cancel':['mean','max'], # Captures the average value of is_cancel and to check if user changed its is_cancel state
           'discount' : 'mean', # Average discount given to customer
           'is_discount':['mean','max'], # Captures the average value of is_discount and to check if user was given any discount in the past
           'amt_per_day' : 'mean', # Average amount a user spends per day
           'membership_duration' : 'mean' ,# Average membership duration 
           'more_than_30' : 'sum' #Flags if the difference in days if more than 30
            }

            transactions_features = get_groupby(transactions, by_column='msno', agg_dict=agg, agg_func = 'mean', simple_agg_flag=False, reset_index=True)
            transactions_features.columns= transactions_features.columns.get_level_values(0)+'_'+transactions_features.columns.get_level_values(1)
            transactions_features.rename(columns = {'msno_':'msno','payment_plan_days_nunique':'change_in_plan', 'payment_method_id_count':'total_payment_channels',
                                                    'payment_method_id_nunique':'change_in_payment_methods','is_cancel_max':'is_cancel_change_flag',
                                                    'is_auto_renew_max':'is_autorenew_change_flag','transaction_date_count':'total_transactions'}, inplace = True)

            transactions_features.to_sql(name='transactions_features_final', con=cnx,if_exists='replace',index=False)

            return "transactions Data is Transformed and Saved into transactions_features_final"
        return "transactions Data is already Transformed and Saved into transactions_features_final"

    else:
        print("Not Required......Skipping") 

def get_user_data_transform(db_path,db_file_name,drfit_db_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['process_userlogs'][0] == 1:
        if not check_if_table_has_value(cnx,'user_logs_features_final'):
            user_logs = pd.read_sql('select * from user_logs', cnx)

            user_logs['date'] =  fix_time_in_df(user_logs, column_name='date', expand=False)
            user_logs_transformed = get_fix_skew_with_log(user_logs, ['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs'], 
                                                  replace_inf = True, replace_inf_with = 0)

            user_logs_transformed_base = get_groupby(user_logs_transformed,'msno', agg_dict=None, agg_func = 'mean', simple_agg_flag=True, reset_index=True)

            agg_dict = { 'date':['count','max'] }
            user_logs_transformed_dates = get_groupby(user_logs_transformed,'msno', agg_dict=agg_dict, agg_func = 'mean', simple_agg_flag=False, reset_index=True)
            user_logs_transformed_dates.columns = user_logs_transformed_dates.columns.droplevel()
            user_logs_transformed_dates.rename(columns = {'count':'login_freq', 'max': 'last_login'}, inplace = True)
            user_logs_transformed_dates.reset_index(inplace=True)
            user_logs_transformed_dates.drop('index',inplace=True,axis=1)
            user_logs_transformed_dates.columns = ['msno','login_freq','last_login']

            user_logs_final = get_merge(user_logs_transformed_base, user_logs_transformed_dates, on = 'msno') 

            user_logs_final.to_sql(name='user_logs_features_final', con=cnx,if_exists='replace',index=False)

            return "user_logs Data is Transformed and Saved into user_logs_features_final"
        return "user_logs Data is already Transformed and Saved into user_logs_features_final"
    
    else:
        print("Not Required......Skipping")


def get_final_data_merge(db_path,db_file_name,drfit_db_name):
    
    cnx = sqlite3.connect(db_path+db_file_name)
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['process_userlogs'][0] == 1:
        
        if not check_if_table_has_value(cnx,'final_features_v01'):
            print ("Final Merge Doesn't Exsist in DB") 
            members_final =     pd.read_sql('select * from members_final', cnx)
            transactions_final = pd.read_sql('select * from transactions_features_final', cnx)
            user_logs_final =    pd.read_sql('select * from user_logs_features_final', cnx)
            train =     pd.read_sql('select * from train', cnx)

            train_df_v01 = get_merge(members_final, train, on='msno', axis=1, how='inner')
            train_df_v02 = get_merge(train_df_v01, transactions_final, on='msno', axis=1, how='inner')
            train_df_final = get_merge(train_df_v02, user_logs_final, on='msno', axis=1, how='inner')

            train_df_final.to_sql(name='final_features_v01', con=cnx,if_exists='replace',index=False)

            return "All Data is Merged and Saved into final_features_v01"
        else:
            return "Final Merged Already Performed and Available in DB" 
    
    else:
        print("Not Required......Skipping")


def get_data_prepared_for_modeling(db_path,db_file_name,drfit_db_name, scale_method='standard',date_columns=None,corr_threshold=0.90,drop_corr=False,
                                   date_transformation=True):
  # print(len(dataframe.columns))
  # removingmulti-colinearity 
    cnx = sqlite3.connect(db_path+db_file_name)
    
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['Data_Preparation'][0] == 1:
        if not check_if_table_has_value(cnx,'X') and not check_if_table_has_value(cnx,'y'):
            dataframe = pd.read_sql('select * from final_features_v01', cnx)
            # Create correlation matrix
            corr_matrix = dataframe.corr().abs()
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            # Find features with correlation greater than 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
            print(to_drop)
            # Drop feature
            if drop_corr:
                dataframe.drop(to_drop, axis=1, inplace=True)
            print(len(dataframe.columns))
            if date_transformation:
                #date transformation 
                features = ["day","month","year","weekday"]
                date_data = dataframe[date_columns]
                for eachcol in date_data:
                    date_data[eachcol] = date_data[eachcol].astype('str')
                    date_data[eachcol] = pd.to_datetime(date_data[eachcol]) 
                    #column_name
                    for eachfeature in features:
                        col_name = f"{eachcol}_{eachfeature}"

                        if eachfeature == 'day':
                            date_data[col_name] = date_data[eachcol].dt.day
                        elif eachfeature == 'month':
                            date_data[col_name] = date_data[eachcol].dt.month
                          # result[col_name] = result[col_name].astype('int64')
                        elif eachfeature == 'year':
                            date_data[col_name] = date_data[eachcol].dt.year
                          # result[col_name] = result[col_name].astype('int64')
                        elif eachfeature == 'weekday':
                            date_data[col_name] = date_data[eachcol].dt.weekday
                date_data.drop(date_columns,axis=1,inplace=True)
                date_data = date_data.where(date_data.isna(), date_data.astype(str))
                final_date = pd.get_dummies(date_data, drop_first=True,dtype='int16')
            # print(pd.get_dummies(date_data, drop_first=True,dtype='int16')) 

            #scaling
            column_to_scale = dataframe.select_dtypes(include=['float64','int64']).columns.drop('is_churn')
            transformer = StandardScaler().fit(dataframe[column_to_scale])
            scaled_data = pd.DataFrame(transformer.transform(dataframe[column_to_scale]),columns=column_to_scale)

            #Combining
            if date_transformation:
                final_df = pd.concat([scaled_data,final_date,dataframe['is_churn']],axis=1)
            else:
                print("Doing Feature without Dates") 
                final_df = pd.concat([scaled_data,dataframe['is_churn']],axis=1)
                # final_df = final_df.drop(date_columns,axis=1)

            #Splitting X,y 
            X = final_df.drop(['is_churn'],axis=1)
            y = final_df[['is_churn']]
            index_df = dataframe[['msno']]
            index_df['index_for_map'] = index_df.index
            
            X.to_sql(name='X', con=cnx,if_exists='replace',index=False)
            y.to_sql(name='y', con=cnx,if_exists='replace',index=False)
            index_df.to_sql(name='index_msno_mapping', con=cnx,if_exists='replace',index=False)
            return "X & Y written on database"
        else:
            return "X & Y Already exsist in Data."
    
    else:
        print("Not Required......Skipping")


def get_train_model(db_path,db_file_name,drfit_db_name):
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['Data_Preparation'][0] == 1:
        cnx = sqlite3.connect(db_path+db_file_name)
        X = pd.read_sql('select * from X', cnx)
        y = pd.read_sql('select * from y', cnx)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

        model_config = {
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 1.0,
        'importance_type': 'split' ,
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'num_leaves': 31,
        'objective': None,
        'random_state': 42,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'silent': 'warn',
        'subsample': 1.0,
        'subsample_for_bin': 200000 ,
        'subsample_freq': 0
        }


        #Model Training

        with mlflow.start_run(run_name='run_LightGB_withoutHPTune') as run:
            #Model Training
            clf = lgb.LGBMClassifier()
            clf.set_params(**model_config) 
            clf.fit(X_train, y_train)

            mlflow.sklearn.log_model(sk_model=clf,artifact_path="models", registered_model_name='LightGBM')
            mlflow.log_params(model_config)    

            # predict the results on training dataset
            y_pred=clf.predict(X_test)

            # # view accuracy
            # acc=accuracy_score(y_pred, y_test)
            # conf_mat = confusion_matrix(y_pred, y_test)
            # mlflow.log_metric('test_accuracy', acc)
            # mlflow.log_metric('confustion matrix', conf_mat)
            
            
            #Log metrics
            acc=accuracy_score(y_pred, y_test)
            conf_mat = confusion_matrix(y_pred, y_test)
            precision = precision_score(y_pred, y_test,average= 'macro')
            recall = recall_score(y_pred, y_test, average= 'macro')
            f1 = f1_score(y_pred, y_test, average='macro')
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[0][0]
            fn = cm[1][0]
            tp = cm[1][1]
            fp = cm[0][1]
            class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
            class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)

            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Precision_0", class_zero[0])
            mlflow.log_metric("Precision_1", class_one[0])
            mlflow.log_metric("Recall_0", class_zero[1])
            mlflow.log_metric("Recall_1", class_one[1])
            mlflow.log_metric("f1_0", class_zero[2])
            mlflow.log_metric("f1_1", class_one[2])
            mlflow.log_metric("False Negative", fn)
            mlflow.log_metric("True Negative", tn)
            # mlflow.log_metric("f1", f1_score)

            runID = run.info.run_uuid
            print("Inside MLflow Run with id {}".format(runID))
    else:
        print("Not Required......Skipping")


def get_train_model_hptune(db_path,db_file_name,drfit_db_name):
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['Model_Training_hpTunning'][0] == 1:
        cnx = sqlite3.connect(db_path+db_file_name)
        X = pd.read_sql('select * from X', cnx)
        y = pd.read_sql('select * from y', cnx)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


        categoricals = []
        indexes_of_categories = [train.columns.get_loc(col) for col in categoricals]

         #Model Training
        gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y) # startifyKFold 

        gridParams = {
            'learning_rate': [0.005, 0.01,0.1],
            'n_estimators': [8,16,24,50],
            'num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting
            'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
            'objective' : ['binary'],
            'max_bin':[255, 510], # large max_bin helps improve accuracy but might slow down training progress
            'random_state' : [500],
            'colsample_bytree' : [0.64, 0.65, 0.66],
            'subsample' : [0.7,0.75],
            'reg_alpha' : [1,1.2],
            'reg_lambda' : [1,1.2,1.4],
            'max_depth': [1,3,5]
            }

        model_params = {
            'objective':'binary', 
            'num_boost_round':200, 
            'metric':'f1',
            'categorical_feature':indexes_of_categories,
            'verbose':-1,
            'force_row_wise':True
                       }

        lgb_estimator = lgb.LGBMClassifier()
        lgb_estimator.set_params(**model_params)

        gsearch = BayesSearchCV(estimator=lgb_estimator, search_spaces=gridParams, cv=gkf,n_iter=32,random_state=0,n_jobs=-1,verbose=-1,scoring='f1')
        lgb_model = gsearch.fit(X, y)
        best_model = lgb_model.best_estimator_
        f1_score = lgb_model.best_score_
        for p in gridParams:
            print(f"Best {p} : {best_model.get_params()[p]}")


        timestamp = str(int(time.time()))
        with mlflow.start_run(run_name=f"LGBM_Bayes_Search_{timestamp}") as run:
            y_pred = best_model.predict(X_test)

            # Log model
            mlflow.sklearn.log_model(best_model,registered_model_name='LightGBM',artifact_path='models')
            # mlflow.mlflow_log_artifact(best_model, artifact_path ="sqlite:///database/mlflow_v01.db")


            # Log params
            model_params = best_model.get_params()
            [mlflow.log_param(p, model_params[p]) for p in gridParams]

            #Log metrics
            acc=accuracy_score(y_pred, y_test)
            conf_mat = confusion_matrix(y_pred, y_test)
            precision = precision_score(y_pred, y_test,average= 'macro')
            recall = recall_score(y_pred, y_test, average= 'macro')
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[0][0]
            fn = cm[1][0]
            tp = cm[1][1]
            fp = cm[0][1]
            class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
            class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)

            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric("f1", f1_score)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Precision_0", class_zero[0])
            mlflow.log_metric("Precision_1", class_one[0])
            mlflow.log_metric("Recall_0", class_zero[1])
            mlflow.log_metric("Recall_1", class_one[1])
            mlflow.log_metric("f1_0", class_zero[2])
            mlflow.log_metric("f1_1", class_one[2])
            mlflow.log_metric("False Negative", fn)
            mlflow.log_metric("True Negative", tn)
            # mlflow.log_metric("f1", f1_score)

            runID = run.info.run_uuid
            print("Inside MLflow Run with id {}".format(runID))
            
    else:
        print("Not Required......Skipping")
    

#'runs:/e220f226ee624a79996e049c81924ec1/models' example:
def get_predict(db_path,db_file_name,ml_flow_path,drfit_db_name):
    cnx_drift = sqlite3.connect(db_path+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['Prediction'][0] == 1:
        mlflow.set_tracking_uri("http://0.0.0.0:6006")
        cnx = sqlite3.connect(db_path+db_file_name)
        logged_model = ml_flow_path
        # Load model as a PyFuncModel.
        loaded_model = mlflow.sklearn.load_model(logged_model)
        # Predict on a Pandas DataFrame.
        X = pd.read_sql('select * from X', cnx)
        predictions_proba = loaded_model.predict_proba(pd.DataFrame(X))
        predictions = loaded_model.predict(pd.DataFrame(X))
        pred_df = X.copy()
        
        pred_df['churn'] = predictions
        pred_df[["Prob of Not Churn","Prob of Churn"]] = predictions_proba
        index_msno_mapping = pd.read_sql('select * from index_msno_mapping', cnx)
        pred_df['index_for_map'] = pred_df.index
        final_pred_df = pred_df.merge(index_msno_mapping, on='index_for_map') 
        final_pred_df.to_sql(name='predictions', con=cnx,if_exists='replace',index=False)
        print (pd.DataFrame(predictions_proba,columns=["Prob of Not Churn","Prob of Churn"]).head()) 
        # pd.DataFrame(predictions,columns=["Prob of Not Churn","Prob of Churn"]).to_sql(name='Final_Predictions', con=cnx,if_exists='replace',index=False)
        return "Predictions are done and save in Final_Predictions Table"
    else:
        print("Not Required......Skipping")


def get_change(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')
    except TypeError:
        return 0
    
def get_reset_process_flags():
    return {
            'load_data': 0,
            'process_transactions': 0,
            'process_members': 0,
            'process_userlogs': 0,
            'merge_data': 0, 
            'Data_Preparation': 0, 
            'Model_Training_plain': 0, 
            'Model_Training_hpTunning':0,
            'Prediction':0
           }


def get_reset_process_flags_flip():
    return {
            'load_data': 1,
            'process_transactions': 1,
            'process_members': 1,
            'process_userlogs': 1,
            'merge_data': 1, 
            'Data_Preparation': 1, 
            'Model_Training_plain': 1, 
            'Model_Training_hpTunning':1,
            'Prediction':1
           }

def get_flush_db_process_flags(db_path,drfit_db_name,flip=True):
    if flip:
        cnx = sqlite3.connect(db_path+drfit_db_name)
        process_flags = get_reset_process_flags_flip()
        process_flags_df = pd.DataFrame(process_flags,index=[0])
        process_flags_df.to_sql(name='process_flags', con=cnx, if_exists='replace', index=False)
    else:
        cnx = sqlite3.connect(db_path+drfit_db_name)
        process_flags = get_reset_process_flags()
        process_flags_df = pd.DataFrame(process_flags,index=[0])
        process_flags_df.to_sql(name='process_flags', con=cnx, if_exists='replace', index=False)

def get_difference(df):
    percnt_df = get_change(df['new'], df['old'])
    return percnt_df
    
def get_data_drift(current_data, old_data, column_list,exclude_list, cnx, metric='std'):
    drift_dict = {}
    drift_dict['old'] = {}
    drift_dict['new'] = {}
    std_deviation_percentage = []
    mean_deviation_percentage = []
    
    for eachCol in column_list:
        if metric == 'std' and eachCol not in exclude_list:
            std1 = current_data[eachCol].std()
            drift_dict['new'][eachCol] = std1
            std2 = old_data[eachCol].std()
            drift_dict['old'][eachCol] = std2
            std_deviation_percentage.append(get_change(std1, std2))
        elif metric =='mean'and eachCol not in exclude_list:
            mean1 = current_data[eachCol].mean()
            drift_dict['new'][eachCol] = mean1
            mean2 = old_data[eachCol].mean()
            drift_dict['old'][eachCol] = mean2
            mean_deviation_percentage.append(get_change(mean1, mean2))
    # print(std_deviation_percentage,mean_deviation_percentage)
    
    #Drift Dict Saving
    print(drift_dict)
    df = pd.DataFrame(drift_dict)
    df['prcnt_difference'] = df.apply(get_difference,axis=1) 
    df['column_name'] = df.index
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['time'] = timestamp
    print(df)
    df.to_sql(name='drift_df', con=cnx, if_exists='append', index=False)
    
    if metric == 'std':
        return np.mean(std_deviation_percentage)
    elif metric == 'mean':
        return np.mean(mean_deviation_percentage)

def get_drift(old_data_directory,new_data_directory,db_path,drfit_db_name,
              metric='std', start_data='2017-03-01', end_date='2017-03-31'):
    
    cnx = sqlite3.connect(db_path+drfit_db_name)
    
    #New Data 
    march_user_logs, march_transactions = get_new_data_appended(old_data_directory,new_data_directory, start_data, end_date)
    march_user_logs, pre_size, post_size = compress_dataframes([march_user_logs])[0]
    march_transactions, pre_size, post_size = compress_dataframes([march_transactions])[0]

    #Old Data 
    transactions = load_data( [f"{old_data_directory}transactions_logs.csv",
                       ]
                     )[0]
    transactions['transaction_date'] = fix_time_in_df(transactions, 'transaction_date', expand=False)
    transactions['membership_expire_date'] = fix_time_in_df(transactions, 'membership_expire_date', expand=False)

    
    user_logs = load_data( [f"{old_data_directory}userlogs.csv",
                       ]
                     )[0]
    user_logs['date'] = fix_time_in_df(user_logs, 'date', expand=False)
    
    transactions, pre_size, post_size = compress_dataframes([transactions])[0]
    user_logs, pre_size, post_size = compress_dataframes([user_logs])[0]

    #Print Statements
    column_list_tran = list(transactions.select_dtypes(include=['int8','int16','int32','float16']).columns)
    print(column_list_tran)
    column_list_userlogs = list(user_logs.select_dtypes(include=['int8','int16','int32','float16']).columns)
    print(column_list_tran)
    exclude_list_tran = ['date'] 
    exclude_list_user_log = ['transaction_date','membership_expire_date']
    print("User Logs Data Drift as ", metric, " is: ", get_data_drift(march_user_logs, user_logs, column_list_userlogs,exclude_list_user_log, cnx, metric='std'))
    print("Transaction Data Drift as ", metric, " is: ", get_data_drift(march_transactions, transactions, column_list_tran,exclude_list_tran, cnx,  metric='std'))
    
    # drift = pd.DataFrame(
    #                 {
    #             'drift_userlog': get_data_drift(march_user_logs, user_logs, column_list_userlogs,exclude_list_user_log, cnx, metric),
    #             'drift_transaction':get_data_drift(march_transactions, transactions, column_list_tran,exclude_list_tran, cnx, metric)
    #                 },
    #                 index=[0]
    #                     )
    # print(drift)
    
    #to test other metric 
#     drift = pd.DataFrame(
#                     {
#                 'drift_userlog': 5.6,
#                 'drift_transaction':7.8
#                     },
#                     index=[0]
#                         )
    
#     drift = pd.DataFrame(
#                     {
#                 'drift_userlog': 24.5,
#                 'drift_transaction':28.30
#                     },
#                     index=[0]
#                         )
    
    drift = pd.DataFrame(
                    {
                'drift_userlog': 50.0,
                'drift_transaction':48.0
                    },
                    index=[0]
                        )
    
    # Save this in Database 
    
    print(drift)
    #Building & Checking Databases
    # build_dbs(db_path, drfit_db_name)
    drift.to_sql(name='drift', con=cnx, if_exists='replace', index=False)
    print("Writing to Database Done... at", db_path+drfit_db_name)
    get_drift_trigger(db_path, drfit_db_name)


def get_drift_trigger(db_path, drfit_db_name):
    cnx = sqlite3.connect(db_path+drfit_db_name)
    
    process_flags = get_reset_process_flags()
    
    print ("Before Change process_flags", process_flags) 

    drift = pd.read_sql('select * from drift', cnx)
    drift_value = drift.mean(axis=1)[0]
    print ("Drift_value.......", drift_value) 
    
    # 0-10 --> No Change. Just Inference/predictions
    # 10-20 --> Previos Model just retrained on New Data. Old Data + New Data 
    # 20-30 --> Hyper Parameter Tunning on same model.
    # 30+   --> Repeat Notebook
    
    if drift_value >= 0 and drift_value <=10:
        # process_flags['Prediction'] = 1
        print("No Change since drift is low")
        
    elif drift_value >= 10 and drift_value <=20:
        process_flags['load_data'] = 1
        process_flags['process_transactions'] = 1
        process_flags['process_members'] = 1
        process_flags['process_userlogs'] = 1
        process_flags['merge_data'] = 1
        process_flags['Data_Preparation'] = 1
        process_flags['Model_Training_plain'] = 1
    
    elif drift_value >= 20 and drift_value <=30:
        process_flags['load_data'] = 1
        process_flags['process_transactions'] = 1
        process_flags['process_members'] = 1
        process_flags['process_userlogs'] = 1
        process_flags['merge_data'] = 1
        process_flags['Data_Preparation'] = 1
        process_flags['Model_Training_hpTunning'] = 1
    else:
        print("Drift is very High, Please do dev/notebook again")
        
    print ("After Change process_flags", process_flags) 
    process_flags_df = pd.DataFrame(process_flags,index=[0])
    process_flags_df.to_sql(name='process_flags', con=cnx, if_exists='replace', index=False)
        
        
        
        
        
        
    
    