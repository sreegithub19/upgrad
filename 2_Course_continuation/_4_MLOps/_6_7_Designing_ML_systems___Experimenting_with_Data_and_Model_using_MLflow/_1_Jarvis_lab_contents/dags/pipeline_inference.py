# Building the DAG using the functions from data_process and model module
import datetime as dt
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from constants_inference import *
import os 
import sqlite3
from sqlite3 import Error
import pandas as pd
import importlib.util
from datetime import datetime
from airflow.operators.email_operator import EmailOperator


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils", "/home/scripts/utils.py")

# Declare Default arguments for the DAG
default_args = {
    'owner': 'upgrad_demo',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'provide_context': True
}


# creating a new dag
dag = DAG('Inference', default_args=default_args, schedule_interval='0 0 * * 2', max_active_runs=1,tags=['inference_pipeline'])

# Integrating different operatortasks in airflow dag

op_reset_processes_flags = PythonOperator(task_id='reset_processes_flag',
                                         python_callable=utils.get_flush_db_process_flags,
                                         op_kwargs={'db_path': db_path,'drfit_db_name':drfit_db_name},
                                         dag=dag)


op_create_db = PythonOperator(task_id='create_check_db', 
                            python_callable=utils.build_dbs,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name},
                            dag=dag)


op_load_data = PythonOperator(task_id='load_data', 
                                python_callable=utils.load_data_from_source,
                                  op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name,
                                             'old_data_directory':old_data_directory,
                                             'new_data_directory':new_data_directory,
                                            'run_on':run_on,
                                              'start_date':start_date,
                                             'end_date':end_date,
                                            'append':append},
                              dag=dag)


op_process_transactions = PythonOperator(task_id='process_transactions',
                                         python_callable=utils.get_membership_data_transform,
                                         op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                                         dag=dag)

op_process_members = PythonOperator(task_id='process_members', 
                                    python_callable=utils.get_transaction_data_transform,
                                    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                                    dag=dag)

op_process_userlogs = PythonOperator(task_id='process_userlogs',
                                    python_callable=utils.get_user_data_transform,
                                    op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                                    dag=dag)

op_merge = PythonOperator(task_id='merge_data',
                        python_callable=utils.get_final_data_merge,
                        op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,
                                             'drfit_db_name':drfit_db_name},
                        dag=dag)


op_process_data = PythonOperator(task_id='data_preparation', 
                            python_callable=utils.get_data_prepared_for_modeling,
                            op_kwargs={'db_path': db_path,
                                       'db_file_name': db_file_name,
                                       'drfit_db_name':drfit_db_name,
                                       'date_columns':date_columns,
                                       'date_transformation':date_transformation
                                      },
                            dag=dag)

op_predict_data = PythonOperator(task_id='Prediction', 
                            python_callable=utils.get_predict,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name,'ml_flow_path':ml_flow_model_path,'drfit_db_name':drfit_db_name},
                            dag=dag)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email = EmailOperator( task_id='send_email', 
                                to='@gmail.com', 
                                subject='Inference Pipeline Execution Finished', 
                                html_content=f"The execution of Inference pipeline is finished @ {timestamp}. Check Airflow Logs for more details or check SQLite Backend for Predictions.", 
                                dag=dag)



# Set the task sequence
op_reset_processes_flags.set_downstream(op_create_db)
op_create_db.set_downstream(op_load_data)
op_load_data.set_downstream([op_process_members,op_process_userlogs,op_process_transactions])
op_process_members.set_downstream(op_merge)
op_process_userlogs.set_downstream(op_merge)
op_process_transactions.set_downstream(op_merge)
op_merge.set_downstream(op_process_data)
op_process_data.set_downstream(op_predict_data)
op_predict_data.set_downstream(send_email)