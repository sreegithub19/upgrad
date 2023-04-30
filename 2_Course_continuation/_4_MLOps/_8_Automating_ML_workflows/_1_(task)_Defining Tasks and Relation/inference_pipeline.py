from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
from heart_disease.utils import *


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=30)
}


ML_inference_dag = DAG(
                dag_id = 'Heart_Disease_ML_dag',
                default_args = default_args,
                description = 'Dag to run inferences on predictions of heart disease patients',
                schedule_interval = '@hourly'
)

load_task = PythonOperator(
            task_id = 'load_task',
            python_callable = get_inference_data,
            dag = ML_inference_dag)

######
#Define task for encoding the categorial variables here
######

######
#Define task for normalising the variables here
######

######
#Define task for getting models prediction here
######

