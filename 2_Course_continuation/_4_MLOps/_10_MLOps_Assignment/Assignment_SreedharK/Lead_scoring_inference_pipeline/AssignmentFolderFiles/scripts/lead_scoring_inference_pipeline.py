##############################################################################
# Import necessary modules
# #############################################################################

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
# import * does not work causing some issues, loading modules as taught in live session
utils = module_from_file("utils", "/home/Assignment/03_inference_pipeline/scripts/utils.py")
constants = module_from_file("utils", "/home/Assignment/03_inference_pipeline/scripts/constants.py") 

db_path=constants.DB_PATH
db_file_name=constants.DB_FILE_NAME
scripts_output=constants.SCRIPTS_OUTPUT

db_file_mlflow=constants.DB_FILE_MLFLOW
tracking_uri=constants.TRACKING_URI

model_name=constants.MODEL_NAME
model_stage=constants.STAGE

one_hot_encoded_features=constants.ONE_HOT_ENCODED_FEATURES
features_to_encode=constants.FEATURES_TO_ENCODE

###############################################################################
# Define default arguments and create an instance of DAG
# ##############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


Lead_scoring_inference_dag = DAG(
                dag_id = 'Lead_scoring_inference_pipeline',
                default_args = default_args,
                description = 'Inference pipeline of Lead Scoring system',
                schedule_interval = '@hourly',
                catchup = False
)

###############################################################################
# Create a task for encode_data_task() function with task_id 'encoding_categorical_variables'
# ##############################################################################

encode_data_task = PythonOperator(task_id='encoding_categorical_variables', 
                            python_callable=utils.encode_features,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 
                                       'one_hot_encoded_features': one_hot_encoded_features, 'features_to_encode': features_to_encode},
                            dag=Lead_scoring_inference_dag)


###############################################################################
# Create a task for load_model() function with task_id 'generating_models_prediction'
# ##############################################################################

load_model = PythonOperator(task_id='generating_models_prediction', 
                            python_callable=utils.get_models_prediction,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 
                                       'model_name': model_name, 'model_stage': model_stage, 'tracking_uri': tracking_uri,},
                            dag=Lead_scoring_inference_dag)

###############################################################################
# Create a task for prediction_col_check() function with task_id 'checking_model_prediction_ratio'
# ##############################################################################

prediction_col_check = PythonOperator(task_id='checking_model_prediction_ratio', 
                            python_callable=utils.prediction_ratio_check,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 
                                       'script_output': scripts_output,},
                            dag=Lead_scoring_inference_dag)

###############################################################################
# Create a task for input_features_check() function with task_id 'checking_input_features'
# ##############################################################################

input_features_check = PythonOperator(task_id='checking_input_features', 
                            python_callable=utils.input_features_check,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 
                                       'one_hot_encoded_features': one_hot_encoded_features,},
                            dag=Lead_scoring_inference_dag)


###############################################################################
# Define relation between tasks
# ##############################################################################

encode_data_task.set_downstream(input_features_check)
input_features_check.set_downstream(load_model)
load_model.set_downstream(prediction_col_check)