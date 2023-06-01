##############################################################################
# Import necessary modules
# #############################################################################


from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime as dt
from datetime import datetime, timedelta
import sys
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file("utils", "/home/dags/Lead_scoring_data_pipeline/utils.py")
data_validation_checks = module_from_file("data_validation_checks", "/home/dags/Lead_scoring_data_pipeline/data_validation_checks.py")
constants = module_from_file("utils", "/home/dags/Lead_scoring_data_pipeline/constants.py")
schema = module_from_file("data_validation_checks", "/home/dags/Lead_scoring_data_pipeline/schema.py")
city_tier_mapping = module_from_file("utils", "/home/dags/Lead_scoring_data_pipeline/city_tier_mapping.py")
significant_categorical_level = module_from_file("data_validation_checks", "/home/dags/Lead_scoring_data_pipeline/significant_categorical_level.py")

#Load all variables to pass it to utils function
db_path=constants.DB_PATH
db_file_name=constants.DB_FILE_NAME
data_directory=constants.DATA_DIRECTORY
interaction_mapping=constants.INTERACTION_MAPPING
index_columns=constants.INDEX_COLUMNS
lead_scoring_file=constants.LEAD_SCORING_FILE

raw_data_schema=schema.raw_data_schema
model_input_schema=schema.model_input_schema
city_tier_mapping_dict=city_tier_mapping.city_tier_mapping_dict
list_platform=significant_categorical_level.list_platform
list_medium=significant_categorical_level.list_medium
list_source=significant_categorical_level.list_platform

###############################################################################
# Define default arguments and DAG
# ##############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


ML_data_cleaning_dag = DAG(
                dag_id = 'Lead_Scoring_Data_Engineering_Pipeline',
                default_args = default_args,
                description = 'DAG to run data pipeline for lead scoring',
                schedule_interval = '@daily',
                catchup = False
)

###############################################################################
# Create a task for build_dbs() function with task_id 'building_db'
# ##############################################################################
building_db = PythonOperator(task_id='building_db', 
                            python_callable=utils.build_dbs,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name},
                            dag=ML_data_cleaning_dag)
###############################################################################
# Create a task for raw_data_schema_check() function with task_id 'checking_raw_data_schema'
# ##############################################################################
raw_data_schema_check = PythonOperator(task_id='checking_raw_data_schema', 
                            python_callable=data_validation_checks.raw_data_schema_check,
                            op_kwargs={'data_directory': data_directory, 'raw_data_schema': raw_data_schema}, 
                            dag=ML_data_cleaning_dag)
###############################################################################
# Create a task for load_data_into_db() function with task_id 'loading_data'
# #############################################################################
load_data_into_db = PythonOperator(task_id='loading_data', 
                            python_callable=utils.load_data_into_db,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'data_directory': data_directory, 'lead_scoring_file': lead_scoring_file},
                            dag=ML_data_cleaning_dag)
###############################################################################
# Create a task for map_city_tier() function with task_id 'mapping_city_tier'
# ##############################################################################
map_city_tier = PythonOperator(task_id='mapping_city_tier', 
                            python_callable=utils.map_city_tier,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'city_tier_mapping': city_tier_mapping_dict},
                            dag=ML_data_cleaning_dag)
###############################################################################
# Create a task for map_categorical_vars() function with task_id 'mapping_categorical_vars'
# ##############################################################################
map_categorical_vars = PythonOperator(task_id='mapping_categorical_vars', 
                            python_callable=utils.map_categorical_vars,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'list_platform': list_platform, 'list_medium': list_medium, 'list_source': list_source},
                            dag=ML_data_cleaning_dag)
###############################################################################
# Create a task for interactions_mapping() function with task_id 'mapping_interactions'
# ##############################################################################
interactions_mapping = PythonOperator(task_id='mapping_interactions', 
                            python_callable=utils.interactions_mapping,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'interaction_mapping': interaction_mapping, 'index_columns': index_columns},
                            dag=ML_data_cleaning_dag)
###############################################################################
# Create a task for model_input_schema_check() function with task_id 'checking_model_inputs_schema'
# ##############################################################################
model_input_schema_check = PythonOperator(task_id='checking_model_inputs_schema', 
                            python_callable=data_validation_checks.model_input_schema_check,
                            op_kwargs={'db_path': db_path, 'db_file_name': db_file_name, 'model_input_schema': model_input_schema},
                            dag=ML_data_cleaning_dag)
###############################################################################
# Define the relation between the tasks
# ##############################################################################

build_dbs.set_downstream(raw_data_schema_check)
raw_data_schema_check.set_downstream(load_data_into_db)

load_data_into_db.set_downstream(map_city_tier)
map_city_tier.set_downstream(map_categorical_vars)
map_categorical_vars.set_downstream(interactions_mapping)

interactions_mapping.set_downstream(model_input_schema_check)
