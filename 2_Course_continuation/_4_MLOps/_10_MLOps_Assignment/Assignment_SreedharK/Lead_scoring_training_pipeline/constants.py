DB_PATH = '/home/airflow/dags/Lead_scoring_data_pipeline/'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db' 

DB_FILE_MLFLOW = 'Lead_scoring_mlflow_production.db'

TRACKING_URI = "http://0.0.0.0:6006"
EXPERIMENT = "Lead_scoring_mlflow_production"


# model config imported from pycaret experimentation
MODEL_CONFIG = {
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

# list of the features that needs to be there in the final encoded dataframe
# ONE_HOT_ENCODED_FEATURES = ['total_leads_droppped', 'city_tier', 'first_platform_c_Level8', 'first_platform_c_Level2', 'first_platform_c_others', 'first_platform_c_Level0',
#                            'first_platform_c_Level7', 'first_utm_medium_c_others', 'first_utm_medium_c_Level13', 'first_utm_source_c_Level6', 'app_complete_flag']

ONE_HOT_ENCODED_FEATURES = ['total_leads_droppped','city_tier','first_platform_c_Level8', 'first_platform_c_others','first_platform_c_Level2','first_utm_medium_c_others','first_utm_medium_c_Level13', 'first_utm_medium_c_Level0','first_platform_c_Level7', 'first_platform_c_Level0', 'app_complete_flag']

# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = ['first_platform_c', 'first_utm_medium_c','first_utm_source_c']
