root_folder = "/home/"
old_data_directory = root_folder+"data/raw/"
new_data_directory = root_folder+"data/new/"
intermediate_path = root_folder+"data/interim/"
db_path = root_folder+"database/"
db_file_name = "feature_store_v01.db"
date_columns = ['registration_init_time','transaction_date_min','transaction_date_max','membership_expire_date_max','last_login']
run_on = "old" #"old" or "new" --> new here will append new data to old data. 
date_transformation = False
start_date = '2017-03-01'
end_date = '2017-03-31'
drfit_db_name = "drift_db_name.db"
append = True # need to put run_on = new. for append to work. 
