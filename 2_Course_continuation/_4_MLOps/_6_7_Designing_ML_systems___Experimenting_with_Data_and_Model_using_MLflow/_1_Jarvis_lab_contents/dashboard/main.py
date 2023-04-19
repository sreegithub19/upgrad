import streamlit as st
import pandas as pd
from airflow.api.client.local_client import Client
import sqlite3
from sqlite3 import Error
from pathlib import Path
import time
from airflow.models import DagRun
from datetime import datetime
import mlflow
import importlib.util
from mlflow import MlflowClient
from io import StringIO 
mlflow.set_tracking_uri("http://0.0.0.0:6006")
client = MlflowClient()


def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.cache
def get_final_features(db_path, db_file_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    df = pd.read_sql('select * from X', cnx)
    return df

@st.cache
def get_final_predictions(db_path, db_file_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    df = pd.read_sql('select * from predictions', cnx)
    return df

@st.cache
def get_final_drift(db_path, db_file_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    df = pd.read_sql('select * from drift_df', cnx)
    return df

def get_most_recent_dag_run(dag_id):
    dag_runs = DagRun.find(dag_id=dag_id)
    dag_runs.sort(key=lambda x: x.execution_date, reverse=True)
    return dag_runs[0] if dag_runs else None

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




st.title("MLOps Pipeline for Customer Churn")
st.header("Pipelines")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Processing", "Model Building", "Inference", "Drift & Monitoring", "Predict"])


with tab1:
    data_const = module_from_file("utils", "/home/dags/constants_data_pipeline.py")
    st.write("The property file for pipeline...please change it here :/home/dags/constants_data_pipeline.py")
    st.code(Path("/home/dags/constants_data_pipeline.py").read_text(), language="python")
    dag_run = get_most_recent_dag_run('Data_End2End_Processing')
    if dag_run:
        st.write(f'The most recent Data_End2End_Processing DagRun was executed at: {dag_run.execution_date}') 
        current_time = datetime.now().strftime("%H:%M:%S")
        st.write("Current System  Time =", current_time)
    if st.button('Trigger Data Pipeline'):
        c = Client(None, None)
        timestamp = str(int(time.time()))
        st.write("The Pipeline is executed. Please wait for the email to arrive and check Airflow")
        c.trigger_dag(dag_id='Data_End2End_Processing', run_id=f'Data_End2End_Processing_{timestamp}', conf={})
        
        #
    try:
        features = get_final_features(data_const.db_path, data_const.db_file_name)
        number = st.number_input('How many rows you want to export?',min_value=10,step=1)

        st.write("**Only run these steps, if the pipeline is finished**")
        if st.button("Check Results!"):
            st.write(features.head(10))

        if st.button("Downlaod Results!"):
            csv = convert_df(features.head(number))
            st.download_button("Press to Download",
                               csv,
                               "file.csv",
                               "text/csv",
                               key='download-csv'
                            )
    except:
        st.write("The Pipeline hasn't been run or backend Database is not ready...wait and re-run.") 
        pass
        

with tab2:
    st.write("The property file for pipeline...please change it here :/home/dags/constants_model_building.py")
    st.code(Path("/home/dags/constants_model_building.py").read_text(), language="python")
    dag_run = get_most_recent_dag_run('Model_Building_Pipeline')
    if dag_run:
        st.write(f'The most recent Model_Building_Pipeline DagRun was executed at: {dag_run.execution_date}') 
        current_time = datetime.now().strftime("%H:%M:%S")
        st.write("Current System  Time =", current_time)
    else:
        st.write("The DAG has not been run so far")
    if st.button('Trigger Model Building Pipeline'):
        c = Client(None, None)
        timestamp = str(int(time.time()))
        st.write("The Pipeline is executed. Please wait for the email to arrive and check Airflow & MLFlow")
        c.trigger_dag(dag_id='Model_Building_Pipeline', run_id=f'Model_Building_Pipeline_{timestamp}', conf={})
    
    if st.button("Check all available models"):
        client = MlflowClient()
        model_name="LightGBM"
        filter_string = "name='{}'".format(model_name)
        results = client.search_registered_models(filter_string=filter_string)
        model_result = []
        for res in results:
            for mv in res.latest_versions:
                model_result.append([mv.name, mv.run_id, mv.version,mv.current_stage,
                                                                                mv.creation_timestamp,
                                                                                mv.description,
                                                                                mv.source]) 
        st.write(pd.DataFrame(model_result,columns=['name', 'run_id','version' ,'stage', 'time' ,'description', 'source']))
        
with tab3:
    inf_const = module_from_file("utils", "/home/dags/constants_inference.py")
    st.write("The property file for pipeline...please change it here :/home/dags/constants_inference.py")
    st.code(Path("/home/dags/constants_inference.py").read_text(), language="python")
    dag_run = get_most_recent_dag_run('Inference')
    if dag_run:
        st.write(f'The most recent Inference DagRun was executed at: {dag_run.execution_date}') 
        current_time = datetime.now().strftime("%H:%M:%S")
        st.write("Current System  Time =", current_time)
    else:
        st.write("The DAG has not been run so far")
    if st.button('Trigger Inference Pipeline'):
        c = Client(None, None)
        timestamp = str(int(time.time()))
        st.write("The Pipeline is executed. Please wait for the email to arrive and check Airflow & SQLite")
        c.trigger_dag(dag_id='Inference', run_id=f'Inference_Pipeline_{timestamp}', conf={})
        
    try:
        inf_results = get_final_predictions(inf_const.db_path, inf_const.db_file_name) #
        infer_number = st.number_input('How many rows you want to export ?',min_value=10,step=1)

        st.write("**Only run these steps, if the pipeline is finished**")
        if st.button("Check Results! "):
            st.write(inf_results.head(10))

        if st.button("Downlaod All Results!"):
            csv = convert_df(inf_results.head(infer_number))
            st.download_button("Press to Download",
                               csv,
                               "file.csv",
                               "text/csv",
                               key='download-csv'
                            )
        if st.button("Downlaod Churn Users!"):
            subset_df = inf_results.head(number)
            subset_df = subset_df[subset_df['churn']==1] 
            st.write(subset_df)
            csv = convert_df(subset_df)
            st.download_button("Press to Download",
                               csv,
                               "file.csv",
                               "text/csv",
                               key='download-csv'
                            )
    except:
        st.write("The Pipeline hasn't been run or backend Database is not ready...wait and re-run.") 
        pass
    
with tab4:
    drift_const = module_from_file("utils", "/home/dags/constants_drift.py")
    st.write("The property file for pipeline...please change it here :/home/dags/constants_drift.py")
    st.code(Path("/home/dags/constants_drift.py").read_text(), language="python")
    dag_run = get_most_recent_dag_run('Drift_Pipeline')
    if dag_run:
        st.write(f'The most recent Drift_Pipeline DagRun was executed at: {dag_run.execution_date}') 
        current_time = datetime.now().strftime("%H:%M:%S")
        st.write("Current System  Time =", current_time)
    else:
        st.write("The DAG has not been run so far")
    if st.button('Trigger Drift_Pipeline Pipeline'):
        c = Client(None, None)
        timestamp = str(int(time.time()))
        st.write("The Pipeline is executed. Please wait for the email to arrive and check Airflow & MLFlow")
        c.trigger_dag(dag_id='Drift_Pipeline', run_id=f'Drift_Pipeline_{timestamp}', conf={})
        
    try:
        features = get_final_drift(drift_const.db_path, drift_const.drfit_db_name)
        # number = st.number_input('How many rows you want to export?',min_value=10,step=1)

        st.write("**Only run these steps, if the pipeline is finished**")
        if st.button("Check Drift Results!"):
            st.write(features)

        if st.button("Downlaod Drift Results!"):
            csv = convert_df(features)
            st.download_button("Press to Download Drift Results",
                               csv,
                               "file.csv",
                               "text/csv",
                               key='download-csv'
                                )
    except:
        st.write("The Pipeline hasn't been run or backend Database is not ready...wait and re-run.") 
        pass
        

with tab5:
    # c = Client(None, None)
    st.write("Upload a File to get the prediction.......")
    # c.trigger_dag(dag_id='Data_End2End_Processing', run_id='Data_End2End_Processing', conf={})

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file,index_col=[0])
        st.write(dataframe)
    
    inf_const = module_from_file("utils", "/home/dags/constants_inference.py")
    st.write("The property file for pipeline...please change it here :/home/dags/constants_inference.py")
    st.code(Path("/home/dags/constants_inference.py").read_text(), language="python")
    dag_run = get_most_recent_dag_run('Inference')
    if dag_run:
        st.write(f'The most recent Inference DagRun was executed at: {dag_run.execution_date}') 
        current_time = datetime.now().strftime("%H:%M:%S")
        st.write("Current System  Time =", current_time)
    else:
        st.write("The DAG has not been run so far")
    
    try:
        mlflow.set_tracking_uri("http://0.0.0.0:6006")
        cnx = sqlite3.connect(inf_const.db_path+inf_const.db_file_name)
        # Load model as a PyFuncModel.
        loaded_model = mlflow.sklearn.load_model(inf_const.ml_flow_model_path)
        # Predict on a Pandas DataFrame.
        # X = pd.read_sql('select * from X', cnx)
        predictions_proba = loaded_model.predict_proba(pd.DataFrame(dataframe))
        predictions = loaded_model.predict(pd.DataFrame(dataframe))
        pred_df = dataframe.copy()
        pred_df['churn'] = predictions
        pred_df[["Prob of Not Churn","Prob of Churn"]] = predictions_proba
        # final_pred_df.to_sql(name='predictions', con=cnx,if_exists='replace',index=False)
        # print (pd.DataFrame(predictions_proba,columns=["Prob of Not Churn","Prob of Churn"]).head())
        if st.button(" Predictions!"):
                # subset_df = inf_results.head(number)
                st.write(pred_df.head())
                csv = convert_df(pred_df)
                st.download_button("Press to Download",
                                   csv,
                                   "file.csv",
                                   "text/csv",
                                   key='download-csv'
                                )
    except:
        st.write("Model Path is not correct, Please change the config in constant file")
    
    if st.button("Churn Users!"):
            # subset_df = inf_results.head(number)
            subset_df = pred_df[pred_df['churn']==1] 
            st.write(subset_df)
            csv = convert_df(subset_df)
            st.download_button("Press to Download",
                               csv,
                               "file.csv",
                               "text/csv",
                               key='download-csv'
                            )
