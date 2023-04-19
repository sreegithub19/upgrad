import streamlit as st



st.title("My First Web Application")
st.header("ML Pipelines") 

tab1, tab2 = st.tabs(['option1','option2']) 

with tab1:
    st.write("WElcome to Tab 1") 
    options = st.multiselect(
    'What are your favorite colors',
    ['Green', 'Yellow', 'Red', 'Blue'],
    ['Yellow', 'Red'])
    st.write('You selected:', options)
    

with tab2:
    st.write("WElcome to Tab 2") 