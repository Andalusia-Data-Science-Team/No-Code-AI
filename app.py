import streamlit as st
import pandas as pd, numpy as np
import utils
from models import model, inference
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.runtime.scriptrunner import get_script_run_ctx as get_report_ctx
from streamlit.web.server import Server
import pickle

uploaded_file = st.file_uploader("Upload Data/Model")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension.lower() == 'csv':
        df = pd.read_csv(uploaded_file)
        st.write(df)
    elif file_extension.lower() in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
        st.write(df)
    elif file_extension.lower() == 'pkl':
        try:
            _model = pickle.load(uploaded_file)
            st.success("Pickle file loaded successfully!")
        except pickle.UnpicklingError as e:
            st.error("Error: Invalid pickle file. Please upload a valid pickle file.")
    else:
        st.error('Please upload a CSV, Excel or Pickle file.')


    if 'df' in locals():
        DF= df.copy()
        cols= DF.columns
        target = st.selectbox('Select The Target', cols)
        selected_options = st.multiselect('Select columns to be removed', cols)
        DF= DF.drop(selected_options, axis= 1)

        value = st.slider("Select validation size %validation", min_value=1, max_value=100, step=1)
        task_type = st.radio("Select Task Type", ["Classification", "Regression"], index=0, help="Select the task type")

        if task_type == "Classification":
            st.write("Classification task selected")

            clean = st.selectbox("Clean Data", ["Remove Missing Data", "Impute Missing Data"])
            outlier = st.selectbox("Remove Outliers", ["Don't Remove Outliers", "Use IQR", "Use Isolation Forest"])
            alg = st.selectbox("Select The Model", ["SVC", "LR"])

        elif task_type == "Regression":
            st.write("Regression task selected")

            clean = st.selectbox("Clean Data", ["Remove Missing Data", "Impute Missing Data"])
            outlier = st.selectbox("Remove Outliers", ["Don't Remove Outliers", "Use IQR", "Use Isolation Forest"])
            alg = st.selectbox("Select The Model", ["Linear Regression", "ElasticNet"])
        else:
            raise ValueError("Invalid Selection")
        
        def process_data(_df, all= False):
            DF= utils.missing(_df, clean)
            DF= utils.remove_outliers(DF, outlier)
            if all:
                return DF

            X_train, X_test, y_train, y_test= utils.handle(DF, target, task_type)
            return X_train, X_test, y_train, y_test
        
        if st.button('Apply'):
            if task_type == "Classification":
                st.write('Perform classification task with option:')
                X_train, X_test, y_train, y_test= process_data(DF)
                report= model(X_train, X_test, y_train, y_test, alg, save= True, task= task_type)
                st.write("Accuracy:")
                st.write(report[0])
                st.write("Confusion Matrix")
                st.write(report[1])

            
            if task_type == "Regression":
                st.write('Perform Regression task with option:')
                X_train, X_test, y_train, y_test= process_data(DF)
                report= model(X_train, X_test, y_train, y_test, alg, save= True, task= task_type)
                st.write("MSE:")
                st.write(report)


        if st.button('Plot Graphs'):
            DF= process_data(DF, all= True)
            st.subheader('Heat Map')
            fig, ax = plt.subplots()
            plot_data= utils.HeatMap(DF)
            sns.heatmap(plot_data,ax = ax,cmap ="YlGnBu", linewidths = 0.1)
            st.pyplot(fig)

            # Display a scatter plot
            st.subheader('Correlation')
            corr_values= utils.corr_plot(DF)
            st.write(corr_values.sort_values(by= 'abs_correlation', ascending= False))
            fig, ax = plt.subplots()
            ax = corr_values.abs_correlation.hist(bins=50, figsize=(12, 8))
            ax.set(xlabel='Absolute Correlation', ylabel='Frequency')
            st.pyplot(fig)


        st.subheader('Inference')
        inf_df = pd.DataFrame(columns=DF.drop([target], axis= 1).columns)
        for column in inf_df.columns:
            user_input = st.text_input(column)
            user_input= utils.inf_proc(user_input)
            inf_df[column] = [user_input]
  
        st.write(inf_df)
        if st.button('Submit'):
            preds= inference(inf_df)
            st.write(preds)
            

        if st.button('Download Model'):
            with open('model.pkl', 'rb') as file:
                model_data = file.read()
            st.download_button(label="Download Model File", data=model_data, file_name="model.pkl")

    elif '_model' in locals():
        st.write("laoded model")
        st.warning("The Loaded Model assumes it has predict method and was train via sklearn")
        st.subheader('Inference')
        st.write('Example: Temperature,Humidity,Wind_Speed,...')
        cols= st.text_input('Please Enter the Columns Comma Seperated')
        cols_data= [item.strip() for item in cols.split(",")]
        inf_df = pd.DataFrame(columns=cols_data)
        for column in inf_df.columns:
            user_input = st.text_input(column)
            inf_df[column] = [user_input]
  
        st.write(inf_df)
        if st.button('Submit'):
            preds= _model.predict(inf_df)
            st.write(preds)

    else:
        raise ValueError('invalid')




