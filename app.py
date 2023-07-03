import streamlit as st
import pandas as pd
import utils
from models import Model
import matplotlib.pyplot as plt
import seaborn as sns

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
        loaded= ''
    else:
        st.error('Please upload a CSV, Excel or Pickle file.')


    if 'df' in locals():
        DF= df.copy()
        cols= DF.columns
        target = st.selectbox('Select The Target', cols)
        task_type = st.radio("Select Task Type", ["Classification", "Regression"], index=0, help="Select the task type")

        if task_type == "Classification":
            st.write("Classification task selected")

            clean = st.selectbox("Clean Data", ["Remove Missing Data", "Impute Missing Data"])
            outlier = st.selectbox("Remove Outliers", ["Don't Remove Outliers", "Use IQR", "Use Isolation Forest"])
            model = st.selectbox("Select The Model", ["SVM", "LR"])

        elif task_type == "Regression":
            st.write("Regression task selected")

        else:
            raise ValueError("Invalid Selection")
        
        def process_data(_df):
            DF= utils.missing(_df, clean)
            DF= utils.remove_outliers(DF, outlier)
            return DF
        
        if st.button('Apply'):
            if task_type == "Classification":
                st.write('Perform classification task with option:')
                DF= process_data(DF)
                st.write(DF)

            
            if task_type == "Regression":
                st.write('Perform regression task with option')

        if st.button('Plot Graphs'):
            DF= process_data(DF)
            st.subheader('Histogram')
            fig, ax = plt.subplots()
            ax.hist(DF['Age'])
            st.pyplot(fig)

            # Display a scatter plot
            st.subheader('Scatter Plot')
            fig, ax = plt.subplots()
            sns.scatterplot(data=DF, x='Age', y=target)
            st.pyplot(fig)

    if 'loaded' in locals():
        st.write("laoded model")




