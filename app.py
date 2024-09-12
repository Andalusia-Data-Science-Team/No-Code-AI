import streamlit as st
import pandas as pd, numpy as np, random
import utils
from models import model, inference, get_corresponding_labels
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib
from interpretability import shap_lime
import plotly.graph_objects as go
matplotlib.use('Agg')

uploaded_file = st.file_uploader("Upload Data/Model")
SEED = int(st.number_input('Enter a Seed', value=42))
st.write(f'Using {SEED} as a seed')
np.random.seed(SEED)


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
        cfg= {'save': True} # for inference stability it's fixed
        DF= df.copy()
        cols= DF.columns
        target = st.selectbox('Select The Target', cols)
        selected_options = st.multiselect('Select columns to be removed', cols)
        DF= DF.drop(selected_options, axis= 1)

        value = st.slider("Select validation size %validation", min_value=1, max_value=100, step=1)
        task_type = st.radio("Select Task Type", ["Regression", "Classification"], index=0, help="Select the task type")

        if task_type == "Classification":
            st.write("Classification task selected")
            cfg['task_type']= task_type
            cfg['clean'] = st.selectbox("Clean Data", ["Remove Missing Data", "Impute Missing Data"])
            cfg['outlier'] = st.selectbox("Remove Outliers", ["Don't Remove Outliers", "Use IQR", "Use Isolation Forest"])
            cfg['alg'] = st.selectbox("Select The Model", ["SVC", "LR", "KNN_cls", "RF_cls", "XGB_cls", 
                                                           "GradientBoosting_cls", "Adaboost", "DecisionTree_cls", "extra_tree"])
            cfg['skew_fix']= st.checkbox('Skew Fix')
            cfg['poly_feat']= st.checkbox('Add Polynomial Features')
            cfg['apply_GridSearch']= st.checkbox('Apply GridSearch')


        elif task_type == "Regression":
            st.write("Regression task selected")
            cfg['task_type']= task_type
            cfg['clean'] = st.selectbox("Clean Data", ["Remove Missing Data", "Impute Missing Data"])
            cfg['outlier'] = st.selectbox("Remove Outliers", ["Don't Remove Outliers", "Use IQR", "Use Isolation Forest"])
            cfg['alg'] = st.selectbox("Select The Model", ["Linear Regression", "ElasticNet", "KNN_reg", "XGB_reg", 
                                                           "Ridge", "Lasso", "SVR"])
            cfg['skew_fix']= st.checkbox('Skew Fix')
            cfg['poly_feat']= st.checkbox('Add Polynomial Features')
            cfg['apply_GridSearch']= st.checkbox('Apply GridSearch')


        else:
            raise ValueError("Invalid Selection")
        
        def process_data(_df, all= False):
            DF= utils.missing(_df, cfg['clean'])
            DF= utils.remove_outliers(DF, cfg['outlier'])
            if all:
                return DF

            X_train, X_test, y_train, y_test= utils.handle(DF, target, task_type)
            return X_train, X_test, y_train, y_test
        
        if st.button('Apply'):
            if task_type == "Classification":
                st.write('Perform classification task with option:')
                X_train, X_test, y_train, y_test= process_data(DF)
                report= model(X_train, X_test, y_train, y_test, cfg)
                st.write("Accuracy:")
                st.write(report[0])
                st.write("Confusion Matrix")
                st.write(report[1])

            
            if task_type == "Regression":
                st.write('Perform Regression task with option:')
                X_train, X_test, y_train, y_test= process_data(DF)
                report= model(X_train, X_test, y_train, y_test, cfg)
                st.write("MSE:")
                st.write(report[0])
                st.write("R2:")
                st.write(report[1])

        if st.button('Plot Graphs'):
            _DF= process_data(DF, all= True)
            st.subheader('Heat Map')
            fig, ax = plt.subplots()
            plot_data= utils.HeatMap(_DF)
            sns.heatmap(plot_data,ax = ax,cmap ="YlGnBu", linewidths = 0.1)
            st.pyplot(fig)

            # Display a scatter plot
            st.subheader('Correlation')
            corr_values= utils.corr_plot(_DF)
            st.write(corr_values.sort_values(by= 'abs_correlation', ascending= False))
            fig, ax = plt.subplots()
            ax = corr_values.abs_correlation.hist(bins=50, figsize=(12, 8))
            ax.set(xlabel='Absolute Correlation', ylabel='Frequency')
            st.pyplot(fig)


        checked_plots= {}
        inf_df = pd.DataFrame(columns=DF.drop([target], axis= 1).columns)
        _inf_df= inf_df.copy()
        selected_cols= inf_df.columns
        shap_df= DF[selected_cols]

        st.header("Xplain")
        X_train, X_test, y_train, y_test= process_data(DF)

        # 1. Feature Dependence
        st.subheader("Feature Dependence")
        shap_summary= st.checkbox('Shap Summary')
        if shap_summary:
            depth= st.selectbox("Depth", [i+1 for i in range(len(selected_cols))])
            summary_type= st.selectbox("Summary Type", ["Aggregate", "Detailed"])
            f= shap_lime(cfg, X_train, X_test, y_train, y_test, plot_shap_summary= {"summary_type": summary_type, "top_k": depth})

            if isinstance(f, list):
                for _p in f:
                    c= 0
                    if isinstance(_p, dict):
                        for _, val in _p.items():
                            st.plotly_chart(_p[f'result_{c}'])
                            c+=1
                    else:
                        st.plotly_chart(_p)
            else:
                st.plotly_chart(f)
        
        shap_dependency= st.checkbox('Shap Dependence')
        if shap_dependency:
            shap_feature_1= st.selectbox('Select The Feature', selected_cols)
            shap_feature_color= st.selectbox('Color Feature', selected_cols)
            shap_feature_2= st.selectbox('Select The Second Feature', selected_cols)
            list_shap_2= st.selectbox(shap_feature_2, [i for i in shap_df[shap_feature_2].unique()])



        # 2. Summary Plot
        summary_plot= st.checkbox('Summary Plot')
        if summary_plot:
            pass

        # 3. Individual Prediction
        feature_contribution= st.checkbox("Feature Contribution")
        if feature_contribution:
            target_cls= st.selectbox("Select the target class", list(y_train.unique()))
            target_cls= get_corresponding_labels(target_cls, True)
            if st.button("Generate Random Number"):
                random_number = random.randint(1, len(X_test))
                _dataframe= DF.iloc[[random_number]]
                st.write(_dataframe)
                proba_preds= inference(_dataframe, True)
                st.write("Model Probability Prediciton")
                st.write(proba_preds)
                fig = go.Figure(data=[go.Pie(values=proba_preds[0])])
                st.plotly_chart(fig)
                figs= shap_lime(cfg, X_train, X_test, y_train, y_test, contribution_plot= {'idx': random_number})
                for fig in figs:
                    st.plotly_chart(fig)

            target_feature= st.selectbox('Select The Feature to See its Contribution', selected_cols)
            list_attr= st.selectbox(target_feature, [i for i in shap_df[target_feature].unique()])
            pdp_feature= st.selectbox('Select the Partial Dependence Feature', selected_cols)
            sorting= st.selectbox("Sorting", ["Absolute", "High to Low", "Low to High", "Importance"])
            summary_type= st.selectbox("The Depth", [i+1 for i in range(len(selected_cols))])
            # TODO
            
            # contribution plot
            # partial dependence plot
            # contribution table

        st.subheader('What If / Inference')
        # st.header('Inference')
        for column in selected_cols:
            user_input = st.text_input(column)
            user_input= utils.inf_proc(user_input)
            inf_df[column] = [user_input]
  
        # p= shap_lime(cfg, X_train, X_test, y_train, y_test, shap_feature= shap_feature)
        # if isinstance(p, list):
        #     for _p in p:
        #         st.pyplot(_p)
        # else:
        #     st.pyplot(p)

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




