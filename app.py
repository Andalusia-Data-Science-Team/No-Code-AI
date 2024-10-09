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

st.info("Click Browse Files")
uploaded_file = st.file_uploader("Upload Data/Model")
SEED = int(st.number_input('Enter a Seed', value=42))
st.write(f'Using {SEED} as a seed')
np.random.seed(SEED)


if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension.lower() == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension.lower() in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
    elif file_extension.lower() == 'pkl':
        try:
            _model = pickle.load(uploaded_file)
            st.success("Pickle file loaded successfully!")
        except pickle.UnpicklingError as e:
            st.error("Error: Invalid pickle file. Please upload a valid pickle file.")
    else:
        st.error('Please upload a CSV, Excel or Pickle file.')

    st.subheader('DataFrame')
    st.write(df)
    num_des_analysis, cat_des_analysis, d_types, missing_percentage, dups_percentage, u_cols= utils.descriptive_analysis(df)
    st.subheader('Numerical Description')
    st.write(num_des_analysis)
    st.subheader('Categorical Description')
    st.write(cat_des_analysis)
    st.subheader('DataFrame Types')
    st.write(d_types)
    st.subheader('Missing per Column %')
    st.write(missing_percentage)
    st.subheader(f'Duplicates {round(dups_percentage * 100, 2)} %')
    # st.write(dups_percentage)
    st.subheader('Unique %')
    st.write(u_cols)
    

    if 'df' in locals():
        cfg= {'save': True} # for inference stability it's fixed
        back_DF= df.copy()
        cols= back_DF.columns
        target = st.selectbox('Select The Target', cols)
        selected_options = st.multiselect('Select columns to be removed', cols)
        DF= back_DF.drop(selected_options, axis= 1)

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
            _DF= utils.missing(_df, cfg['clean'])
            _DF= utils.remove_outliers(_DF, cfg['outlier'])
            if all:
                return _DF

            X_train, X_test, y_train, y_test= utils.handle(_DF, target, task_type)
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
            st.subheader('Outlier-Inlier Percentage')
            outlier_plt_df= utils.missing(back_DF, cfg['clean'])
            outlier_plt= utils.outlier_inlier_plot(outlier_plt_df)
            st.pyplot(outlier_plt)
            heatMap_DF= process_data(back_DF, all= True)
            st.subheader('Heat Map')
            fig, ax = plt.subplots()
            plot_data= utils.HeatMap(heatMap_DF)
            mask = np.zeros_like(plot_data)
            mask[np.triu_indices_from(mask, k=1)] = True
            sns.heatmap(plot_data, mask = mask, annot=True, center=0, fmt='.2f', linewidths=2)
            st.pyplot(fig)

            skew_plots= utils.plot_numeric_features(back_DF)
            for skew_plot in skew_plots:
                st.pyplot(skew_plot)
            
            pca= utils.PCA_visualization(back_DF)
            st.pyplot(pca)


            # Display a scatter plot
            st.subheader('Correlation')
            corr_values= utils.corr_plot(back_DF)
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
            shap_summary_normalized= st.checkbox('Do Not Normalize Shap Summary')
            f= shap_lime(cfg, X_train, X_test, y_train, y_test, plot_shap_summary= {"summary_type": summary_type, 
                                                                                    "top_k": depth,
                                                                                    "normalize": not shap_summary_normalized})

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


        # 2. Individual Prediction
        feature_contribution= st.checkbox("Feature Contribution")
        classes= None
        if feature_contribution:
            target_cls= st.selectbox("Select the target class", list(y_train.unique()))
            if type(DF[target][0]) == str:
                classes, target_cls= get_corresponding_labels(target_cls, True)
                st.write(classes)
            if st.button("Generate Random Index"):
                X_test= X_test.reset_index()
                _sub_DF= DF.reset_index()
                random_number = random.randint(1, len(X_test))
                _dataframe= X_test.iloc[[random_number]]
                st.write(_dataframe)
                if task_type == "Classification": 
                    proba_preds= inference(_dataframe, True)
                    st.write("Model Probability Prediciton")
                    st.write(proba_preds)
                    if classes:
                        fig = go.Figure(data=[go.Pie(values=proba_preds[0], labels=list(classes.keys()))])
                    else:
                        fig = go.Figure(data=[go.Pie(values=proba_preds[0])])
                    st.plotly_chart(fig)
                figs= shap_lime(cfg, X_train, X_test, y_train, y_test, plot_contribution= {'idx': random_number, 
                                                                                           'agg': False,
                                                                                           'max_display': 8})
                for fig in figs[0]:
                    st.pyplot(fig)

                # table

        st.subheader('What If / Inference')
        for column in selected_cols:
            if d_types.loc[column].values == 'object':
                user_input= st.selectbox(f"{column}", [i for i in DF[column].unique()])
            else:
                user_input = st.text_input(column)
                user_input= utils.inf_proc(user_input)
            inf_df[column] = [user_input]
  

        st.write(inf_df)
        if st.button('Submit'):
            if task_type == "Classification":
                preds= inference(inf_df, True)
                if classes:
                    fig = go.Figure(data=[go.Pie(values=preds[0], labels=list(classes.keys()))])
                else:
                    fig = go.Figure(data=[go.Pie(values=preds[0])])
                st.plotly_chart(fig)
            figs= shap_lime(cfg, X_train, X_test, y_train, y_test, plot_contribution= {'data': inf_df, 
                                                                                       'agg': False,
                                                                                       'max_display': 8})
            for fig in figs[0]:
                st.pyplot(fig)
            else:
                # regression
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




