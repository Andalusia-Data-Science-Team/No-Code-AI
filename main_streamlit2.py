import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
import src.insight.utils as utils
from src.insight.models import model, inference, get_corresponding_labels
import plotly.graph_objects as go
import altair as alt

# Set page configuration
st.set_page_config(page_title="Business AI Tool", layout="wide")

# Page Title
st.title("📈 AI-Powered Insights: Zero-Code Data Analysis & Modeling")

# File Upload Section
st.markdown("### Step 1: Upload Training  Data")
uploaded_file = st.file_uploader("Upload a CSV, Excel, or Pickle file", type=["csv", "xls", "xlsx", "pkl"])

# Seed Input
# SEED = st.number_input("Enter a random seed value", value=42, help="Set a seed for reproducibility.")
SEED = 42
np.random.seed(SEED)

if uploaded_file:
    # Handle file upload
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
    elif file_extension == "pkl":
        try:
            _model = pickle.load(uploaded_file)
            st.success("✅ Model file loaded successfully!")
        except pickle.UnpicklingError:
            st.error("❌ Invalid Pickle file. Please upload a valid file.")
            st.stop()
    else:
        st.error("❌ Unsupported file type. Please upload a CSV, Excel, or Pickle file.")
        st.stop()

    # Data Preview
    st.markdown("### Step 2: Data Overview")
    st.write("Here is a preview of your uploaded data:")
    st.dataframe(df.head())

    # Perform descriptive analysis
    num_desc, cat_desc, d_types, missing, dups, unique = utils.descriptive_analysis(df)

    # Perform descriptive analysis
    num_desc, cat_desc, d_types, missing, dups, unique = utils.descriptive_analysis(df)

    # Overview Section
    with st.expander("Overview", expanded=True):
        st.write("""
        Explore your dataset's key characteristics, including numerical summaries, categorical insights, data types, missing values, and duplicates.
        Use the sections below for detailed breakdowns and visualizations.
        """)
        st.write(f"**Number of Rows:** {df.shape[0]} | **Number of Columns:** {df.shape[1]}")

    # Numerical Description
    with st.expander("Numerical Description"):
        st.write("""
        A statistical summary of numeric columns, including metrics like mean, standard deviation, minimum, and maximum values. 
        These metrics help identify trends, anomalies, and overall data distribution.
        """)
        st.dataframe(num_desc.style.format(precision=2).background_gradient(cmap='coolwarm'), use_container_width=True)

    # Categorical Description
    with st.expander("Categorical Description"):
        st.write("""
        A summary of categorical columns showing the count and unique values for each category. 
        This helps you understand the variety and frequency of categories in your dataset.
        """)
        st.dataframe(cat_desc.style.format(precision=2).background_gradient(cmap='coolwarm'), use_container_width=True)

    # Data Types
    # with st.expander("DataFrame Types"):
    #     st.write("""
    #     A list of column data types (e.g., integer, float, object). Ensuring the correct data types is crucial for accurate analysis and modeling.
    #     """)
    #     st.dataframe(d_types, use_container_width=True)

    # Missing Data
    # Missing Data
    with st.expander("Missing Values (% per Column)"):
        st.write("""
        This chart highlights the percentage of missing values in each column. Missing data should be addressed to maintain analysis and modeling accuracy.
        """)

        missing = missing.reset_index()  # Convert Series to DataFrame
        col_index = missing.columns[0]
        missing.columns = [col_index, 'missing %']

        # Ensure 'Missing_Percentage' is numeric
        missing['missing %'] = pd.to_numeric(missing['missing %'], errors='coerce')

        # Drop rows with invalid numeric data
        missing = missing.dropna(subset=['missing %'])

        # Plot using Altair
        chart = alt.Chart(missing).mark_bar().encode(
            x=alt.X(col_index, sort=None),
            y=alt.Y('missing %', title='Missing Percentage (%)'),
            tooltip=[col_index, 'missing %']
        ).properties(
            width=800,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

    # Duplicate Rows
    with st.expander("Duplicate Rows"):
        st.write(f"""
        **Duplicate Rows:** {round(dups * 100, 2)}%

        Duplicate rows represent repeated records, which may lead to biased analysis. It's recommended to remove them if unnecessary.
        """)

    # Unique Values
    with st.expander("Unique Values (% per Column)"):
        if isinstance(unique, pd.Series):
            unique = unique.reset_index()  # Convert Series to DataFrame
            unique.columns = ['Column', 'Unique_Percentage']

        # Ensure 'Unique_Percentage' is numeric
        unique['Unique_Percentage'] = pd.to_numeric(unique['Unique_Percentage'], errors='coerce')

        # Drop rows with invalid numeric data
        unique = unique.dropna(subset=['Unique_Percentage'])

        chart = alt.Chart(unique).mark_bar().encode(
            x=alt.X('Column', sort=None),
            y=alt.Y('Unique_Percentage', title='Unique_Percentage (%)'),
            tooltip=['Column', 'Unique_Percentage']
        ).properties(
            width=800,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    📖 **Tip**: Use this summary to clean, preprocess, and understand your data better before training models. 
    If you encounter high missing values or duplicates, consider cleaning the data for optimal results.
    """)

    # Task Configuration
    st.markdown("### Step 3: Configure Your Task")
    # cfg = {"save": True}
    # target = st.selectbox("🎯 Select Target Variable", df.columns, help="Choose the column you want to predict.")
    # exclude_columns = st.multiselect("❌ Select Columns to Exclude", df.columns)
    # df_filtered = df.drop(columns=exclude_columns)
    cfg = {'save': True}  # for inference stability it's fixed
    back_DF = df.copy()
    cols = back_DF.columns
    target = st.selectbox("🎯 Select Target Variable", cols, help="Choose the column you want to predict.")
    selected_options = st.multiselect("❌ Select Columns to Exclude", cols, help= "Excluded columns will not be used by the model")
    DF = back_DF.drop(selected_options, axis=1)

    validation_size = st.slider("📊 Select Validation Size (%) : recommended value: 20% ", min_value=1, max_value=100, value=20)
    task_type = st.radio("⚙️ Select Task Type", ["Classification", "Regression", "Time"], index=0, help="Select the task type")

    model_mapping = {
        # Classification Models
        "Support Vector Classifier (SVC)": {
            "code": "SVC",
            "recommendation": "Best for small to medium-sized datasets with well-separated classes. Often used in text classification or when kernel tricks are needed for non-linear decision boundaries."
        },
        "Logistic Regression Classifier (LR)": {
            "code": "LR",
            "recommendation": "Ideal for binary classification tasks. Simple, interpretable, and effective for linearly separable data."
        },
        "Random Forest Classifier (RF_cls)": {
            "code": "RF_cls",
            "recommendation": "Great for handling high-dimensional datasets, noisy data, or when feature importance is required. Often used in tabular data and ensemble learning."
        },
        "Extreme Gradient Boosting Classifier (XGB_cls)": {
            "code": "XGB_cls",
            "recommendation": "Recommended for structured/tabular data with complex relationships. Excels in competitions due to its speed and accuracy."
        },

        # Regression Models
        "Linear Regression": {
            "code": "Linear Regression",
            "recommendation": "A simple yet effective model for predicting continuous outcomes in linearly correlated data."
        },
        "Ridge Regression": {
            "code": "Ridge",
            "recommendation": "An extension of Linear Regression that is great for handling multicollinearity in the data."
        },
        "Lasso Regression": {
            "code": "Lasso",
            "recommendation": "Similar to Ridge Regression but performs feature selection by shrinking less important feature coefficients to zero."
        },
        "Support Vector Regressor (SVR)": {
            "code": "SVR",
            "recommendation": "Effective for small to medium-sized datasets, particularly with non-linear relationships."
        },
        "Extreme Gradient Boosting Regressor (XGB_reg)": {
            "code": "XGB_reg",
            "recommendation": "A powerful model for structured/tabular data, particularly when interactions among variables are complex."
        },

    }

    # Provide explanations and visuals dynamically based on selection
    if task_type == "Classification":
        st.markdown("#### Classification: 🔍")
        st.write("""
        **Use Case**: Predict a category or label (e.g., spam vs. non-spam, yes vs. no).

        **Examples**:
        - Will a customer churn? (Yes/No)
        - What type of product is being purchased? (Category A, B, or C)

        **Models**: SVC, Logistic Regression, Random Forest, XGBoost.
        """)
        st.image("static/imgs/classification.png",
                 caption="Example: Classification", use_container_width =True)

    elif task_type == "Regression":
        st.markdown("#### Regression: 📈")
        st.write("""
        **Use Case**: Predict a continuous value (e.g., sales, stock prices, temperatures).

        **Examples**:
        - What will next month's revenue be?
        - How much will a house sell for?

        **Models**: Linear Regression, Ridge, Lasso, SVR, XGBoost.
        """)
        st.image("static/imgs/regression.png",
                 caption="Example: Linear Regression", use_container_width =True)

    elif task_type == "Time":
        st.markdown("#### Time Series: ⏳")
        st.write("""
        **Use Case**: Analyze patterns over time and predict future values (e.g., sales trends, stock prices, weather forecasting).

        **Examples**:
        - What will the sales be next week?
        - How will electricity demand change throughout the day?

        **Models**: Prophet, LSTM.
        """)
        st.image("static/imgs/timeseries.png",
                 caption="Example: Time Series Forecasting", use_container_width =True)

    # Task-Specific Configuration
    if task_type == "Classification":
        available_models = {k: v for k, v in model_mapping.items() if "Classifier" in k}
        st.markdown("#### Classification Options")
        cfg["task_type"] = task_type
        cfg["clean"] = st.selectbox("🧹 Data Cleaning", ["Remove Missing Data", "Impute Missing Data"])
        cfg["outlier"] = st.selectbox("📉 Handle Outliers", ["Don't Remove", "Use IQR", "Isolation Forest"])

        # User-friendly model selection
        selected_model_label = st.selectbox(
            "🤖 Select Model",
            list(available_models.keys()),
            help="Choose the machine learning algorithm to train your model."
        )

        # Retrieve backend code and recommendation for the selected model
        selected_model = available_models[selected_model_label]
        cfg["alg"] = selected_model["code"]

        cfg["skew_fix"] = st.checkbox("🔄 Fix Skewed Data")
        cfg["poly_feat"] = st.checkbox("🔢 Add Polynomial Features")
        cfg["apply_GridSearch"] = st.checkbox("🔍 Optimize Hyperparameters")

    elif task_type == "Regression":
        st.markdown("#### Regression Options")
        cfg["task_type"] = task_type
        available_models = {k: v for k, v in model_mapping.items() if ("Regression" in k or "Regressor" in k) and 'Logistic' not in k}

        cfg["clean"] = st.selectbox("🧹 Data Cleaning", ["Remove Missing Data", "Impute Missing Data"])
        cfg["outlier"] = st.selectbox("📉 Handle Outliers", ["Don't Remove", "Use IQR", "Isolation Forest"])
        # User-friendly model selection
        selected_model_label = st.selectbox(
            "🤖 Select Model",
            list(available_models.keys()),
            help="Choose the machine learning algorithm to train your model."
        )

        # Retrieve backend code and recommendation for the selected model
        selected_model = available_models[selected_model_label]
        cfg["alg"] = selected_model["code"]
        cfg["skew_fix"] = st.checkbox("🔄 Fix Skewed Data")
        cfg["poly_feat"] = st.checkbox("🔢 Add Polynomial Features")
        cfg["apply_GridSearch"] = st.checkbox("🔍 Optimize Hyperparameters")

    elif task_type == "Time":
        st.markdown("#### Time Series Options")
        cfg["task_type"] = task_type
        ts_kw = {}
        cfg["clean"] = st.selectbox("🧹 Data Cleaning", ["Remove Missing Data", "Impute Missing Data"])
        cfg["outlier"] = st.selectbox("📉 Handle Outliers", ["Don't Remove", "Use IQR", "Isolation Forest"])
        cfg["alg"] = st.selectbox("📈 Select Model", ["Prophet", "LSTM"])

        if cfg['alg'] == 'Prophet':
            ts_kw['date_col'] = st.selectbox('Select The Date Column', DF.columns)
            ts_kw['target_col'] = target
            ts_kw['prophet_params'] = {}
            ts_kw['selected_cols'] = {}
            ts_kw['freq'] = '1min'
            ts_kw['f_period'] = 5
            cfg['ts_config'] = ts_kw

        cfg['skew_fix'] = st.checkbox('Skew Fix')
        cfg['poly_feat'] = False
        cfg['apply_GridSearch'] = False

        print(cfg['task_type'])

    # Execute Task
    # Execute Task
    if st.button("🚀 Train Model"):
        if task_type == "Classification":
            st.write("Perform classification task with option:")
            X_train, X_test, y_train, y_test = utils.process_data(DF, cfg, target, task_type)
            report = model(X_train, X_test, y_train, y_test, cfg)
            st.write("Accuracy:")
            st.write(report[0])
            st.write("Confusion Matrix")
            st.write(report[1])

        elif task_type == "Regression":
            st.write("Perform Regression task with option:")
            X_train, X_test, y_train, y_test = utils.process_data(DF, cfg, target, task_type)
            report = model(X_train, X_test, y_train, y_test, cfg)
            st.write("MSE:")
            st.write(report[0])
            st.write("R2:")
            st.write(report[1])

        elif task_type == "Time":
            st.write("Performing Time Series Analysis")
            ts_df = utils.process_data(DF, cfg, target, task_type, all=True)
            pf = model(ts_df, cfg=cfg)
            #st.plotly_chart(pf.slide_display())
            #st.pyplot(pf.plot_forcast())
            #st.pyplot(pf.plot_component())
            # Constrain Plotly Slide Display
            plotly_fig = pf.slide_display()
            # plotly_fig.update_layout(
            #     width=600,  # Adjust width
            #     height=300  # Adjust height
            # )
            st.plotly_chart(plotly_fig, use_container_width=False)

            # Constrain Matplotlib Forecast Plot
            forecast_fig = pf.plot_forcast()
            #forecast_fig.set_size_inches(10, 3)  # Adjust size
            st.pyplot(forecast_fig)

            # Constrain Matplotlib Component Plot
            component_fig = pf.plot_component()
            #component_fig.set_size_inches(10, 3)  # Adjust size
            st.pyplot(component_fig)
            
    if task_type != "Time":
    
        # Inference Section
        st.markdown("### 🔍 What If / Inference")
        st.write("""Provide input values to test your trained model interactively.""")

        # Create an empty DataFrame for user inputs
        inf_df = pd.DataFrame(columns=DF.columns)

        # Dynamic User Input Form
        st.markdown("#### Provide Input Values for Prediction")
        input_cols = [column for column in DF.columns if column != target ]
        for column in input_cols :
            if d_types.loc[column].values[0] == 'object':  # Categorical columns
                user_input = st.selectbox(
                    f"Select a value for **{column}**:",
                    options=DF[column].unique(),
                    help=f"Choose a category for the column '{column}'."
                )
            else:  # Numeric columns
                user_input = st.text_input(
                    f"Enter a value for **{column}**:",
                    value=str(np.mean(DF[column])),
                    help=f"Provide a numeric value for the column '{column}'."
                )
                # Preprocess numeric input
                user_input = utils.inf_proc(user_input)
            inf_df[column] = [user_input]

        # Display the prepared input DataFrame
        st.markdown("### 🛠 Prepared Inference Data")
        st.dataframe(inf_df[input_cols], use_container_width=True)

        # Define classes (specific to classification tasks)
        if task_type == "Classification":
            st.markdown("#### Define Class Labels (Optional)")
            try:
                # Attempt to define classes using target column
                classes = {label: idx for idx, label in enumerate(sorted(DF[target].unique()))}
                st.write(f"Classes defined from target column: {classes}")
            except KeyError:
                classes = None
                st.warning("Unable to define classes automatically. Check your data or model.")

        # Run Inference Button
        if st.button("🚀 Run Inference"):
            try:
                if task_type == "Classification":
                    preds = inference(inf_df, True)
                    st.markdown("#### 📊 Model Prediction Results")
                    if classes:
                        fig = go.Figure(data=[
                            go.Pie(values=preds[0], labels=list(classes.keys()), hole=0.4)
                        ])
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("Class Probabilities:")
                        st.write({k: v for k, v in zip(classes.keys(), preds[0])})
                    else:
                        st.error("Class labels are missing. Unable to display probabilities.")

                elif task_type == "Regression":
                    preds = max(inference(inf_df),1)

                    st.markdown("#### 📊 Predicted Output")
                    st.write(f"**Prediction:** for {target} {preds}")
                    st.markdown("""
                       **📝 Note**: Regression models predict continuous numeric values, 
                       which can range over an interval rather than being limited to discrete categories.
                       """)

                elif task_type == "Time":
                    st.markdown("#### Time Series Prediction")
                    ts_results = model(inf_df, cfg)
                    st.write("Forecast Results:")
                    st.write(ts_results)

            except Exception as e:
                st.error(f"An error occurred during inference: {e}")


    # Upload testing data
    # Add functionality to upload test data
    st.subheader("🔍 Upload Test Data and Predict")
    st.write("""
    Upload your test dataset to generate predictions using the selected model. Ensure the test dataset has the same structure (columns) as the training data.
    """)

    # File uploader for test data
    test_data_file = st.file_uploader("Upload Test Data (CSV, Excel, or Pickle)", type=["csv", "xls", "xlsx", "pkl"],
                                      key="test_data_uploader")

    if test_data_file:
        # Handle file upload
        file_extension = test_data_file.name.split(".")[-1]
        try:
            if file_extension == "csv":
                test_df = pd.read_csv(test_data_file)
            elif file_extension in ["xls", "xlsx"]:
                test_df = pd.read_excel(test_data_file)
            elif file_extension == "pkl":
                test_df = pickle.load(test_data_file)
            else:
                st.error("❌ Unsupported file type. Please upload a CSV, Excel, or Pickle file.")
                st.stop()

            # Validate test data structure
            if set(test_df.columns) != set(DF.columns) - {target}:
                st.error("❌ The columns in the test data must match the training data (excluding the target column).")
            else:
                st.success("✅ Test data uploaded successfully!")
                st.write("Here is a preview of your test data:")
                st.dataframe(test_df.head())

                # Run predictions
                if st.button("🚀 Run Predictions"):
                    X_test = test_df.copy()  # Ensure the test data does not include the target column
                    predictions = inference(X_test)  # Replace this with your prediction function

                    test_df['Predictions'] = predictions  # Append predictions to the test data
                    test_df['Predictions'] = test_df['Predictions'].apply(lambda x: max(x, 1))
                    st.success("✅ Predictions generated successfully!")
                    st.write("Here is the test data with predictions:")
                    st.dataframe(test_df)

                    # Download predictions
                    st.markdown("### 📥 Download Predictions")
                    csv = test_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")