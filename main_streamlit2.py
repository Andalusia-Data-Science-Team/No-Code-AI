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
st.title("📈 Business AI Tool: Data Analysis and Modeling")

# File Upload Section
st.markdown("### Step 1: Upload Your Data")
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
    with st.expander("DataFrame Types"):
        st.write("""
        A list of column data types (e.g., integer, float, object). Ensuring the correct data types is crucial for accurate analysis and modeling.
        """)
        st.dataframe(d_types, use_container_width=True)

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
    selected_options = st.multiselect("❌ Select Columns to Exclude", cols)
    DF = back_DF.drop(selected_options, axis=1)

    validation_size = st.slider("📊 Select Validation Size (%)", min_value=1, max_value=100, value=20)
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
            "code": "LinearRegression",
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
    if st.button("🚀 Run Task"):
        if task_type == "Classification":
            st.write('Perform classification task with option:')
            X_train, X_test, y_train, y_test = utils.process_data(DF, cfg, target, task_type)
            report = model(X_train, X_test, y_train, y_test, cfg)
            st.write("Accuracy:")
            st.write(report[0])
            st.write("Confusion Matrix")
            st.write(report[1])

        if task_type == "Regression":
            st.write('Perform Regression task with option:')
            X_train, X_test, y_train, y_test = utils.process_data(DF, cfg, target, task_type)
            report = model(X_train, X_test, y_train, y_test, cfg)
            st.write("MSE:")
            st.write(report[0])
            st.write("R2:")
            st.write(report[1])

        if task_type == "Time":
            st.write("Performing Time Series Analysis")
            ts_df = utils.process_data(DF, cfg, target, task_type, all=True)
            pf = model(ts_df, cfg=cfg)
            st.plotly_chart(pf.slide_display())
            st.pyplot(pf.plot_forcast())
            st.pyplot(pf.plot_component())
