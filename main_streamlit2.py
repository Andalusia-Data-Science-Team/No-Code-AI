import streamlit as st
import pandas as pd
import numpy as np
import pickle
import src.insight.utils as utils
from src.insight.models import model, inference
import plotly.graph_objects as go
import altair as alt
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx


def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception:
        return None

    return session_info.request.remote_ip


def download_preds(df):
    st.markdown("### 📥 Download Predictions")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )


# Page Title
st.title("📈 AI-Powered Insights: Zero-Code Data Analysis & Modeling")

# File Upload Section
st.markdown("### Step 1: Upload Training Data")
uploaded_file = st.file_uploader(
    "Upload a CSV, Excel, or Parquet file",
    type=["csv", "xls", "xlsx", "parquet"],
)

# Seed Input
SEED = 42
np.random.seed(SEED)

if uploaded_file:
    # Handle file upload
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
    elif file_extension == "parquet":
        df = pd.read_parquet(uploaded_file)
    else:
        st.error(
            "❌ Unsupported file type. Please upload a CSV, Excel, or Parquet file."
        )
        st.stop()

    # Data Preview
    st.markdown("### Step 2: Data Overview")
    st.write("Here is a preview of your uploaded data:")
    st.dataframe(df.head())

    # Perform descriptive analysis
    num_desc, cat_desc, d_types, missing, dups, unique = utils.descriptive_analysis(df)

    # st.write("Data Types")
    # st.dataframe(d_types)

    # Overview Section
    with st.expander("Overview", expanded=True):
        st.write(
            """
        Explore your dataset's key characteristics, including numerical summaries, categorical insights, data types, missing values, and duplicates.
        Use the sections below for detailed breakdowns and visualizations.
        """
        )
        st.write(
            f"**Number of Rows:** {df.shape[0]} | **Number of Columns:** {df.shape[1]}"
        )

    # Numerical Description
    with st.expander("Numerical Description"):
        st.write(
            """
        A statistical summary of numeric columns, including metrics like mean, standard deviation, minimum, and maximum values.
        These metrics help identify trends, anomalies, and overall data distribution.
        """
        )
        if num_desc is not None:
            st.dataframe(
                num_desc.style.format(precision=2).background_gradient(cmap="coolwarm"),
                use_container_width=True,
            )
        else:
            st.write("The uploaded file doesn't contain numerical data")

    # Categorical Description
    with st.expander("Categorical Description"):
        st.write(
            """
        A summary of categorical columns showing the count and unique values for each category.
        This helps you understand the variety and frequency of categories in your dataset.
        """
        )
        if cat_desc is not None:
            st.dataframe(
                cat_desc.style.format(precision=2).background_gradient(cmap="coolwarm"),
                use_container_width=True,
            )
        else:
            st.write("The uploaded file doesn't contain categorical data")
    # Data Types
    # with st.expander("DataFrame Types"):
    #     st.write("""
    #     A list of column data types (e.g., integer, float, object). Ensuring the correct data types is crucial for accurate analysis and modeling.
    #     """)
    #     st.dataframe(d_types, use_container_width=True)

    # Missing Data
    with st.expander("Missing Values (% per Column)"):
        st.write(
            """
        This chart highlights the percentage of missing values in each column. Missing data should be addressed to maintain analysis and modeling accuracy.
        """
        )

        missing = missing.reset_index()  # Convert Series to DataFrame
        col_index = missing.columns[0]
        missing.columns = [col_index, "missing %"]

        # Ensure 'Missing_Percentage' is numeric
        missing["missing %"] = pd.to_numeric(missing["missing %"], errors="coerce")

        # Drop rows with invalid numeric data
        missing = missing.dropna(subset=["missing %"])

        # Plot using Altair
        chart = (
            alt.Chart(missing)
            .mark_bar()
            .encode(
                x=alt.X(col_index, sort=None),
                y=alt.Y("missing %", title="Missing Percentage (%)"),
                tooltip=[col_index, "missing %"],
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    # Duplicate Rows
    with st.expander("Duplicate Rows"):
        st.write(
            f"""
        **Duplicate Rows:** {round(dups * 100, 2)}%

        Duplicate rows represent repeated records, which may lead to biased analysis. It's recommended to remove them if unnecessary.
        """
        )

    # Unique Values
    with st.expander("Unique Values (% per Column)"):
        if isinstance(unique, pd.Series):
            unique = unique.reset_index()  # Convert Series to DataFrame
            unique.columns = ["Column", "Unique_Percentage"]

        # Ensure 'Unique_Percentage' is numeric
        unique["Unique_Percentage"] = pd.to_numeric(
            unique["Unique_Percentage"], errors="coerce"
        )

        # Drop rows with invalid numeric data
        unique = unique.dropna(subset=["Unique_Percentage"])

        chart = (
            alt.Chart(unique)
            .mark_bar()
            .encode(
                x=alt.X("Column", sort=None),
                y=alt.Y("Unique_Percentage", title="Unique_Percentage (%)"),
                tooltip=["Column", "Unique_Percentage"],
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    📖 **Tip**: Use this summary to clean, preprocess, and understand your data better before training models.
    If you encounter high missing values or duplicates, consider cleaning the data for optimal results.
    """
    )

    # Data Preprocessing
    st.markdown("### Step 3: Data Preprocessing")
    if "df" in locals():
        cfg = {"save": True}  # for inference stability it's fixed
        cfg["model_kw"] = dict()

        with st.expander(" :broom: Handle Missing Data"):
            cfg["clean"] = st.selectbox(
                "Choose how to handle missing data:",
                ["Remove Missing Data", "Impute Missing Data"],
                help="Select whether to remove or impute missing values.",
            )
        with st.expander("📉 Handle Outliers"):
            # Create two columns for horizontal alignment
            col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
            with col1:
                st.write(
                    """
                Removes or adjusts extreme data points that can skew results. Choose from:
                - **Don't Remove**: Keep all data points, including outliers.
                - **Use IQR**: Remove data points that are too far from the usual range of most data. For example, it’s like
                  ignoring the very low and very high prices in a list of product sales to focus on the middle range.
                - **Isolation Forest**: Use machine learning algorithms to handle outliers.
                """
                )

            with col2:
                st.image("static/imgs/outliers.png", use_container_width=True)
            cfg["outlier"] = st.selectbox(
                "How would you like to handle outliers?",
                ["Don't Remove", "Use IQR", "Isolation Forest"],
                help="Select an outlier handling technique.",
            )
        with st.expander("🔄 Fix Skewed Data"):
            col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
            with col1:
                st.write(
                    """
                Transforms skewed distributions to be more symmetrical, enhancing model accuracy.
                Particularly useful for data with highly asymmetric values.
                """
                )
            with col2:
                st.image("static/imgs/skewness.png", use_container_width=True)
            cfg["skew_fix"] = st.checkbox(
                "Fix Skewed Data?", help="Enable or disable skewed data transformation."
            )

        with st.expander("🔢 Add Polynomial Features"):
            col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
            with col1:
                st.write(
                    """
            Creates new features by raising existing features to a power, helping the model capture non-linear relationships.
            Useful for regression tasks with complex data relationships.
            """
                )
            with col2:
                st.image("static/imgs/polynomials.png", use_container_width=True)
            cfg["poly_feat"] = st.checkbox(
                "Add Polynomial Features?",
                help="Enable or disable the addition of polynomial features.",
            )

        # Task Configuration
        st.markdown("### Step 4: Configure Your Task")
        # cfg = {"save": True}
        # target = st.selectbox("🎯 Select Target Variable", df.columns, help="Choose the column you want to predict.")
        # exclude_columns = st.multiselect("❌ Select Columns to Exclude", df.columns)
        # df_filtered = df.drop(columns=exclude_columns)

        task_type = st.radio(
            "⚙️ Select Task Type",
            ["Classification", "Regression", "Cluster", "Time"],
            index=0,
            help="Select the task type",
        )

        model_mapping = {
            # Classification Models
            "Support Vector Classifier (SVC)": {
                "code": "SVC",
                "recommendation": """Best for small to medium-sized datasets with well-separated classes.
                  Often used in text classification or when kernel tricks are needed for non-linear decision boundaries.""",
            },
            "Logistic Regression Classifier (LR)": {
                "code": "LR",
                "recommendation": "Ideal for binary classification tasks. Simple, interpretable, and effective for linearly separable data.",
            },
            "Random Forest Classifier (RF_cls)": {
                "code": "RF_cls",
                "recommendation": """Great for handling high-dimensional datasets, noisy data, or when feature importance is required.
                  Often used in tabular data and ensemble learning.""",
            },
            "Extreme Gradient Boosting Classifier (XGB_cls)": {
                "code": "XGB_cls",
                "recommendation": "Recommended for structured/tabular data with complex relationships. Excels in competitions due to its speed and accuracy.",
            },
            # Regression Models
            "Linear Regression": {
                "code": "Linear Regression",
                "recommendation": "A simple yet effective model for predicting continuous outcomes in linearly correlated data.",
            },
            "Ridge Regression": {
                "code": "Ridge",
                "recommendation": "An extension of Linear Regression that is great for handling multicollinearity in the data.",
            },
            "Lasso Regression": {
                "code": "Lasso",
                "recommendation": "Similar to Ridge Regression but performs feature selection by shrinking less important feature coefficients to zero.",
            },
            "Support Vector Regressor (SVR)": {
                "code": "SVR",
                "recommendation": "Effective for small to medium-sized datasets, particularly with non-linear relationships.",
            },
            "Extreme Gradient Boosting Regressor (XGB_reg)": {
                "code": "XGB_reg",
                "recommendation": "A powerful model for structured/tabular data, particularly when interactions among variables are complex.",
            },
        }

        # Provide explanations and visuals dynamically based on selection
        if task_type == "Classification":
            st.markdown("#### Classification: 🔍")
            col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
            with col1:
                st.write(
                    """
            **Use Case**: Predict a category or label (e.g., spam vs. non-spam, yes vs. no).

            **Examples**:
            - Will a customer churn? (Yes/No)
            - What type of product is being purchased? (Category A, B, or C)

            **Models**: SVC, Logistic Regression, Random Forest, XGBoost.
            """
                )
            with col2:
                st.image(
                    "static/imgs/classification.png",
                    caption="Example: Classification",
                    use_container_width=True,
                )

        elif task_type == "Regression":
            st.markdown("#### Regression: 📈")
            col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
            with col1:
                st.write(
                    """
            **Use Case**: Predict a continuous value (e.g., sales, stock prices, temperatures).

            **Examples**:
            - What will next month's revenue be?
            - How much will a house sell for?

            **Models**: Linear Regression, Ridge, Lasso, SVR, XGBoost.
            """
                )
            with col2:
                st.image(
                    "static/imgs/regression.png",
                    caption="Example: Linear Regression",
                    use_container_width=False,
                )

        elif task_type == "Cluster":
            st.markdown("#### Clustering: 🌀")
            col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
            with col1:
                st.write(
                    """
                **Use Case**: Identify groups or clusters of similar data points without predefined labels.

                **Examples**:
                - Segmenting customers based on purchase behaviors.
                - Grouping similar products based on features.
                - Detecting anomalies by identifying clusters of unexpected data points.

                **Models**: K-Means, Gaussian Mixture
                """
                )
            with col2:
                st.image(
                    "static/imgs/clustering.png",
                    caption="Example: Clustering Analysis",
                    use_container_width=False,
                )
        elif task_type == "Time":
            st.markdown("#### Time Series: ⏳")
            col1, col2 = st.columns([2, 1])  # Adjust the column width ratio as needed
            with col1:
                st.write(
                    """
            **Use Case**: Analyze patterns over time and predict future values (e.g., sales trends, stock prices, weather forecasting).

            **Examples**:
            - What will the sales be next week?
            - How will electricity demand change throughout the day?

            **Models**: Prophet
            """
                )
            with col2:
                st.image(
                    "static/imgs/timeseries.png",
                    caption="Example: Time Series Forecasting",
                    use_container_width=True,
                )

        back_DF = df.copy()
        cols = back_DF.columns
        # To show only numerical columns when doing regression or time series forecasting
        if task_type == "Time" or task_type == "Regression":
            target_col = back_DF.select_dtypes(include=np.number).columns
        else:
            target_col = cols

        target = (
            st.selectbox(
                "🎯 Select Target Variable",
                target_col,
                help="Choose the column you want to predict.",
            )
            if task_type != "Cluster"
            else None
        )

        if task_type != "Cluster":  # == "Classification" or task_type == "Regression"
            validation_size = st.slider(
                "📊 Select Validation Size (%) : recommended value: 20% ",
                min_value=1,
                max_value=100,
                value=20,
            )
            selected_options = st.multiselect(
                "❌ Select Columns to Exclude",
                cols,
                help="Excluded columns will not be used by the model",
            )
            DF = back_DF.drop(selected_options, axis=1)
        else:
            validation_size = 0
            selected_options = st.multiselect(
                "Select Columns to Include For Clustering",
                cols,
                help="Included columns will be used by the model",
            )
            DF = back_DF[selected_options]

        # Task-Specific Configuration
        if task_type == "Classification":
            available_models = {
                k: v for k, v in model_mapping.items() if "Classifier" in k
            }
            st.markdown("#### Classification Options")
            cfg["task_type"] = task_type
            # cfg["clean"] = st.selectbox("🧹 Data Cleaning", ["Remove Missing Data", "Impute Missing Data"])
            # cfg["outlier"] = st.selectbox("📉 Handle Outliers", ["Don't Remove", "Use IQR", "Isolation Forest"])

            # User-friendly model selection
            selected_model_label = st.selectbox(
                "🤖 Select Model",
                list(available_models.keys()),
                help="Choose the machine learning algorithm to train your model.",
            )

            # Retrieve backend code and recommendation for the selected model
            selected_model = available_models[selected_model_label]
            cfg["alg"] = selected_model["code"]

            # cfg["skew_fix"] = st.checkbox("🔄 Fix Skewed Data")
            # cfg["poly_feat"] = st.checkbox("🔢 Add Polynomial Features")
            cfg["apply_GridSearch"] = st.checkbox(
                "🔍 Optimize Hyperparameters: Hyperparameter optimization fine-tunes a "
                "machine learning model, like adjusting settings on a machine to "
                "achieve peak efficiency and accuracy"
            )
            # cfg["apply_GridSearch"] = False

        elif task_type == "Regression":
            st.markdown("#### Regression Options")
            cfg["task_type"] = task_type
            available_models = {
                k: v
                for k, v in model_mapping.items()
                if ("Regression" in k or "Regressor" in k) and "Logistic" not in k
            }

            # cfg["clean"] = st.selectbox("🧹 Data Cleaning", ["Remove Missing Data", "Impute Missing Data"])
            # cfg["outlier"] = st.selectbox("📉 Handle Outliers", ["Don't Remove", "Use IQR", "Isolation Forest"])
            # User-friendly model selection
            selected_model_label = st.selectbox(
                "🤖 Select Model",
                list(available_models.keys()),
                help="Choose the machine learning algorithm to train your model.",
            )

            # Retrieve backend code and recommendation for the selected model
            selected_model = available_models[selected_model_label]
            cfg["alg"] = selected_model["code"]
            # cfg["skew_fix"] = st.checkbox("🔄 Fix Skewed Data")
            # cfg["poly_feat"] = st.checkbox("🔢 Add Polynomial Features")
            cfg["apply_GridSearch"] = st.checkbox(
                "🔍 Optimize Hyperparameters: Hyperparameter optimization fine-tunes a "
                "machine learning model, like adjusting settings on a machine to "
                "achieve peak efficiency and accuracy"
            )
            # cfg["apply_GridSearch"] = False

        elif task_type == "Cluster":
            cfg["task_type"] = task_type
            cfg["alg"] = st.selectbox("Select The Model", ["kmeans", "gmm"])
            if cfg["alg"] == "kmeans":
                clusters = st.number_input(
                    "Enter the number of clusters",
                    min_value=1,
                    max_value=None,
                    step=1,
                )
                # for additinal kwargs
                cfg["model_kw"]["n_clusters"] = clusters

            elif cfg["alg"] == "gmm":
                clusters = st.number_input(
                    "Enter the number of clusters",
                    min_value=1,
                    max_value=None,
                    step=1,
                )
                cfg["model_kw"]["n_components"] = clusters
            else:
                st.error("Not supported models")

            # cfg["pca"] = st.checkbox("Apply PCA")
            # if cfg["pca"]:
            #     pca_data = utils.process_data(
            #         back_DF, cfg, target, task_type, validation_size, selected_options, all=True
            #     )
            #     pca_fig, pca_comp = utils.PCA_visualization(pca_data)
            #     st.pyplot(pca_fig)
            #     pca_val = st.selectbox(
            #         "from this data select the number of PCA compenets you want",
            #         [i for i in range(1, pca_comp + 1)],
            #     )
            #     cfg["pca_comp"] = pca_val
            # cfg["apply_GridSearch"] = st.checkbox("Apply GridSearch")
            cfg["apply_GridSearch"] = False

        elif task_type == "Time":
            st.markdown("#### Time Series Options")
            cfg["task_type"] = task_type
            ts_kw = {}
            # cfg["clean"] = st.selectbox("🧹 Data Cleaning", ["Remove Missing Data", "Impute Missing Data"])
            # cfg["outlier"] = st.selectbox("📉 Handle Outliers", ["Don't Remove", "Use IQR", "Isolation Forest"])
            # temporary disable model selection and use Prophet Model by default
            # cfg["alg"] = st.selectbox("📈 Select Model", ["Prophet", "LSTM"])
            cfg["alg"] = "Prophet"

        if cfg["alg"] == "Prophet":
            # adjust frequency options to be selected by the user
            freq_options = {
                "Minutes": "min",
                "Hours": "h",
                "Days": "D",
                "Weeks": "W",
                "Months": "ME",
            }

            ts_kw["date_col"] = st.selectbox("Select The Date Column", DF.columns)
            ts_kw["target_col"] = target
            ts_kw["prophet_params"] = {}
            ts_kw["selected_cols"] = DF.columns.tolist()  # {}
            # Create a select box to choose frequency
            selected_freq_label = st.selectbox(
                "Select forecast frequency:", options=freq_options.keys()
            )
            selected_freq = freq_options[
                selected_freq_label
            ]  # Get corresponding frequency value
            num_of_points = st.number_input(
                f"Select how many {selected_freq_label.lower()} to forecast", value=1
            )
            ts_kw["freq"] = selected_freq
            ts_kw["f_period"] = int(num_of_points)
            ts_kw["validation_size"] = validation_size / 100
            cfg["ts_config"] = ts_kw

        cfg["apply_GridSearch"] = False

        print(cfg["task_type"])

    # Execute Task
    if st.button("🚀 Train Model"):
        utils.log_user_action("User IP", get_remote_ip())

        if task_type == "Classification":
            st.write("Perform classification task with option:")
            X_train, X_test, y_train, y_test = utils.process_data(
                DF, cfg, target, task_type, validation_size, selected_options
            )
            report = model(X_train, X_test, y_train, y_test, cfg)
            # st.write("Accuracy:")
            # st.write(report[0])
            # st.write("Confusion Matrix")
            # st.write(report[1])
            st.write("**Accuracy (%)**: Use when overall correctness matters most.")
            st.write(report["accuracy"] * 100)
            # st.write("**Precision (%)**: Use when false positives are costly (e.g., unnecessary alerts).")
            # st.write(report['precision']*100)
            # st.write("**Recall (%)**:  Use when missing true positives is critical (e.g., detecting fraud).")
            # st.write(report['recall']*100)
            # st.write("**F1-Score (%)**: Use for a balanced approach when data is uneven (both recall and percision "
            #          "matters).")
            # st.write(report['f1']*100)
        # st.write("**Confusion Matrix (%)**")
        # st.write(report['cm'])

        elif task_type == "Regression":
            st.write("Perform Regression task with option:")
            X_train, X_test, y_train, y_test = utils.process_data(
                DF, cfg, target, task_type, validation_size, selected_options
            )
            report = model(X_train, X_test, y_train, y_test, cfg)
            st.write("MSE:")
            st.write(report[0])
            st.write("R2:")
            st.write(report[1])

        elif task_type == "Time":
            st.write("Performing Time Series Analysis")
            ts_df = utils.process_data(
                DF, cfg, target, task_type, validation_size, selected_options, all=True
            )
            pf, report = model(ts_df, cfg=cfg)
            st.write(f"RMSE: {report[0]}")
            st.write(f"MAPE: {report[1]}")

            # Raw Data Plot
            raw_fig = pf.slide_display()
            st.plotly_chart(raw_fig, use_container_width=True)

            # Validation Data Plot
            forecast_fig = pf.plot_test_with_actual()
            st.plotly_chart(forecast_fig, use_container_width=True)

            with st.expander("📈 Understanding Components Plots"):
                st.write(
                    """
                - **Trend Plot**: Represents the overall pattern and long-term movement in the data ignoring short-term fluctuations.
                If your data has an increasing or decreasing pattern, this plot will show a rising or falling trend.
                A shaded blue area may be found representing the uncertainty interval, which widens over time since future predictions become more uncertain.
                - **Weekly Seasonality Plot**: Shows how values vary across different days of the week.
                The values on the Y-axis represent the relative effect of the seasonality component on the forecasted values
                (e.g., negative values indicate a decrease from the baseline on this weekday and vice versa).
                - **Daily Seasonality Plot**: Shows how values vary across different hours of the day.
                If the dataset doesn't containt an hourly timestamp, the daily seasonality plot may not be found.
                - **Yearly Seasonality Plot**: Shows how values vary across different months of the year.
                If the dataset doesn't cover a full year, the yearly seasonality plot may not be found.
                - **Extra Regressors Plot**: Shows how external variables impact the forecast.
                """
                )
            st.pyplot(pf.plot_component())

            # For Univariate inference
            if len(DF.columns) == 2:
                st.session_state.ts_preds = pf.inference()
                # Predicted Data Plot
                st.session_state.predictions_fig = pf.plot_predictions(
                    st.session_state.ts_preds
                )

            elif task_type == "Cluster":
                st.write("Perform Clustering task with option:")

    if task_type == "Cluster":
        cluster_df = utils.process_data(
            DF, cfg, target, task_type, validation_size, selected_options, all=True
        )
        report = model(cluster_df, cfg=cfg)
        # st.pyplot(report)
        X_test = (
            cluster_df.copy()
        )  # Ensure the test data does not include the target column
        predictions = inference(
            X_test, cfg
        )  # Replace this with your prediction function

        cluster_df["cluster"] = predictions  # Append predictions to the test data
        # cluster_df['Cluster'] = cluster_df['Predictions'].apply(lambda x: max(x, 1))
        st.success("✅ Predictions generated successfully!")
        st.write("Here is the test data with predictions:")
        st.dataframe(cluster_df)

        x_col = st.selectbox("Choose X-axis", options=cluster_df.columns[:-1])
        y_col = st.selectbox("Choose Y-axis", options=cluster_df.columns[:-1])

        # Only run plotting if both selections are made
        if st.button("🚀 Plot  Clusters"):
            cluster_plot = utils.cluster_scatter_plot(
                cluster_df, x_col, y_col, cluster_col="cluster"
            )
            st.write(cluster_plot)

        download_preds(cluster_df)
        st.stop()

    if task_type != "Time":
        # Inference Section
        st.markdown("### 🔍 What If / Inference")
        st.write("""Provide input values to test your trained model interactively.""")

        # Create an empty DataFrame for user inputs
        inf_df = pd.DataFrame(columns=DF.columns)

        # Dynamic User Input Form
        st.markdown("#### Provide Input Values for Prediction")
        input_cols = [column for column in DF.columns if column != target]
        for column in input_cols:
            if d_types.loc[column].values[0] == "object":  # Categorical columns
                user_input = st.selectbox(
                    f"Select a value for **{column}**:",
                    options=DF[column].unique(),
                    help=f"Choose a category for the column '{column}'.",
                )
            else:  # Numeric columns
                user_input = st.text_input(
                    f"Enter a value for **{column}**:",
                    value=str(np.mean(DF[column])),
                    help=f"Provide a numeric value for the column '{column}'.",
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
                classes = {
                    label: idx for idx, label in enumerate(sorted(DF[target].unique()))
                }
                st.write(f"Classes defined from target column: {classes}")
            except KeyError:
                classes = None
                st.warning(
                    "Unable to define classes automatically. Check your data or model."
                )

    if (task_type == "Time" and len(DF.columns) == 2) or task_type != "Time":
        # Run Inference Button
        if st.button("🚀 Run Inference"):
            try:
                if task_type == "Classification":
                    preds = inference(inf_df, cfg, True)
                    st.markdown("#### 📊 Model Prediction Results")

                    if classes:
                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    values=preds[0],
                                    labels=list(classes.keys()),
                                    hole=0.4,
                                )
                            ]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("Class Probabilities:")
                        st.write(preds)
                        # st.write({k: v for k, v in zip(classes.keys(), preds[0])})
                    else:
                        st.error(
                            "Class labels are missing. Unable to display probabilities."
                        )

                elif task_type == "Regression":
                    preds = max(inference(inf_df, cfg), 1)

                    st.markdown("#### 📊 Predicted Output")
                    st.write(f"**Prediction:** for {target} {preds}")
                    st.markdown(
                        """
                    **📝 Note**: Regression models predict continuous numeric values,
                    which can range over an interval rather than being limited to discrete categories.
                    """
                    )

                elif task_type == "Cluster":
                    preds = inference(inf_df, cfg)

                    st.markdown("#### 📊 Predicted Cluster")
                    st.write(f"**Predicted Cluster:**  {preds}")

                elif task_type == "Time":
                    st.dataframe(st.session_state.ts_preds)
                    st.plotly_chart(
                        st.session_state.predictions_fig, use_container_width=True
                    )
                    download_preds(st.session_state.ts_preds)

            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

    if (task_type == "Time" and len(DF.columns) > 2) or task_type != "Time":
        # Upload testing data
        # Add functionality to upload test data
        st.subheader("🔍 Upload Test Data and Predict")
        st.write(
            """
        Upload your test dataset to generate predictions using the selected model.
        Ensure the test dataset has the same structure (columns) as the training data.
        """
        )

        # File uploader for test data
        test_data_file = st.file_uploader(
            "Upload Test Data (CSV, Excel, or Pickle)",
            type=["csv", "xls", "xlsx", "pkl"],
            key="test_data_uploader",
        )

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
                    st.error(
                        "❌ Unsupported file type. Please upload a CSV, Excel, or Pickle file."
                    )
                    st.stop()

                # Validate test data structure
                if (
                    set(test_df.columns) != set(DF.columns) - {target}
                    and task_type != "Time"
                ):
                    st.error(
                        "❌ The columns in the test data must match the training data (excluding the target column)."
                    )
                    st.stop()

                elif task_type == "Time" and set(test_df.columns) != set(DF.columns) - {
                    target,
                    ts_kw["date_col"],
                }:
                    st.error(
                        "❌ The columns in the test data must match the training data (excluding the target and date columns)."
                    )
                    st.stop()

                elif task_type == "Time" and len(test_df) != ts_kw["f_period"]:
                    st.error(
                        "❌ The number of rows in the test data must match the selected number of points to forecast."
                    )
                    st.stop()
                else:
                    st.success("✅ Test data uploaded successfully!")
                    st.write("Here is a preview of your test data:")
                    st.dataframe(test_df.head())

                    # Run predictions
                    if st.button("🚀 Run Predictions"):
                        if task_type == "Time":
                            # Only for multivariate
                            test_df, preds_plot = inference(test_df, cfg)
                        else:
                            X_test = (
                                test_df.copy()
                            )  # Ensure the test data does not include the target column
                            predictions = inference(
                                X_test, cfg
                            )  # Replace this with your prediction function

                            test_df["Predictions"] = (
                                predictions  # Append predictions to the test data
                            )
                            test_df["Predictions"] = test_df["Predictions"].apply(
                                lambda x: max(x, 1)
                            )
                        st.success("✅ Predictions generated successfully!")
                        st.write("Here is the test data with predictions:")
                        st.dataframe(test_df)
                        if task_type == "Time":
                            st.plotly_chart(preds_plot, use_container_width=True)

                        download_preds(test_df)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
