import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import re
from shap.plots import colors
import matplotlib
from sklearn.decomposition import PCA

from scipy import stats
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# import plotly.graph_objects as go
import plotly.express as px
import logging
import os

# Define logger
user_logger = logging.getLogger("user_logs")
user_logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)  # Creates the logs directory if it doesn't exist

# Prevent adding multiple handlers
if not user_logger.handlers:
    handler = logging.FileHandler(
        "logs/user_actions.log", mode="a"
    )  # Create a file handler that directs log messages to a file
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )  # Formatter defines how the log messages look
    handler.setFormatter(formatter)  # Formats logs before saving them
    user_logger.addHandler(handler)  # Attach handler to the logger


def log_user_action(type, action):
    user_logger.info(f"{type}: {action}")


def list_wrap(x):
    """A helper to patch things since slicer doesn't handle arrays of arrays (it does handle lists of arrays)"""
    if isinstance(x, np.ndarray) and len(x.shape) == 1 and isinstance(x[0], np.ndarray):
        return [v for v in x]
    else:
        return x


labels = {
    "MAIN_EFFECT": "SHAP main effect value for\n%s",
    "INTERACTION_VALUE": "SHAP interaction value",
    "INTERACTION_EFFECT": "SHAP interaction value for\n%s and %s",
    "VALUE": "SHAP value (impact on model output)",
    "GLOBAL_VALUE": "mean(|SHAP value|) (average impact on model output magnitude)",
    "VALUE_FOR": "SHAP value for\n%s",
    "PLOT_FOR": "SHAP plot for %s",
    "FEATURE": "Feature %s",
    "FEATURE_VALUE": "Feature value",
    "FEATURE_VALUE_LOW": "Low",
    "FEATURE_VALUE_HIGH": "High",
    "JOINT_VALUE": "Joint SHAP value",
    "MODEL_OUTPUT": "Model output value",
}


def format_value(s, format_str):
    """Strips trailing zeros and uses a unicode minus sign."""
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r"\.?0+$", "", s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s


def missing(_df, clean_method="Remove Missing Data"):
    df = _df.copy()
    df.drop_duplicates(inplace=True)
    df.replace(to_replace=[None], value=np.nan, inplace=True)  # Added after testing
    df = df.dropna(axis=1, how="all")  # Added after testing

    if clean_method == "Remove Missing Data":
        df.dropna(inplace=True)
        return df

    elif clean_method == "Impute Missing Data":
        numeric_features = df.select_dtypes(include=np.number).columns
        if len(numeric_features) != 0:  # Added after testing
            numeric_imputer = SimpleImputer(strategy="mean")
            df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

        categorical_features = df.select_dtypes(include=object).columns
        if len(categorical_features) != 0:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_features] = categorical_imputer.fit_transform(
                df[categorical_features]
            )
        return df

    else:
        raise ValueError("Invalid input for imputing")


def resample(df: pd.DataFrame, date_col, target_col, freq):  # For TimeSeries data
    """
    Resample the data to the inferred frequency and fill missing values.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    others = [col for col in df.columns if col not in [target_col]]
    df_others = df[
        others
    ]  # A slice of the dataframe containing date_col and every other col except target_col
    _df: pd.DataFrame = df[[date_col, target_col]]
    _df = (
        _df.set_index(date_col, inplace=False).resample(freq).mean()
    )  # Mean for aggregation
    _df[target_col] = _df[target_col].interpolate(method="linear")
    _df = _df.reset_index()
    resampled = _df.merge(df_others, on=date_col, how="left")
    return resampled


def IQR(_df, lower_bound=0.25, upper_bound=0.75, multiplier=1.5):
    df = _df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]
    sub_df = df[numeric_cols]

    q1 = sub_df[numeric_cols].quantile(lower_bound)
    q3 = sub_df[numeric_cols].quantile(upper_bound)
    iqr = q3 - q1
    sub_df = sub_df[
        ~((sub_df < (q1 - multiplier * iqr)) | (sub_df > (q3 + multiplier * iqr))).any(
            axis=1
        )
    ]
    df = df.loc[sub_df.index]
    return df


def IF(_df):
    isolation_forest = IsolationForest()
    df = _df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]

    num_df = df[numeric_cols]
    outlier_pred = isolation_forest.fit_predict(num_df)

    clean_features = df[outlier_pred == 1]
    return clean_features


def remove_outliers(_df, method="Don't Remove Outliers"):
    if method == "Use IQR":
        return IQR(_df)
    elif method == "Use Isolation Forest":
        return IF(_df)
    else:
        return _df


def handle(_df, trg, split_value, cls="Classification"):
    X = _df.drop([trg], axis=1)
    y = _df[trg]

    if cls == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_value, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_value, shuffle=False
        )

    return X_train, X_test, y_train, y_test


def plot_numeric_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [col for col in numeric_cols if df[col].nunique() >= 10]

    plots = []
    for col in valid_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(data=df[col], ax=ax)

        skewness = stats.skew(df[col].dropna())
        _, p_value = stats.normaltest(df[col].dropna())

        ax.set_title(
            f"{col} Distribution\nSkewness: {skewness:.2f}, Normality Test p-value: {p_value:.4f}"
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

        plots.append(fig)

    excluded_cols = set(numeric_cols) - set(valid_cols)
    if excluded_cols:
        print(
            f"Excluded columns (less than 10 unique values): {', '.join(excluded_cols)}"
        )

    return plots


class PCADataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_names=None):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        df = pd.DataFrame(X)
        if self.column_names is not None:
            df.columns = self.column_names
        return df


def pca_preprocessing(df: pd.DataFrame):

    st = StandardScaler()
    if issparse(df):
        df = pd.DataFrame(df.toarray())

    df_uq = pd.DataFrame(
        [[i, len(df[i].unique())] for i in df.columns],
        columns=["Variable", "Unique Values"],
    ).set_index("Variable")
    excluded_cols = list(df_uq[df_uq["Unique Values"] == 2].index)
    columns_to_scale = df.columns.difference(excluded_cols)
    data_scaled = df.copy()
    data_scaled[columns_to_scale] = st.fit_transform(data_scaled[columns_to_scale])
    return data_scaled


def PCA_visualization(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [col for col in numeric_cols if df[col].nunique() >= 10]
    df = df[valid_cols]

    data_scaled = pca_preprocessing(df)

    # Apply PCA
    pca = PCA().fit(data_scaled)

    # Calculate the Cumulative Sum of the Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Set seaborn plot style
    sns.set(rc={"axes.facecolor": "#fcf0dc"}, style="darkgrid")

    # Plot the cumulative explained variance against the number of components
    plt.figure(figsize=(20, 10))

    # Bar chart for the explained variance of each component
    barplot = sns.barplot(
        x=list(range(1, len(cumulative_explained_variance) + 1)),
        y=explained_variance_ratio,
    )

    # Line plot for the cumulative explained variance
    (lineplot,) = plt.plot(
        range(0, len(cumulative_explained_variance)),
        cumulative_explained_variance,
        marker="o",
        linestyle="--",
    )

    # Set labels and title
    plt.xlabel("Number of Components", fontsize=14)
    plt.ylabel("Explained Variance", fontsize=14)
    plt.title("Cumulative Variance vs. Number of Components", fontsize=18)

    # Customize ticks and legend
    plt.xticks(range(0, len(cumulative_explained_variance)))
    plt.legend(
        handles=[barplot.patches[0], lineplot],
        labels=[
            "Explained Variance of Each Component",
            "Cumulative Explained Variance",
        ],
        loc=(0.62, 0.1),
        frameon=True,
    )

    # Display the variance values for both graphs on the plots
    x_offset = -0.3
    y_offset = 0.01
    for i, (ev_ratio, cum_ev_ratio) in enumerate(
        zip(explained_variance_ratio, cumulative_explained_variance)
    ):
        plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
        if i > 0:
            plt.text(
                i + x_offset,
                cum_ev_ratio + y_offset,
                f"{cum_ev_ratio:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.grid(axis="both")
    return plt, pca.n_components_


def pca_data(df: pd.DataFrame, n_components: int):
    pca = PCA(n_components=n_components)
    data_for_pca = pca_preprocessing(df)
    proc_data_pca = pca.fit_transform(data_for_pca)

    return pd.DataFrame(
        proc_data_pca, columns=["PC" + str(i + 1) for i in range(pca.n_components_)]
    )


def HeatMap(_df: pd.DataFrame):
    return _df.select_dtypes("number").corr()


def corr_plot(_df: pd.DataFrame):
    corr_matrix = _df.select_dtypes("number").corr()
    tril_index = np.tril_indices_from(corr_matrix)
    for coord in zip(*tril_index):
        corr_matrix.iloc[coord[0], coord[1]] = np.nan

    corr_values = (
        corr_matrix.stack()
        .to_frame()
        .reset_index()
        .rename(
            columns={"level_0": "feature1", "level_1": "feature2", 0: "correlation"}
        )
    )
    corr_values["abs_correlation"] = corr_values.correlation.abs()
    return corr_values


def inf_proc(item):
    try:
        fixed_item = float(item)
        return fixed_item
    except Exception:
        return item


def check_empty(df):
    return True if df.empty else False


def descriptive_analysis(df):
    if check_empty(df):
        raise ValueError("The uploaded DataFrame is empty")
    cat_des_analysis = None
    num_des_analysis = None
    if len(df.select_dtypes(include=np.number).columns) != 0:
        num_des_analysis = df.describe().T
    if len(df.select_dtypes(include=object).columns) != 0:
        cat_des_analysis = df.describe(include="object").T
    d_types = pd.DataFrame(df.dtypes, columns=["type"])
    missing_percentage = pd.DataFrame(
        (df.isna().sum() / len(df)) * 100, columns=["missing %"]
    ).round(2)
    dups_percentage = (len(df[df.duplicated()]) / len(df)) * 100
    unq_percentage = unique_percentage(df)

    return (
        num_des_analysis,
        cat_des_analysis,
        d_types,
        missing_percentage,
        dups_percentage,
        unq_percentage,
    )


def outlier_inlier_plot(df):
    model = IsolationForest(contamination=0.05, random_state=0)
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]

    num_df = df[numeric_cols]
    num_df["Outlier_Scores"] = model.fit_predict(num_df.iloc[:, 1:].to_numpy())

    num_df["Is_Outlier"] = [1 if x == -1 else 0 for x in num_df["Outlier_Scores"]]

    outlier_percentage = num_df["Is_Outlier"].value_counts(normalize=True) * 100

    plt.figure(figsize=(12, 4))
    outlier_percentage.plot(kind="barh")

    for index, value in enumerate(outlier_percentage):
        plt.text(value, index, f"{value:.2f}%", fontsize=15)

    plt.title("Percentage of Inliers and Outliers")
    plt.xticks(ticks=np.arange(0, 115, 5))
    plt.xlabel("Percentage (%)")
    plt.ylabel("Is Outlier")
    plt.gca().invert_yaxis()

    return plt


def convert_numeric(df):
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except Exception:
            pass
    return df


def process_data(_df, cfg, target, task_type, split_value, selected_options, all=False):
    log_user_action("cfgs", cfg)
    log_user_action("Dropped Columns", selected_options)
    log_user_action("Original Number of Rows", _df.shape[0])
    log_user_action("Original Number of Columns", _df.shape[1])

    if cfg["outlier"] != "Use Isolation Forest":
        # Remove outliers before imputation for a more precise mean calculation
        if task_type != "Time":
            _DF = remove_outliers(_df, cfg["outlier"])
            _DF = missing(_DF, cfg["clean"])
        else:
            _DF = remove_outliers(_df, cfg["outlier"])
            _DF = resample(
                _DF,
                cfg["ts_config"]["date_col"],
                cfg["ts_config"]["target_col"],
                cfg["ts_config"]["freq"],
            )
            _DF = missing(_DF, cfg["clean"])
    else:  # Because Isolation Forest doesn't handle null values
        if task_type != "Time":
            _DF = missing(_df, cfg["clean"])
            _DF = remove_outliers(_DF, cfg["outlier"])
        else:
            _DF = missing(_df, cfg["clean"])
            _DF = remove_outliers(_df, cfg["outlier"])
            _DF = resample(
                _DF,
                cfg["ts_config"]["date_col"],
                cfg["ts_config"]["target_col"],
                cfg["ts_config"]["freq"],
            )
            _DF = missing(_DF, cfg["clean"])

    log_user_action("Number of Rows After Processing", _DF.shape[0])
    log_user_action("Number of columns After Processing", _DF.shape[1])
    if all:  # Not to split data when doing clustering or time series
        return _DF

    X_train, X_test, y_train, y_test = handle(_DF, target, split_value, task_type)
    return X_train, X_test, y_train, y_test


class SkewnessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, skew_limit=0.8, forced_fix=False):
        self.skew_limit = skew_limit
        self.forced_fix = forced_fix
        self.method_dict = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.method_dict = self.extracrt_recommeneded_features(X)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X_transformed = X.copy()
        for method, features in self.method_dict.items():

            if method == "log":
                # Apply log transformation to the specified features
                X_transformed[:, features] = np.log1p(X_transformed[:, features])
            elif method == "sqrt":
                # Apply square root transformation to the specified features
                X_transformed[:, features] = np.sqrt(X_transformed[:, features])
            elif method == "boxcox":
                # Apply Box-Cox transformation to the specified features
                for feature in features:
                    X_transformed[:, feature], _ = stats.boxcox(
                        X_transformed[:, feature]
                    )
            elif method == "yeojohnson":
                for feature in features:
                    X_transformed[:, feature], _ = stats.yeojohnson(
                        X_transformed[:, feature]
                    )
            elif method == "cube":
                # Apply Cube transformation to the specified features
                X_transformed[:, features] = np.cbrt(X_transformed[:, features])

        return X_transformed

    def extracrt_recommeneded_features(self, X):
        skew_vals = np.abs(stats.skew(X, axis=0))
        skew_col_indices = np.where(skew_vals > self.skew_limit)[0]
        method_dict = {}

        for feature_idx in skew_col_indices:
            feature = X[:, feature_idx]

            method = self.recommend_skewness_reduction_method(feature, self.forced_fix)
            if method not in method_dict:
                method_dict[method] = [feature_idx]
            else:
                method_dict[method].append(feature_idx)

        return method_dict

    def recommend_skewness_reduction_method(
        self, feature: pd.Series, forced_fix=False
    ) -> str:

        skewness_dict = {}
        all = {}

        transformed_log = np.log1p(feature)
        _, p_value = stats.normaltest(transformed_log)

        # The p-value is a measure of the evidence against the null hypothesis of normality.
        # A low p-value (typically less than 0.05) suggests that the data is significantly different from a normal distribution,
        # indicating that the fix for skewness was not successful in achieving normality.
        if p_value > 0.05:
            skewness_dict["log"] = p_value
        else:
            all["log"] = p_value

        transformed_sqrt = np.sqrt(feature)
        _, p_value = stats.normaltest(transformed_sqrt)
        if p_value > 0.05:
            skewness_dict["sqrt"] = p_value
        else:
            all["sqrt"] = p_value

        if (feature < 0).any() or (feature == 0).any():
            transformed_yeojohnson, _ = stats.yeojohnson(feature)
            _, p_value = stats.normaltest(transformed_yeojohnson)
            if p_value > 0.05:
                skewness_dict["yeojohnson"] = p_value
            else:
                all["yeojohnson"] = p_value

        else:
            transformed_boxcox, _ = stats.boxcox(feature + 0.0001)
            _, p_value = stats.normaltest(transformed_boxcox)
            if p_value > 0.05:
                skewness_dict["boxcox"] = p_value
            else:
                all["boxcox"] = p_value

        transformed_cbrt = np.cbrt(feature)
        _, p_value = stats.normaltest(transformed_cbrt)
        if p_value > 0.05:
            skewness_dict["cube"] = p_value
        else:
            all["cube"] = p_value

        if len(skewness_dict) > 0:
            return max(skewness_dict, key=lambda y: abs(skewness_dict[y]))
        else:
            if forced_fix:
                print("No Fix, using best transformers")
                return max(all, key=lambda y: abs(all[y]))
            else:
                return "No Fix"


def unique_percentage(df):
    return (
        pd.DataFrame(
            {
                "Column": df.columns,
                "Unique_Percentage": [
                    (df[col].nunique() / len(df[col])) * 100 for col in df.columns
                ],
            }
        )
        .sort_values(by="Unique_Percentage", ascending=False)
        .reset_index(drop=True)
    )


def my_waterfall(
    values,
    shap_values_base,
    shap_values_display_data,
    shap_values_data,
    feature_names,
    max_display=-1,
    show=False,
    lower_bounds=None,
    upper_bounds=None,
):
    """
    For shap_values is an object of Explanation:

    sv_shape: shap_values.shape
    shap_values_base: shap_values.base_values
    shap_values_display_data: shap_values.display_data
    shap_values_data: shap_values.data
    feature_names: shap_values.feature_names
    values: shap_values.values
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    """

    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # make sure we only have a single explanation to plot
    # sv_shape = shap_values.shape
    # if len(sv_shape) != 1:
    #     emsg = (
    #         "The my_waterfall plot can currently only plot a single explanation, but a "
    #         f"matrix of explanations (shape {sv_shape}) was passed!"
    #     )
    #     raise ValueError(emsg)

    base_values = float(shap_values_base)

    features = (
        shap_values_display_data
        if shap_values_display_data is not None
        else shap_values_data
    )
    # feature_names = shap_values.feature_names
    # values = shap_values.values

    # unwrap pandas series
    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array(
            [labels["FEATURE"] % str(i) for i in range(len(values))]
        )

    # init variables we use for tracking the plot locations
    if max_display == -1 or max_display >= int(len(values)):
        num_features = int(len(values))
    else:
        num_features = max_display
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))

    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(12, num_features * row_height + 4)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color="#bbbbbb",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:

            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = (
                    format_value(float(features[order[i]]), "%0.03f")
                    + " = "
                    + feature_names[order[i]]
                )
            else:
                yticklabels[rng[i]] = (
                    str(features[order[i]]) + " = " + str(feature_names[order[i]])
                )

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=colors.red_rgb,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=colors.blue_rgb,
        alpha=0,
    )

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            max(dist - hl_scaled, 0.000001),
            0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array(
                    [[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]
                ),
                ecolor=colors.light_red_rgb,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            format_value(pos_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="left",
                verticalalignment="center",
                color=colors.red_rgb,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -max(-dist - hl_scaled, 0.000001),
            0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array(
                    [[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]
                ),
                ecolor=colors.light_blue_rgb,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            format_value(neg_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="right",
                verticalalignment="center",
                color=colors.blue_rgb,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    plt.yticks(
        ytick_pos,
        yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]],
        fontsize=13,
    )

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(
        base_values,
        0,
        1 / num_features,
        color="#bbbbbb",
        linestyle="--",
        linewidth=0.5,
        zorder=-1,
    )
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    # plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(
        [base_values, base_values + 1e-8]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(
        ["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"],
        fontsize=12,
        ha="left",
    )
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
    ax3.set_xticklabels(
        ["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left"
    )
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_color("#999999")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(
            22 / 72.0, -1 / 72.0, fig.dpi_scale_trans
        )
    )

    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if show:
        plt.show()
    else:
        return plt


def og_waterfall(shap_values, max_display=10, show=True):
    # return
    """Plots an explanation of a single prediction as a waterfall plot.

    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features.

    Features are sorted by the magnitude of their SHAP values with the smallest
    magnitude features grouped together at the bottom of the plot when the number of
    features in the models exceeds the ``max_display`` parameter.

    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional :class:`.Explanation` object that contains the feature values and SHAP values to plot.

    max_display : int
        The maximum number of features to display (default is 10).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after it
        has been created, returning the current axis via plt.gca().

    Examples
    --------
    See `waterfall plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html>`_.

    """
    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # make sure we only have a single explanation to plot
    sv_shape = shap_values.shape
    if len(sv_shape) != 1:
        emsg = (
            "The waterfall plot can currently only plot a single explanation, but a "
            f"matrix of explanations (shape {sv_shape}) was passed! Perhaps try "
            "`shap.plots.waterfall(shap_values[0])` or for multi-output models, "
            "try `shap.plots.waterfall(shap_values[0, 0])`."
        )
        raise ValueError(emsg)

    base_values = float(shap_values.base_values)

    features = (
        shap_values.display_data
        if shap_values.display_data is not None
        else shap_values.data
    )

    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # unwrap pandas series
    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array(
            [labels["FEATURE"] % str(i) for i in range(len(values))]
        )

    # init variables we use for tracking the plot locations
    num_features = len(values)
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color="#bbbbbb",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = (
                    format_value(float(features[order[i]]), "%0.03f")
                    + " = "
                    + feature_names[order[i]]
                )
            else:
                yticklabels[rng[i]] = (
                    str(features[order[i]]) + " = " + str(feature_names[order[i]])
                )

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=colors.red_rgb,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=colors.blue_rgb,
        alpha=0,
    )

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            max(dist - hl_scaled, 0.000001),
            0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array(
                    [[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]
                ),
                ecolor=colors.light_red_rgb,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            format_value(pos_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="left",
                verticalalignment="center",
                color=colors.red_rgb,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -max(-dist - hl_scaled, 0.000001),
            0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array(
                    [[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]
                ),
                ecolor=colors.light_blue_rgb,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            format_value(neg_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="right",
                verticalalignment="center",
                color=colors.blue_rgb,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    plt.yticks(
        ytick_pos,
        yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]],
        fontsize=13,
    )

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(
        base_values,
        0,
        1 / num_features,
        color="#bbbbbb",
        linestyle="--",
        linewidth=0.5,
        zorder=-1,
    )
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    # plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(
        [base_values, base_values + 1e-8]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(
        ["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"],
        fontsize=12,
        ha="left",
    )
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
    ax3.set_xticklabels(
        ["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left"
    )
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_color("#999999")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(
            22 / 72.0, -1 / 72.0, fig.dpi_scale_trans
        )
    )

    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if show:
        plt.show()
    else:
        return plt.gca()


def silhouette_analysis(df, start_k, stop_k, figsize=(20, 50)):
    """
    Perform Silhouette analysis for a range of k values and visualize the results.
    """
    # comment the pca in this step
    # df= pca_data(df, 3)
    # Set the size of the figure
    plt.figure(figsize=figsize)

    # Create a grid with (stop_k - start_k + 1) rows and 2 columns
    grid = gridspec.GridSpec(stop_k - start_k + 1, 2)

    # Assign the first plot to the first row and both columns
    _ = plt.subplot(grid[0, :])

    # First plot: Silhouette scores for different k values
    sns.set_palette(["darkorange"])

    silhouette_scores = []

    # Iterate through the range of k values
    for k in range(start_k, stop_k + 1):
        km = KMeans(
            n_clusters=k, init="k-means++", n_init=10, max_iter=100, random_state=0
        )
        km.fit(df)
        labels = km.predict(df)
        score = silhouette_score(df, labels)
        silhouette_scores.append(score)

    best_k = start_k + silhouette_scores.index(max(silhouette_scores))

    plt.plot(range(start_k, stop_k + 1), silhouette_scores, marker="o")
    plt.xticks(range(start_k, stop_k + 1))
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Average Silhouette Score for Different k Values", fontsize=15)

    # Add the optimal k value text to the plot
    optimal_k_text = f"The k value with the highest Silhouette score is: {best_k}"
    plt.text(
        10,
        0.23,
        optimal_k_text,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(facecolor="#fcc36d", edgecolor="#ff6200", boxstyle="round, pad=0.5"),
    )

    # Second plot (subplot): Silhouette plots for each k value
    colors = sns.color_palette("bright")

    for i in range(start_k, stop_k + 1):
        km = KMeans(
            n_clusters=i, init="k-means++", n_init=10, max_iter=100, random_state=0
        )
        row_idx, col_idx = divmod(i - start_k, 2)

        # Assign the plots to the second, third, and fourth rows
        ax = plt.subplot(grid[row_idx + 1, col_idx])

        visualizer = SilhouetteVisualizer(km, colors=colors, ax=ax)
        visualizer.fit(df)

        # Add the Silhouette score text to the plot
        score = silhouette_score(df, km.labels_)
        ax.text(
            0.97,
            0.02,
            f"Silhouette Score: {score:.2f}",
            fontsize=12,
            ha="right",
            transform=ax.transAxes,
            color="red",
        )

        ax.set_title(f"Silhouette Plot for {i} Clusters", fontsize=15)

    plt.tight_layout()
    return plt


def cluster_scatter_plot(df, x_col, y_col, cluster_col="cluster"):
    # """Create an interactive scatter plot for clustering results."""
    # fig = go.Figure()
    #
    # # Add scatter plot for each cluster
    # for cluster in df[cluster_col].unique():
    #     cluster_data = df[df[cluster_col] == cluster]
    #     fig.add_trace(
    #         go.Scatter(
    #             x=cluster_data[x_col],
    #             y=cluster_data[y_col],
    #             mode="markers",
    #             name=f"Cluster {cluster}",
    #             marker=dict(size=10, opacity=0.8),
    #             text=cluster_data[cluster_col],  # Add cluster labels as hover text
    #         )
    #     )
    #
    # # Update layout to ensure continuous axes
    # fig.update_layout(
    #     # xaxis=dict(title=x_col, type='linear'),  # Ensure x-axis is treated as continuous
    #     # yaxis=dict(title=y_col, type='linear'),  # Ensure y-axis is treated as continuous
    #     title="Cluster Scatter Plot",
    #     showlegend=True,
    # )
    cluster_order = (
        df[cluster_col].value_counts(normalize=True).sort_values(ascending=True)
    )
    # Create a scatter plot using Plotly Express
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="cluster",
        labels={x_col: x_col, y_col: y_col},
        color_continuous_scale=px.colors.sequential.Plasma,
        category_orders={"cluster": cluster_order.to_dict()},
    )

    # Update marker size and edge color
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")))

    return fig
