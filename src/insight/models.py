from abc import ABC, abstractmethod
import warnings

from .utils import SkewnessTransformer, silhouette_analysis, PCADataFrameTransformer
from .model_utils import grid_dict

import pickle
import pandas as pd, numpy as np

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    PolynomialFeatures,
    MinMaxScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNet,
    Lasso,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from .ts_models import ProphetModel

import matplotlib

matplotlib.use("Agg")
# prophet_kw= {'date_col': None,
#              'target_col': None}

models_dict = {
    "SVC": SVC(probability=True),
    "LR": LogisticRegression(),
    "Linear Regression": LinearRegression(),
    "ElasticNet": ElasticNet(),
    "KNN_cls": KNeighborsClassifier(),
    "KNN_reg": KNeighborsRegressor(),
    "RF_cls": RandomForestClassifier(),
    "XGB_cls": XGBClassifier(),
    "XGB_reg": XGBRegressor(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "extra_tree": ExtraTreesClassifier(),
    "SVR": SVR(),
    "GradientBoosting_cls": GradientBoostingClassifier(),
    "Adaboost": AdaBoostClassifier(),
    "DecisionTree_cls": DecisionTreeClassifier(),
    "kmeans": KMeans(),
    "dbscan": DBSCAN(),
    "gmm": GaussianMixture
}

normilizers = {
    "standard": StandardScaler(),
    "min-max": MinMaxScaler(feature_range=(0, 1)),
}


class BaseModel(ABC):
    def __init__(self, algorithm, grid=False):
        self.pipeline = None
        self.algorithm = algorithm
        self.grid = grid

    @abstractmethod
    def build_pipeline(self, X, poly_feat=False, skew_fix=False):
        """This method must be implemented by subclasses and should set self.pipeline."""
        pass

    def ensure_pipeline_built(self):
        """Ensure the pipeline has been built by checking if self.pipeline is set."""
        if self.pipeline is None:
            raise ValueError(
                "The pipeline has not been built. Ensure `build_pipeline` sets self.pipeline."
            )

    def fit(self, X, y):
        self.ensure_pipeline_built()
        self.pipeline.fit(X, y)

    def predict(self, X):
        self.ensure_pipeline_built()
        return self.pipeline.predict(X)


class KFoldModel(BaseEstimator, TransformerMixin):
    def __init__(self, model, n_splits=5, shuffle=True, random_state=None):
        self.model = model
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv = None
        self.scores_ = []
        self.best_estimators_ = []

    def fit(self, X, y):
        self.cv = KFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )
        self.scores_ = []
        self.best_estimators_ = []

        for train_index, val_index in tqdm(
            self.cv.split(X), total=self.n_splits, desc="K-Fold Progress"
        ):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            self.model.fit(X_train, y_train)
            self.best_estimators_.append(self.model.best_estimator_)

            score = accuracy_score(y_val, self.model.predict(X_val))
            self.scores_.append(score)

        return self

    def predict(self, X):
        return self.best_estimators_[-1].predict(X)


class GridSearchModel(BaseEstimator, TransformerMixin):
    def __init__(self, alg, grid_params=None):
        self.alg = alg
        self.grid_params = grid_params if grid_params is not None else {}
        self.grid_search = None
        self.best_estimator_ = None

    def fit(self, X, y=None):
        self.grid_search = GridSearchCV(
            estimator=self.alg, param_grid=self.grid_params, cv=3
        )
        self.grid_search.fit(X, y)
        print("grid search applied")

        self.best_estimator_ = self.grid_search.best_estimator_
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_prob(self, X):
        return self.best_estimator_.predict_proba(X)


class Model:
    def __init__(self, algorithm, grid=False, model_kws={}):
        """
        Initialize the model with an algorithm and optional grid search.

        Parameters:
        algorithm (sklearn-compatible algorithm)
        grid (bool): Whether to apply grid search. Default is False.
        """
        self.pipeline = None
        self.model = None
        self.label_encoder = LabelEncoder()
        if grid is True and len(model_kws) != 0:
            warnings.warn(
                "Can't use grid search with predefined model kwargs, setting kwargs to None..."
            )
            model_kws = None

        if algorithm in models_dict.keys():
            self.algorithm = models_dict[algorithm]
        else:
            raise NotImplementedError

        if model_kws is not None:
            self.algorithm.set_params(**model_kws)

        self.grid = grid

        if grid:
            self.grid_model = GridSearchModel(
                alg=self.algorithm, grid_params=grid_dict[algorithm]
            )

    def build_pipeline(self, X, poly_feat=False, skew_fix=False):
        """
        Build the preprocessing pipeline.

        Parameters:
        X (pd.DataFrame): The input data to derive preprocessing steps.
        poly_feat (bool): Whether to apply polynomial feature generation.
        skew_fix (bool): Whether to apply skewness correction.
        """
        categorical_features = X.select_dtypes("object").columns.tolist()
        numerical_features = X.select_dtypes("number").columns.tolist()

        categorical_transformer = Pipeline(
            steps=[
                (
                    "cat_imp",
                    SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
                ),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numerical_transformer = Pipeline(
            steps=[("num_imp", SimpleImputer(missing_values=np.nan, strategy="mean"))]
        )

        if not isinstance(self.algorithm, KMeans):
            numerical_transformer.steps.append(
                ("scaler", MinMaxScaler(feature_range=(0, 1)))
            )  # Only scale if not KMeans

        if skew_fix:
            numerical_transformer.steps = [
                step for step in numerical_transformer.steps if step[0] != "num_imp"
            ]
            numerical_transformer.steps.append(
                ("skew_fix", SkewnessTransformer(skew_limit=0.75))
            ),
            numerical_transformer.steps.append(
                ("num_imp", SimpleImputer(missing_values=np.nan, strategy="mean"))
            )

        if poly_feat:
            numerical_transformer.steps.append(
                ("Polynomial_Features", PolynomialFeatures(degree=2))
            )
            print("poly features applied")

        if isinstance(self.algorithm, KMeans):
            numerical_transformer.steps.append(
                ("to_dataframe", PCADataFrameTransformer())
            )

        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, categorical_features),
                ("numerical", numerical_transformer, numerical_features),
            ]
        )

        if self.grid:
            model_step = ("model", self.grid_model)

        else:
            model_step = ("model", self.algorithm)

        steps = [("preprocessor", preprocessor), model_step]

        self.pipeline = Pipeline(steps=steps)

    def reverse_label(self, y):
        classes = self.label_encoder.classes_
        encoded_values = self.label_encoder.transform(classes)
        d = {cls: enc for cls, enc in zip(classes, encoded_values)}

        return d, self.label_encoder.inverse_transform(y)

    def encode_label(self, y):
        classes = self.label_encoder.classes_
        encoded_values = self.label_encoder.transform(classes)
        d = {cls: enc for cls, enc in zip(classes, encoded_values)}

        return d, self.label_encoder.transform(y)

    def train(self, X: pd.DataFrame, y: pd.Series = None, skew=False, poly=False):
        """
        Train the model.

        Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): Target labels (if applicable, for supervised tasks).
        skew (bool): Apply skewness correction.
        poly (bool): Apply polynomial features.
        """
        self.build_pipeline(X, skew_fix=skew, poly_feat=poly)
        if y is not None and y.dtypes == "object":
            y = self.label_encoder.fit_transform(y)

        self.pipeline.fit(X, y)
        self.model = self.pipeline.named_steps["model"]

    def preprocess(self, X):
        """
        Preprocess the input data using the fitted pipeline.

        Parameters:
        X (pd.DataFrame): The input data to preprocess.
        """
        return self.pipeline.named_steps["preprocessor"].transform(X)

    def predict(self, X):
        """
        Predict clusters or labels for the input data.

        Parameters:
        X (pd.DataFrame): The input data.

        Returns:
        array-like: Predicted clusters or labels.
        """
        X = self.preprocess(X)
        return self.model.predict(X)

    def predict_prob(self, X):
        """
        Predict probabilities for the input data (for models supporting probabilities).

        Parameters:
        X (pd.DataFrame): The input data.

        Returns:
        array-like: Predicted probabilities.
        """
        X = self.preprocess(X)
        return self.model.predict_proba(X)

    def cls_metrics(self, X: pd.DataFrame, y_true: pd.Series):
        """
        Compute classification metrics (accuracy and confusion matrix).

        Parameters:
        X (pd.DataFrame): Input features.
        y_true (pd.Series): True labels.

        Returns:
        tuple: Confusion matrix and accuracy score.
        """
        y_pred = self.predict(X)
        if y_true.dtypes == "object":
            y_pred = self.label_encoder.inverse_transform(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # recall = recall_score(y_true, y_pred,pos_label=self.label_encoder.inverse_transform([1]))
        # perc = precision_score(y_true, y_pred,pos_label=self.label_encoder.inverse_transform([1]))
        # f1 = f1_score(y_true,y_pred,pos_label=self.label_encoder.inverse_transform([1]))

        metrics_dict = {
            "cm": cm,
            "accuracy": accuracy,
            # 'recall':recall,
            # 'precision':perc,
            # 'f1':f1
        }

        # return cm, accuracy
        # return the whole dict
        return metrics_dict

    def reg_metrics(self, X, y_true):
        """
        Compute regression metrics (MSE and R2 score).

        Parameters:
        X (pd.DataFrame): Input features.
        y_true (pd.Series): True target values.

        Returns:
        tuple: Mean squared error and R2 score.
        """
        y_pred = self.predict(X)

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mse, r2

    def cluster_metrics(self, X):
        X = self.preprocess(X)
        return silhouette_analysis(X, 2, 15)

    def save_model(self, file_path):
        """
        Save the trained model to a file.

        Parameters:
        file_path (str): Path to save the model.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved successfully as: {file_path}")


def model(X_train=None, X_test=None, y_train=None, y_test=None, cfg=None):
    # global prophet_kw
    if cfg["task_type"] == "Time":
        prophet_kw = cfg["ts_config"]
        print("Setting Prophet args")
        pf = ProphetModel(**prophet_kw)
        pf.fit_transform(X_train)
        return pf

    _model = Model(cfg["alg"], cfg["apply_GridSearch"], model_kws=cfg["model_kw"])
    _model.train(X_train, y_train, cfg["skew_fix"], cfg["poly_feat"])
    if cfg["save"]:
        _model.save_model("model.pkl")

    if cfg["task_type"] == "Classification":
        # p= _model.predict_prob(X_test)
        # cm, acc= _model.cls_metrics(X_test, y_test)
        metrics_dict = _model.cls_metrics(X_test, y_test)
        return metrics_dict

    elif cfg["task_type"] == "Regression":
        mse, r2 = _model.reg_metrics(X_test, y_test)
        return mse, r2

    else:
        return None
        # return _model.cluster_metrics(X_train)


def inference(X, proba=False):
    try:
        _model: Model
        with open("model.pkl", "rb") as f:
            _model = pickle.load(f)

        if proba:
            return _model.predict_prob(X)

        return _model.pipeline.predict(X)
    except FileNotFoundError:
        print("Model file not found.")

    except pickle.UnpicklingError:
        print("Error loading the pickle model.")


def get_corresponding_labels(y, encode=False):
    try:
        with open("model.pkl", "rb") as f:
            _model = pickle.load(f)

        if encode:
            return _model.encode_label([y])

        return _model.reverse_label([y])
    except FileNotFoundError:
        print("Model file not found.")

    except pickle.UnpicklingError:
        print("Error loading the pickle model.")
