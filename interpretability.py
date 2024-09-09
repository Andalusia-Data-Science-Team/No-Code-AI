import shap
import lime
import lime.lime_tabular
import pandas as pd, numpy as np
from models import GridSearchModel

from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle

import plotly.express as px

import matplotlib
matplotlib.use('Agg')

class Interpretability:
    def __init__(self, model, model_type, X_train, X_test, y_train=None, y_test=None):

        _model = model.named_steps['model']
        self.processor= model.named_steps['preprocessor']
        self.model_type = model_type
        self.X_train = self.processor.transform(X_train)
        self.X_test = self.processor.transform(X_test)
        cats= self.processor.named_transformers_['categorical'].get_feature_names_out().tolist()
        nums= self.processor.named_transformers_['numerical'].get_feature_names_out().tolist()
        self.all= cats + nums
        self.y_train = y_train
        self.y_test = y_test
        self.explainer = None
        self.shap_values = None
        self.lime_explainer = None
        self.model= self.model_check(_model)
        self._compute_shap_values()

    @property
    def get_shape_vals(self):
        return self.shap_values

    def _compute_shap_values(self):
        """
        TODO
        one output: array of shape (#num_samples, *X.shape[1:]).
        multiple outputs: array of shape (#num_samples, *X.shape[1:], #num_outputs).
        """
        if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier,
                                   ExtraTreesClassifier, XGBClassifier, XGBRegressor)):
            self.explainer = shap.TreeExplainer(self.model, feature_names= self.all)

        elif isinstance(self.model, (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)):
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        elif isinstance(self.model, (SVC, SVR, KNeighborsClassifier, KNeighborsRegressor, AdaBoostClassifier)):
            if self.model_type == 'classification':
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, self.X_train)
        else:
            # Default to KernelExplainer for any unsupported models
            self.explainer = shap.KernelExplainer(self.model.predict, self.X_train)


        self.shap_values = self.explainer.shap_values(self.X_test)
                
    def plot_variable_importance(self):
        shap.summary_plot(self.shap_values, self.X_test, feature_names= self.all)
        return plt
    
    def plot_shap_summary(self, feature_names, summary_type= 'agg', top_k= 10):
        # if classification:
        assert summary_type in ('agg', 'detailed'), f'the summary type {summary_type} is not supported!'
        # Changed in version 0.45.0: Return type for models with multiple outputs changed from list to np.ndarray.
        if isinstance(self.shap_values, list):
            shap_values= self.shap_values[0]
        else:
            shap_values = self.shap_values
        num_classes = shap_values.shape[2]
        tmp = {
            f'class {class_idx}': np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
            for class_idx in range(num_classes)
        }
        importance_df = pd.DataFrame(tmp)
        importance_df['agg']= importance_df.sum(axis= 1)
        # feature_names should set the index
        importance_df.index = importance_df.index.astype(str)

        if summary_type == 'agg':
            cls_imp= importance_df['agg'].sort_values(ascending= False)[:top_k]
            fig = px.bar(cls_imp, x=cls_imp.values, y=cls_imp.index, orientation='h', title= f"Feature Importance of the Aggregated Classes")
            fig.update_layout(
                xaxis_range=[0, cls_imp.max() * 1.1], # padding max val
                height=600, 
                xaxis_title="Feature Importance",
                yaxis_title="Features",
                yaxis={'categoryorder':'total ascending'},
                showlegend=False
            )
            return fig
        else:
            res= {}
            # excluding the agg 
            for i in range(importance_df.shape[1] - 1):
                cls_imp= importance_df[f'class {i}'].sort_values(ascending= False)[:top_k]
                fig = px.bar(cls_imp, x=cls_imp.values, y=cls_imp.index, orientation='h', title= f"Feature Importance of Class {i}")
                fig.update_layout(
                    xaxis_range=[0, cls_imp.max() * 1.1], # padding max val
                    height=600, 
                    xaxis_title="Feature Importance",
                    yaxis_title="Features",
                    yaxis={'categoryorder':'total ascending'},
                    showlegend=False
                )
                res[f'result_{i}']= fig
            
            return res

    def plot_summary_label(self, label_index):
        if self.model_type == 'classification' and self.shap_values is not None:
            shap.summary_plot(self.shap_values[label_index], self.X_test, feature_names= self.all)

    def plot_dependence(self, feature_name):
        if self.shap_values is not None:
            shap.dependence_plot(feature_name, self.shap_values, self.X_test)
            return plt

    def explain_with_lime(self, num_features=5, num_samples=1000):
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values, 
            feature_names=self.X_train.columns, 
            class_names=np.unique(self.y_train) if self.model_type == 'classification' else None,
            discretize_continuous=True
        )
        i = np.random.randint(0, self.X_test.shape[0])
        exp = self.lime_explainer.explain_instance(self.X_test.iloc[i].values, 
                                                   self.model.predict_proba if self.model_type == 'classification' else self.model.predict, 
                                                   num_features=num_features, 
                                                   num_samples=num_samples)
        exp.show_in_notebook(show_table=True)
        return exp
    
    def model_check(self, model):
        if hasattr(model, 'best_estimator_'):
            return model.best_estimator_
        else:
            return model

def shap_lime(cfg, X_train, X_test, y_train=None, y_test=None, m=None, **kwargs):
    plts = []
    
    if m is None:
        try:
            with open('model.pkl', 'rb') as f:
                m = pickle.load(f)
        except FileNotFoundError:
            print("Model file not found.")
            return plts
        except pickle.UnpicklingError:
            print("Error loading the pickle model.")
            return plts
    
    interpreter = Interpretability(m, cfg['task_type'], X_train, X_test, y_train, y_test)
    
    if kwargs:
        for method_name, method_params in kwargs.items():
            if hasattr(interpreter, method_name):
                method = getattr(interpreter, method_name)
                
                plts.append(method(**method_params))
    
    return plts

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# def create_sample_data():
#     """Create sample data for demonstration."""
#     data = pd.DataFrame({
#         'color': ['red', 'blue', 'green', 'red', 'blue'] * 20,
#         'size': ['small', 'medium', 'large', 'small', 'large'] * 20,
#         'shape': ['circle', 'square', 'triangle', 'circle', 'square'] * 20
#     })
#     return data

# def encode_and_get_feature_names(data):
#     """Encode the data using OneHotEncoder and get feature names."""
#     encoder = OneHotEncoder(sparse=False)
#     encoded_data = encoder.fit_transform(data)
#     feature_names = encoder.get_feature_names_out()
#     return encoded_data, feature_names

# def aggregate_shap_values(shap_values, feature_names):
#     """
#     Aggregate SHAP values from one-hot encoded features back to original feature names.
    
#     :param shap_values: numpy array of shape (n_samples, n_features, n_classes)
#     :param feature_names: array of one-hot encoded feature names from OneHotEncoder
#     :return: pandas DataFrame with aggregated SHAP values
#     """
#     # Extract original feature names
#     original_feature_names = np.unique([name.split('_')[0] for name in feature_names])
    
#     # Create a mapping from original feature names to their one-hot encoded columns
#     feature_mapping = {name: [col for col in feature_names if col.startswith(name)] 
#                        for name in original_feature_names}
    
#     # Initialize a dictionary to store aggregated SHAP values
#     aggregated_shap = {name: np.zeros((shap_values.shape[0], shap_values.shape[2])) 
#                        for name in original_feature_names}
    
#     # Aggregate SHAP values
#     for orig_name, encoded_cols in feature_mapping.items():
#         for col in encoded_cols:
#             col_index = np.where(feature_names == col)[0][0]
#             aggregated_shap[orig_name] += shap_values[:, col_index, :]
    
#     # Convert to DataFrame
#     result = pd.DataFrame(aggregated_shap)
    
#     return result

# # Example usage
# np.random.seed(42)  # for reproducibility

# # Create sample data
# data = create_sample_data()

# # Encode data and get feature names
# encoded_data, feature_names = encode_and_get_feature_names(data)

# # Create mock SHAP values (in practice, these would come from your model)
# n_samples, n_features, n_classes = encoded_data.shape[0], encoded_data.shape[1], 3
# shap_values = np.random.randn(n_samples, n_features, n_classes)

# # Aggregate SHAP values
# aggregated_shap_df = aggregate_shap_values(shap_values, feature_names)

# print("Original feature names:", data.columns.tolist())
# print("\nOne-hot encoded feature names:", feature_names.tolist())
# print("\nAggregated SHAP values:")
# print(aggregated_shap_df.head())