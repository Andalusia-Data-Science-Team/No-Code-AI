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
        self.original_cols= X_train.columns.tolist()
        self.X_train = self.processor.transform(X_train)
        self.X_test = self.processor.transform(X_test)
        self.cats= self.processor.named_transformers_['categorical'].get_feature_names_out().tolist()
        self.nums= self.processor.named_transformers_['numerical'].get_feature_names_out().tolist()
        self.all_feature_names= self.nums + self.cats
        self.ohe_feature_names= self.processor.named_transformers_['categorical']['one_hot_encoder'].get_feature_names_out()
        # self.rest_of_cat_features= list(set(self.cats) - set(self.ohe_feature_names.tolist()))
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
            self.explainer = shap.TreeExplainer(self.model, feature_names= self.all_feature_names)

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
        shap.summary_plot(self.shap_values, self.X_test, feature_names= self.all_feature_names)
        return plt
    
    def plot_shap_summary(self, summary_type= 'agg', top_k= 10):
        # if classification:
        assert summary_type in ('agg', 'detailed'), f'the summary type {summary_type} is not supported!'
        # Changed in version 0.45.0: Return type for models with multiple outputs changed from list to np.ndarray.
        if isinstance(self.shap_values, list):
            shap_values= self.shap_values[0]
        else:
            shap_values = self.shap_values

        shap_values_df= self.process_ohe(shap_values, self.all_feature_names, self.original_cols)
        num_classes = shap_values.shape[2]
        if summary_type == 'agg':
            shap_values_df= self.agg_dataframes(shap_values_df, num_classes)
            shap_values= self.plot_preprocessing(shap_values_df, agg= True, num_cls= num_classes)
        else:
            shap_values= self.plot_preprocessing(shap_values_df, num_cls= num_classes)
        
        tmp = {
            f'class {class_idx}': np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
            for class_idx in range(num_classes)
        }
        importance_df = pd.DataFrame(tmp)
        importance_df['agg']= importance_df.sum(axis= 1)
        # feature_names should set the index
        importance_df.index = self.original_cols

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
            shap.summary_plot(self.shap_values[label_index], self.X_test, feature_names= self.all_feature_names)

    def plot_dependence(self, feature_name):
        if self.shap_values is not None:
            shap.dependence_plot(feature_name, self.shap_values, self.X_test)
            return plt
        
    def plot_preprocessing(self, shap_values_df, num_cls= None, agg= False):
        arrays_by_class = []

        for class_num in range(num_cls):
            cols_for_class = [col for col in shap_values_df.columns if f'_class_{class_num}' in col]
            class_df = shap_values_df[cols_for_class]
            class_array = class_df.to_numpy()
            arrays_by_class.append(class_array)

        result_array = np.stack(arrays_by_class, axis=-1)
        return result_array


    def agg_dataframes(self, df, class_nums):
        dfs_by_class = {}
        for class_num in range(class_nums):  # Since we know there are 3 classes (1, 2, 3)
            cols_for_class = [col for col in df.columns if f'_class_{class_num}' in col]
            dfs_by_class[f'class_{class_num}'] = df[cols_for_class]

        dfs = list(dfs_by_class.values())
        result = pd.DataFrame(index=dfs[0].index, columns=dfs[0].columns)
        for df in dfs:
            result = result.add(df, fill_value=0)

        return result

    def process_ohe(self, shap_values, feature_names, original_feature_names):
        aggregated_shap = {}
        feature_names= np.array(feature_names)
        for name in original_feature_names:
            if name in feature_names:  # numerical feature or non OHE feature (dont know if any exist ¯\_(ツ)_/¯)
                idx = np.where(feature_names == name)[0][0]
                aggregated_shap[name] = shap_values[:, idx, :]
            else:  # categorical feature
                encoded_features = [f for f in feature_names if f.startswith(f"{name}_")]
                # aggregation is done over the same class, as the shap value is always "a function of the number of outputs"
                aggregated_shap[name] = np.sum([shap_values[:, np.where(feature_names == f)[0][0], :] 
                                                for f in encoded_features], axis=0)
        data_flattened = {key: pd.DataFrame(value, columns=[f'{key}_class_{i}' for i in range(value.shape[1])]) for key, value in aggregated_shap.items()}
        return pd.concat(data_flattened.values(), axis=1)

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
