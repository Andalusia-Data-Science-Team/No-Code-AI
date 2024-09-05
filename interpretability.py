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

    def _compute_shap_values(self):
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
    
    plts.append(interpreter.plot_variable_importance())

    if kwargs:
        for method_name, method_params in kwargs.items():
            if hasattr(interpreter, method_name):
                method = getattr(interpreter, method_name)
                
                plts.append(method(**method_params))
    
    return plts