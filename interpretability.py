import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Interpretability:
    def __init__(self, model, model_type, X_train, X_test, y_train=None, y_test=None):

        self.model = model
        self.model_type = model_type
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.explainer = None
        self.shap_values = None
        self.lime_explainer = None

    def compute_shap_values(self):
        """Compute SHAP values for the model."""
        if self.model_type == 'classification':
            self.explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'tree_') else shap.KernelExplainer(self.model.predict_proba, self.X_train)
        else:
            self.explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'tree_') else shap.KernelExplainer(self.model.predict, self.X_train)

        self.shap_values = self.explainer.shap_values(self.X_test)
        
    def plot_variable_importance(self):
        """Plot SHAP summary plot (variable importance)."""
        shap.summary_plot(self.shap_values, self.X_test)

    def plot_summary_label(self, label_index):
        """Plot SHAP summary plot for a specific label (for classification)."""
        if self.model_type == 'classification' and self.shap_values is not None:
            shap.summary_plot(self.shap_values[label_index], self.X_test)

    def plot_dependence(self, feature_name):
        """Plot SHAP dependence plot for a specific feature."""
        if self.shap_values is not None:
            shap.dependence_plot(feature_name, self.shap_values, self.X_test)

    def explain_with_lime(self, num_features=5, num_samples=1000):
        """Explain a prediction using LIME."""
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values, 
            feature_names=self.X_train.columns, 
            class_names=np.unique(self.y_train) if self.model_type == 'classification' else None,
            discretize_continuous=True
        )
        i = np.random.randint(0, self.X_test.shape[0])  # Randomly select a test instance
        exp = self.lime_explainer.explain_instance(self.X_test.iloc[i].values, 
                                                   self.model.predict_proba if self.model_type == 'classification' else self.model.predict, 
                                                   num_features=num_features, 
                                                   num_samples=num_samples)
        exp.show_in_notebook(show_table=True)
        return exp
