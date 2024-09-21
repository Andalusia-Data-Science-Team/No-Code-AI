import shap

import lime
import lime.lime_tabular
import pandas as pd, numpy as np
from utils import my_waterfall

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
import logging

import plotly.express as px
import plotly.graph_objects as go

import matplotlib
matplotlib.use('Agg')
# np.set_printoptions(precision=7, suppress=False)

class Interpretability:
    def __init__(self, model, model_type, X_train, X_test, y_train=None, y_test=None, apply_prior= False):

        logging.basicConfig(filename='Xplain.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.apply_prior= apply_prior
        _model = model.pipeline.named_steps['model']
        self.processor= model.pipeline.named_steps['preprocessor']
        self.model_type = model_type
        self.original_cols= X_train.columns.tolist()
        self.X_train = self.processor.transform(X_train)
        self.X_test = self.processor.transform(X_test)
        self.base_data= X_test
        self.cats= self.processor.named_transformers_['categorical'].get_feature_names_out().tolist()
        self.nums= self.processor.named_transformers_['numerical'].get_feature_names_out().tolist()
        self.all_feature_names= self.nums + self.cats
        self.ohe_feature_names= self.processor.named_transformers_['categorical']['one_hot_encoder'].get_feature_names_out()
        self.y_train = y_train
        self.y_test = y_test
        self.explainer = None
        self.shap_values = None
        self.lime_explainer = None
        self.model= self.model_check(_model)
        self._compute_shap_values()
        self.num_cls= self.shap_values.shape[2]

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
            if self.apply_prior:
                # First check if you can fully replace explainer
                self.explainer = shap.TreeExplainer(self.model, self.X_train.toarray(), feature_names= self.all_feature_names)
            else:
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


        if self.apply_prior:
            self.shap_values = self.explainer.shap_values(self.X_test.toarray())
        else:
            self.shap_values = self.explainer.shap_values(self.X_test)

        explainer = shap.Explainer(self.model)
        self.shap_values_explainer = explainer(self.X_test)
        
        self.base_value= self.explainer.expected_value
            # Changed in version 0.45.0: Return type for models with multiple outputs changed from list to np.ndarray.
        if isinstance(self.shap_values, list):
            self.shap_values= self.shap_values[0]
            self.shap_values_explainer= self.shap_values_explainer[0]
        else:
            self.shap_values = self.shap_values
            self.shap_values_explainer = self.shap_values_explainer

    def plot_variable_importance(self):
        shap.summary_plot(self.shap_values, self.X_test, feature_names= self.all_feature_names)
        return plt
    
    def plot_shap_summary(self, summary_type= 'Aggregate', top_k= 10):
        # if classification:
        assert summary_type in ('Aggregate', 'Detailed'), f'the summary type {summary_type} is not supported!'

        self.logger.info(f"Shap Value for TreeExplainer is \n{self.shap_values}")
        self.logger.info(f"Shap Value for Explainer is \n{self.shap_values_explainer.values}")
        # return shap_values, shap_values_explainer, wrong_shap_values, self.all_feature_names, self.original_cols

        shap_values_df= self.process_ohe(self.shap_values_explainer.values, self.all_feature_names, self.original_cols)

        num_classes = self.shap_values.shape[2]

        if summary_type == 'Aggregate':
            shap_values_df= self.agg_dataframes(shap_values_df, num_classes)
            shap_values= self.plot_preprocessing(shap_values_df, agg= True, num_cls= num_classes)
        else:
            shap_values= self.plot_preprocessing(shap_values_df, num_cls= num_classes)
        
        tmp = {
            f'class {class_idx}': np.mean(np.abs(self.shap_values[:, :, class_idx]), axis=0)
            for class_idx in range(num_classes)
        }
        
        importance_df = pd.DataFrame(tmp)
        importance_df['Aggregate']= importance_df.sum(axis= 1)
        # feature_names should set the index
        importance_df.index = self.original_cols

        if summary_type == 'Aggregate':
            cls_imp= importance_df['Aggregate'].sort_values(ascending= False)[:top_k]
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
            # excluding the Aggregate 
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
    def contribution_plot(self, idx, sort= 'high-to-low'):
        figs = []
        shap_values_df= self.process_ohe(self.shap_values, self.all_feature_names, self.original_cols)
        shap_values= self.plot_preprocessing(shap_values_df, num_cls= self.num_cls)

        for class_index in range(self.num_cls):
            fig = self._contribution_plot(shap_values, self.original_cols, self.base_value, idx, sort, class_index)
            figs.append(fig)

        return figs

    def _contribution_plot(self, shap_values, feature_names, base_value, instance_index, sort_order='high-to-low', class_index=1):

        # Handle multi-dimensional SHAP values
        instance_shap = shap_values[class_index][instance_index]
        mean_shap = np.mean(shap_values[class_index], axis=0)
        if isinstance(base_value, np.ndarray):
            base_value = base_value[class_index]
        
        # Calculate the difference between instance and mean SHAP values
        diff_shap = instance_shap - mean_shap
        
        # Sort the features based on the specified order
        if sort_order == 'high-to-low':
            sorted_idx = np.argsort(diff_shap)[::-1]
        elif sort_order == 'low-to-high':
            sorted_idx = np.argsort(diff_shap)
        else:  # 'absolute'
            sorted_idx = np.argsort(np.abs(diff_shap))[::-1]
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = diff_shap[sorted_idx]
        
        # Calculate final prediction value
        final_value = base_value + np.sum(instance_shap)
        mean_prediction = base_value + np.sum(mean_shap)
        
        # Create a DataFrame for the waterfall chart
        df = pd.DataFrame({
            'Feature': sorted_features + ['Final Prediction'],
            'Value': np.concatenate([sorted_values, [final_value - mean_prediction]]),
            'Text': [f'{v:.4f}' for v in sorted_values] + [f'{final_value - mean_prediction:.4f}']
        })
        
        df['Measure'] = ['relative'] * len(sorted_features) + ['total']
        
        # Create the waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Contribution", orientation="v",
            measure=df['Measure'],
            x=df['Feature'],
            textposition="outside",
            text=df['Text'],
            y=df['Value'],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            decreasing={"marker":{"color":"#FF0000"}},
            increasing={"marker":{"color":"#00FF00"}},
            totals={"marker":{"color":"#0000FF"}},
        ))
        
        # Update layout
        fig.update_layout(
            title=f"SHAP value differences for instance {instance_index}, class {class_index} vs. dataset mean<br>(Final prediction: {final_value:.4f}, Mean prediction: {mean_prediction:.4f})",
            showlegend=False,
            xaxis_title="",
            yaxis_title="SHAP value difference",
            xaxis={'categoryorder':'total ascending'},
            waterfallgap=0.2,
        )
        
        # Add a base line at zero (representing the mean prediction)
        fig.add_shape(type="line",
            x0=-0.5, y0=0, x1=len(sorted_features)-0.5, y1=0,
            line=dict(color="red", width=2, dash="dot")
        )
        
        return fig
    
    def process_explainer_values(self, explainer_values, feature_names, original_feature_names, base_data):
        # https://github.com/shap/shap/issues/1252
        # For a classifier that gives us the shap values it's recommended to check the positive class,  not the negatives
        sv_shape= explainer_values.shape 
        shap_values_base= explainer_values.base_values   
        shap_values_display_data= explainer_values.display_data  
                                                            
        values= explainer_values.values  
        lower_bounds = getattr(explainer_values, "lower_bounds", None)   
                                                                    
                                                                    
        upper_bounds = getattr(explainer_values, "upper_bounds", None)

        aggregated_shap = {}
        feature_names= np.array(feature_names)
        for name in original_feature_names:
            if name in feature_names:  # numerical feature or non OHE feature (dont know if any exist ¯\_(ツ)_/¯)
                idx = np.where(feature_names == name)[0][0]
                aggregated_shap[f'{name}_shape_values'] = values[:, idx, :]
                if shap_values_display_data is not None:
                    raise NotImplementedError("can't be used")

            else:  # categorical feature
                encoded_features = [f for f in feature_names if f.startswith(f"{name}_")]
                # aggregation is done over the same class, as the shap value is always "a function of the number of outputs"
                aggregated_shap[f'{name}_shape_values'] = np.sum([values[:, np.where(feature_names == f)[0][0], :] 
                                                for f in encoded_features], axis=0)

                if shap_values_display_data is not None:
                    raise NotImplementedError("can't be used")
                    
        data_flattened = {key: pd.DataFrame(value, columns=[f'{key}_class_{i}' for i in range(value.shape[1])]) for key, value in aggregated_shap.items()}
        agg_df= pd.concat(data_flattened.values(), axis=1)
        agg_df= agg_df.set_index(base_data.index)
        _df= pd.concat([agg_df, base_data], axis=1)
        assert len(_df) == len(self.base_data), "Miss dimension between the shap and base data."
        return agg_df, sv_shape, shap_values_base, lower_bounds, upper_bounds

    def plot_contribution(self, idx, agg= True):
        # aggregation and postprocessing have conflict; as waterfall requires Eplainer object
        # shap_values_df= self.process_ohe(self.shap_values_explainer.values, self.all_feature_names, self.original_cols)
        # print(getattr(self.shap_values_explainer, "lower_bounds", None))
        # print(getattr(self.shap_values_explainer, "upper_bounds", None))
        # shap.waterfall_plot(self.shap_values_explainer[:, :, 0][idx], show= False)
        # return plt

        # takes the data, slice based on idx and class, then pass it to the
        # return self.shap_values_explainer
        shap_values_df, sv_shape, shap_values_base, lower_bounds, upper_bounds= self.process_explainer_values(self.shap_values_explainer, 
                                                                                                  self.all_feature_names, 
                                                                                                  self.original_cols, 
                                                                                                  self.base_data)
        
        sv_shape= sv_shape[1]
        if not agg:
            plts= []
            for i in range(self.num_cls):
                shap_values_base= shap_values_base[idx][i]
                shap_values= self.plot_preprocessing(shap_values_df, num_cls= self.num_cls)[:,:,i][idx]
            
                plts.append(my_waterfall(shap_values, sv_shape, shap_values_base, None, shap_values_df, self.original_cols, 
                                lower_bounds= lower_bounds, upper_bounds= upper_bounds))
                
            
            return plts
        else:
            shap_values_base= np.sum(shap_values_base[idx])/len(shap_values_base[idx])
            _shap_values_df= self.agg_dataframes(shap_values_df, self.num_cls)
            shap_values= self.plot_preprocessing(shap_values_df, num_cls= self.num_cls)[idx]
            return shap_values_df, _shap_values_df, self.shap_values_explainer, self.all_feature_names, self.original_cols
            df_idx= self.base_data.iloc[idx]
            return my_waterfall(shap_values, sv_shape, shap_values_base, None, df_idx, self.original_cols, 
                                lower_bounds= lower_bounds, upper_bounds= upper_bounds)


        
    def plot_preprocessing(self, shap_values_df, num_cls= None, agg= False):
        arrays_by_class = []

        for class_num in range(num_cls):
            cols_for_class = [col for col in shap_values_df.columns if f'_class_{class_num}' in col]
            class_df = shap_values_df[cols_for_class]
            class_array = class_df.to_numpy()
            arrays_by_class.append(class_array)

        result_array = np.stack(arrays_by_class, axis=-1)
        return result_array

    def aggregate_class_columns(self, df):

        features= df.columns.tolist()
        aggregated_df = pd.DataFrame()

        for feature in features:
            matching_columns = [col for col in df.columns if col.startswith(f'{feature}_class_')]
            
            aggregated_df[f'{feature}_aggregated'] = df[matching_columns].sum(axis=1)
        return aggregated_df

    def agg_dataframes(self, df, class_nums):
        dfs_by_class = {}
        for class_num in range(class_nums):
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

def shap_lime(cfg, X_train, X_test, y_train=None, y_test=None, m=None, apply_prior= False, **kwargs):
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
    
    interpreter = Interpretability(m, cfg['task_type'], X_train, X_test, y_train, y_test, apply_prior= apply_prior)
    
    if kwargs:
        for method_name, method_params in kwargs.items():
            if hasattr(interpreter, method_name):
                method = getattr(interpreter, method_name)
                
                plts.append(method(**method_params))
    
    return plts
