import shap

import lime
import lime.lime_tabular
import pandas as pd, numpy as np
from utils import my_waterfall, og_waterfall

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
        self.base_data= X_test.reset_index(drop= True)
        try:
            self.cats= self.processor.named_transformers_['categorical'].get_feature_names_out().tolist()
        except:
            self.cats= []
        try:
            self.nums= self.processor.named_transformers_['numerical'].get_feature_names_out().tolist()
        except:
            self.nums= []

        self.all_feature_names= self.nums + self.cats # all feature names would change the order?

        try:
            self.ohe_feature_names= self.processor.named_transformers_['categorical']['one_hot_encoder'].get_feature_names_out()
        except:
            self.ohe_feature_names= []
            
        self.y_train = y_train
        self.y_test = y_test
        self.explainer = None
        self.shap_values = None
        self.lime_explainer = None
        self.model= self.model_check(_model)
        self._compute_shap_values()
        if len(self.shap_values.shape) > 2:
            self.num_cls= self.shap_values.shape[2]
            self.dim3= True
        else:
            self.num_cls= 1 # set for binray classification and regression
            self.dim3= False


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
    
    def normalize_feature_importance_column_wise(self, df):
        df_abs = df.abs()
        col_sums = df_abs.sum(axis=0)
        df_normalized = df_abs.div(col_sums, axis=1)
        return df_normalized
    
    def plot_shap_summary(self, summary_type= 'Aggregate', top_k= 10, normalize= True):
        assert summary_type in ('Aggregate', 'Detailed'), f'the summary type {summary_type} is not supported!'
        self.logger.info(f"Shap Value for TreeExplainer is \n{self.shap_values}")
        self.logger.info(f"Shap Value for Explainer is \n{self.shap_values_explainer.values}")

        shap_values_df= self.process_ohe(self.shap_values_explainer.values, self.all_feature_names, self.original_cols, self.dim3)
        agg_shap_values_df= self.aggregate_features(shap_values_df, self.num_cls)
        df_class_dict= self.agg_dataframes(shap_values_df, self.num_cls)
        df_class_dict["Aggregate"]= agg_shap_values_df
        importance_df= self.plot_preprocessing(df_class_dict, normalize)
            
        # feature_names should set the index'
        _cols= ["Aggregate"] + [f"class_{i}" for i in range(self.num_cls)]
        importance_df.index = self.original_cols
        importance_df.columns = _cols
        
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
                cls_imp= importance_df[f'class_{i}'].sort_values(ascending= False)[:top_k]
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
        shap_values_df= self.process_ohe(self.shap_values, self.all_feature_names, self.original_cols, self.dim3)
        shap_values= self.plot_preprocessing(shap_values_df)

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
    
    def process_explainer_values(self, explainer_values, feature_names, original_feature_names, base_data, dim3= True):
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
                if dim3:
                    aggregated_shap[f'{name}_shape_values'] = values[:, idx, :]
                else:
                    aggregated_shap[f'{name}_shape_values'] = values[:, idx]

                if shap_values_display_data is not None:
                    raise NotImplementedError("can't be used")

            else:  # categorical feature
                encoded_features = [f for f in feature_names if f.startswith(f"{name}_")]
                # aggregation is done over the same class, as the shap value is always "a function of the number of outputs"
                if dim3:
                    aggregated_shap[f'{name}_shape_values'] = np.sum([values[:, np.where(feature_names == f)[0][0], :] 
                                                        for f in encoded_features], axis=0)
                else:
                    aggregated_shap[f'{name}_shape_values'] = np.sum([values[:, np.where(feature_names == f)[0][0]] 
                                                        for f in encoded_features], axis=0)

                if shap_values_display_data is not None:
                    raise NotImplementedError("can't be used")
                    
        if dim3:
            data_flattened = {key: pd.DataFrame(value, columns=[f'{key}_class_{i}' for i in range(value.shape[1])]) for key, value in aggregated_shap.items()}
            agg_df = pd.concat(data_flattened.values(), axis=1)
        else:
            agg_df = pd.DataFrame(aggregated_shap)
            agg_df.columns = [f'{col}_class_0' for col in agg_df.columns]
        print("agg_df shape: ", agg_df.shape)
        index= base_data.index
        agg_df= agg_df.set_index(index)
        _df= pd.concat([agg_df, base_data], axis=1)
        assert len(_df) == len(base_data), f"Miss dimension between the shap: {len(_df)} and base data: {len(base_data)}."
        return agg_df, sv_shape, shap_values_base, lower_bounds, upper_bounds

    def plot_contribution(self, data= None, idx= 0, agg= True, normalize= True):

        # return og_waterfall(self.shap_values_explainer[:,:,0][0])
        if data is None:
            shap_values_df, _, shap_values_base, lower_bounds, upper_bounds= self.process_explainer_values(self.shap_values_explainer, 
                                                                                                    self.all_feature_names, 
                                                                                                    self.original_cols, 
                                                                                                    self.base_data,
                                                                                                    self.dim3)
        else:
            inf_data= self.processor.transform(data)
            explainer = shap.Explainer(self.model)
            shap_values_explainer = explainer(inf_data)
            data= data.reset_index(drop= True)
            shap_values_df, _, shap_values_base, lower_bounds, upper_bounds= self.process_explainer_values(shap_values_explainer, 
                                                                                                    self.all_feature_names, 
                                                                                                    self.original_cols, 
                                                                                                    data,
                                                                                                    self.dim3)
                   
        if not agg:
            plts= []
            dict_dfs= self.agg_dataframes(shap_values_df, self.num_cls)
            for i in range(self.num_cls):
                cls_df= dict_dfs[f'class_{i}']
                if self.dim3:
                    proc_shap_values_base= shap_values_base[idx][i]
                else:
                    proc_shap_values_base= shap_values_base[idx]
                # shap_values= np.array(shap_values_df)[:,:,i][idx]
                # return self.shap_values[:,:,i]
                _shap_values= cls_df#.reset_index()
                proc_shap_values= np.array(_shap_values.iloc[idx])
                base_df= self.base_data#.reset_index()
                if data is not None:
                    proc_base_df= np.array(base_df.iloc[idx])
                else:
                    proc_base_df= np.array(base_df.iloc[idx])[1:]

                plts.append(my_waterfall(proc_shap_values, proc_shap_values_base, None, proc_base_df, self.original_cols, 
                                lower_bounds= lower_bounds, upper_bounds= upper_bounds))
                
            
            return plts
        else:
            # TODO
            shap_values_base= np.sum(shap_values_base[idx])/len(shap_values_base[idx])
            shap_values_df= self.aggregate_features(shap_values_df, self.num_cls)
            shap_values= self.plot_preprocessing(shap_values_df, normalize)[idx]
            df_idx= self.base_data.iloc[idx]
            return my_waterfall(shap_values, shap_values_base, None, df_idx, self.original_cols, 
                                lower_bounds= lower_bounds, upper_bounds= upper_bounds)


        
    def plot_preprocessing(self, d_dfs, normalized= True) -> pd.DataFrame:
        """
        must be with .abs() or it will show the negative impact between features
        """
        arrays_by_class = []

        for _, df in d_dfs.items():
            df= df.abs().sum(axis= 0)
            arr= np.array(df)
            if normalized:
                total_sum = np.sum(arr)
                normalized_arr = arr / total_sum
                arrays_by_class.append(normalized_arr)
            else:
                arrays_by_class.append(arr)

        return pd.DataFrame(arrays_by_class).T


    def agg_dataframes(self, df, class_nums):
        """
        Aggregates columns of a dataframe based on class number identifiers and 
        groups them into separate dataframes.

        This method takes a dataframe `df` and an integer `class_nums`, which 
        specifies the number of classes. It creates a dictionary where the keys 
        are class labels in the format `'class_{class_num}'` and the values are 
        dataframes containing columns from `df` that match the pattern 
        `'_class_{class_num}'` in their column names.
        """
        dfs_by_class = {}
        for class_num in range(class_nums):
            cols_for_class = [col for col in df.columns if f'_class_{class_num}' in col]
            dfs_by_class[f'class_{class_num}'] = df[cols_for_class]

        return dfs_by_class

    def aggregate_features(self, df, num_classes):
        """
        Aggregate the n-classes
        """
        class_suffixes = tuple(f'_class_{i}' for i in range(num_classes))
        features = set('_'.join(col.split('_')[:-2]) for col in df.columns if col.endswith(class_suffixes)) # that's redundunt using self.original would suffice.
        
        df_agg = pd.DataFrame()
        
        for feature in features:
            cols_to_agg = [col for col in df.columns if col.startswith(feature) and col.endswith(class_suffixes)]
            df_agg[f'{feature}_class_agg'] = df[cols_to_agg].sum(axis=1)
        
        return df_agg

    def process_ohe(self, shap_values, feature_names, original_feature_names, dim3= True):
        aggregated_shap = {}
        feature_names= np.array(feature_names)
        for name in original_feature_names:
            if name in feature_names:  # numerical feature or non OHE feature (dont know if any exist ¯\_(ツ)_/¯)
                idx = np.where(feature_names == name)[0][0]
                aggregated_shap[name] = shap_values[:, idx, :] if dim3 else shap_values[:, idx]
            else:  # categorical feature
                encoded_features = [f for f in feature_names if f.startswith(f"{name}_")]
                # aggregation is done over the same class, as the shap value is always "a function of the number of outputs" in case of dim3
                if dim3:
                    aggregated_shap[name] = np.sum([shap_values[:, np.where(feature_names == f)[0][0], :] 
                                                    for f in encoded_features], axis=0)
                else:
                    aggregated_shap[name] = np.sum([shap_values[:, np.where(feature_names == f)[0][0]] 
                                                    for f in encoded_features], axis=0)

        if dim3:
            data_flattened = {key: pd.DataFrame(value, columns=[f'{key}_class_{i}' for i in range(value.shape[1])]) for key, value in aggregated_shap.items()}
            return pd.concat(data_flattened.values(), axis=1)
        else:
            df= pd.DataFrame(aggregated_shap)
            df.columns = [f'{col}_class_0' for col in df.columns]
            return df
    
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
