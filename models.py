from utils import SkewnessTransformer
from model_utils import grid_dict

import pickle
import pandas as pd, numpy as np

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from interpretability import Interpretability

import matplotlib
matplotlib.use('Agg')

models_dict= {'SVC': SVC(), 'LR': LogisticRegression(), 'Linear Regression': LinearRegression(),
              'ElasticNet': ElasticNet(), 'KNN_cls': KNeighborsClassifier(), 'KNN_reg': KNeighborsRegressor(),
              'RF_cls': RandomForestClassifier(), 'XGB_cls': XGBClassifier(), 'XGB_reg': XGBRegressor(),
              'Ridge': Ridge(), 'Lasso': Lasso(), 'extra_tree': ExtraTreesClassifier(), 'SVR': SVR(),
              'GradientBoosting_cls': GradientBoostingClassifier(), 'Adaboost': AdaBoostClassifier(),
              'DecisionTree_cls': DecisionTreeClassifier()}

normilizers= {'standard': StandardScaler(), 'min-max': MinMaxScaler(feature_range = (0, 1))}


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
        self.cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        self.scores_ = []
        self.best_estimators_ = []

        for train_index, val_index in tqdm(self.cv.split(X), total=self.n_splits, desc="K-Fold Progress"):
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
        print(self.grid_params)
        self.grid_search = GridSearchCV(estimator=self.alg, param_grid=self.grid_params, cv=3)
        self.grid_search.fit(X, y)
        print('grid search applied')

        self.best_estimator_ = self.grid_search.best_estimator_
        print(self.best_estimator_, type(self.best_estimator_))

        return self
    
    def transform(self, X):
        return X
    
    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


class Model:
    def __init__(self, algorithm, grid=False):
        self.pipeline = None
        self.model = None
        self.label_encoder = LabelEncoder()

        if algorithm in models_dict.keys():
            self.algorithm = models_dict[algorithm]
        else:
            raise NotImplementedError
        
        self.grid = grid

        if grid:
            self.grid_model = GridSearchModel(alg= self.algorithm, grid_params= grid_dict[algorithm])

    def build_pipeline(self, X, poly_feat=False, skew_fix=False):
        categorical_features = X.select_dtypes('object').columns.tolist()
        numerical_features = X.select_dtypes('number').columns.tolist() 

        categorical_transformer = Pipeline(steps=[
            ('cat_imp', SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ])

        if skew_fix:
            numerical_transformer.steps.append(('skew_fix', SkewnessTransformer(skew_limit=0.75))),
            numerical_transformer.steps.append(('num_imp', SimpleImputer(missing_values=np.nan, strategy="mean")))

        if poly_feat:
            numerical_transformer.steps.append(('Polynomial_Features', PolynomialFeatures(degree=2)))
            print('poly features applied')

        preprocessor = ColumnTransformer(transformers=[
            ('categorical', categorical_transformer, categorical_features),
            ('numerical', numerical_transformer, numerical_features)
        ])

        if self.grid:
            model_step = ('model', self.grid_model)
            
        else:
            model_step = ('model', self.algorithm)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            model_step
        ])

    def train(self, X: pd.DataFrame, y: pd.Series, skew, poly):
        self.build_pipeline(X, skew_fix=skew, poly_feat=poly)
        if y.dtypes == 'object':
            y = self.label_encoder.fit_transform(y)


        self.pipeline.fit(X, y)
        self.model = self.pipeline.named_steps['model']


    def preprocess(self, X):
        return self.pipeline.named_steps['preprocessor'].transform(X)

    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)

    def cls_metrics(self, X, y_true):
        y_pred = self.predict(X)
        if y_true.dtypes == 'object':
            y_pred = self.label_encoder.inverse_transform(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        return cm, accuracy
    
    def reg_metrics(self, X, y_true):
        y_pred = self.predict(X)

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mse, r2

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved successfully as: {file_path}")



def model(X_train, X_test, y_train, y_test, cfg):
    _model= Model(cfg['alg'], cfg['apply_GridSearch'])
    _model.train(X_train, y_train, cfg['skew_fix'], cfg['poly_feat'])
    if cfg['save']:
        _model.save_model('model.pkl')

    if cfg['task_type'] == "Classification":
        cm, acc= _model.cls_metrics(X_test, y_test)
        return acc, cm

    elif cfg['task_type'] == "Regression":
        mse, r2= _model.reg_metrics(X_test, y_test)
        return mse, r2
    
    else:
        raise ValueError('invalid Task')

def shap_lime(cfg, X_train, X_test, y_train=None, y_test=None, m= None):
    if m is not None:
            interpreter= Interpretability(m, cfg['task_type'], X_train, X_test, y_train, y_test)
            x= interpreter.compute_shap_values()
            fig= interpreter.plot_variable_importance()
            return x, fig
    else:
        try:
            with open('model.pkl', 'rb') as f:
                m= pickle.load(f)
        except FileNotFoundError:
            print("Model file not found.")

        except pickle.UnpicklingError:
            print("Error loading the pickle model.")

        interpreter= Interpretability(m, cfg['task_type'], X_train, X_test, y_train, y_test)
        x= interpreter.compute_shap_values()
        p= interpreter.plot_variable_importance()
        return x, p

def inference(X):
    try:
        with open('model.pkl', 'rb') as f:
            _model= pickle.load(f)

        pred= _model.predict(X)
        return pred
    except FileNotFoundError:
        print("Model file not found.")

    except pickle.UnpicklingError:
        print("Error loading the pickle model.")

