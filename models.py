import pickle
import pandas as pd, numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class Model:
    def __init__(self, algorithm):
        self.pipeline = None
        self.model = None
       
        self.label_encoder = LabelEncoder()
        if algorithm == 'SVC':
            self.algorithm= LinearSVC()
        elif algorithm == 'LR':
            self.algorithm= LogisticRegression()

    def build_pipeline(self, X):
        categorical_features= X.select_dtypes('object').columns.tolist()
        numerical_features= X.select_dtypes('number').columns.tolist() 
        categorical_transformer = Pipeline(steps=[
            ('one_hot_encoder', OneHotEncoder())
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('categorical', categorical_transformer, categorical_features),
            ('numerical', numerical_transformer, numerical_features)
        ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.algorithm)
        ])

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.build_pipeline(X)
        # print(self.pipeline)
        if y.dtypes == 'object':
            y = self.label_encoder.fit_transform(y)

        self.pipeline.fit(X, y)

        self.model = self.pipeline.named_steps['model']

    def preprocess(self, X):
        return self.pipeline.named_steps['preprocessor'].transform(X)

    def predict(self, X):
        X= self.preprocess(X)
        return self.model.predict(X)
 
    def print_accuracy_score(self, X, y_true):
        y_pred = self.predict(X)
        if y_true.dtypes == 'object':
            y_true = self.label_encoder.transform(y_true)
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"Accuracy Score: {accuracy:.4f}")
        return accuracy

    def print_confusion_matrix(self, X, y_true):
        y_pred = self.predict(X)
        if y_true.dtypes == 'object':
            y_pred = self.label_encoder.inverse_transform(y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return cm

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved successfully as: {file_path}.pkl")



def model(X_train, X_test, y_train, y_test, alg, save=False):
    _model= Model(alg)
    _model.train(X_train, y_train)
    if save:
        _model.save_model('model.pkl')
    acc= _model.print_accuracy_score(X_test, y_test)
    cm= _model.print_confusion_matrix(X_test, y_test)

    return acc, cm
