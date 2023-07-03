import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class Model:
    def __init__(self, df, algorithm):
        self.pipeline = None
        self.model = None 
        self.label_encoder = LabelEncoder()
        self.categorical_features= df.select_dtypes('object').columns.tolist()
        self.numerical_features= df.select_dtypes('number').columns.tolist() 
        if algorithm == 'SVC':
            self.algorithm= LinearSVC()
        elif algorithm == 'LR':
            self.algorithm= LogisticRegression()

    def build_pipeline(self):
        categorical_transformer = Pipeline(steps=[
            ('one_hot_encoder', OneHotEncoder())
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('categorical', categorical_transformer, self.categorical_features),
            ('numerical', numerical_transformer, self.numerical_features)
        ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.algorithm)
        ])

    def train_model(self, X, y):
        self.build_pipeline()

        if y.dtypes == 'object':
            # Apply label encoding to the target column
            y = self.label_encoder.fit_transform(y)

        self.pipeline.fit(X, y)

        # Get the trained model
        self.model = self.pipeline.named_steps['model']

    def predict(self, X):
        return self.model.predict(X)
 
    def print_accuracy_score(self, X, y_true):
        y_pred = self.predict(X)
        if self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy Score: {accuracy:.4f}")

    def print_confusion_matrix(self, X, y_true):
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        print("Confusion Matrix:")
        print(cm)

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved successfully as: {file_path}")


