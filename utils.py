import pandas as pd, numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


def missing(_df, clean_method= 'Remove Missing Data'):
    df= _df.copy()
    df.drop_duplicates(inplace= True)

    if clean_method == 'Remove Missing Data':
        df.dropna(inplace= True)
        return df
    
    elif clean_method == 'Impute Missing Data':
        numeric_features = df.select_dtypes(include=np.number).columns
        numeric_imputer = SimpleImputer(strategy='mean')
        df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])


        categorical_features = df.select_dtypes(include=object).columns
        if len(categorical_features) != 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
        return df
    
    else:
        raise ValueError('Invalied input for imputing')
    

def IQR(_df, lower_bound=0.25, upper_bound=0.75, multiplier=1.5):
    df= _df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]
    sub_df= df[numeric_cols]

    q1 = sub_df[numeric_cols].quantile(lower_bound)
    q3 = sub_df[numeric_cols].quantile(upper_bound)
    iqr = q3 - q1
    sub_df = sub_df[~((sub_df < (q1 - multiplier * iqr)) |(sub_df > (q3 + multiplier * iqr))).any(axis=1)]
    df= df.loc[sub_df.index]
    return df

def IF(_df):
    isolation_forest= IsolationForest()
    df= _df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]
    
    num_df= df[numeric_cols]
    outlier_pred = isolation_forest.fit_predict(num_df)

    clean_features = df[outlier_pred == 1]
    return clean_features

def remove_outliers(_df, method= "Don't Remove Outliers"):
    if method == 'Use IQR':
        return IQR(_df)
    elif method == 'Use Isolation Forest':
        return IF(_df)
    else:
        return _df
    
def handle(_df, trg, cls= 'Classification'):
    X= _df.drop([trg], axis= 1)
    y= _df[trg]

    if cls == 'Classification':
        X_train, X_test, y_train, y_test= train_test_split(X, y, stratify= y)
    else:
        X_train, X_test, y_train, y_test= train_test_split(X, y)

    return X_train, X_test, y_train, y_test

