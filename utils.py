import pandas as pd, numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


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
        X_train, X_test, y_train, y_test= train_test_split(X, y, shuffle= False)

    return X_train, X_test, y_train, y_test


def HeatMap(_df):
    return _df.select_dtypes('number').corr()

def corr_plot(_df):
    corr_matrix= _df.select_dtypes('number').corr()
    tril_index= np.tril_indices_from(corr_matrix)
    for coord in zip(*tril_index):
        corr_matrix.iloc[coord[0], coord[1]] = np.nan

    corr_values = (corr_matrix
                .stack()
                .to_frame()
                .reset_index()
                .rename(columns={'level_0':'feature1',
                                    'level_1':'feature2',
                                    0:'correlation'}))
    corr_values['abs_correlation'] = corr_values.correlation.abs()
    return corr_values

def inf_proc(item):
    try:
        fixed_item = float(item)
        return fixed_item
    except:
        return item


class SkewnessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, skew_limit=0.8, forced_fix= False):
        self.skew_limit = skew_limit
        self.forced_fix = forced_fix
        self.method_dict = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X= X.to_numpy()
        self.method_dict = self.extracrt_recommeneded_features(X)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X= X.to_numpy()
        X_transformed = X.copy()
        for method, features in self.method_dict.items():

            if method == 'log':
                # Apply log transformation to the specified features
                X_transformed[:, features] = np.log1p(X_transformed[:, features])
            elif method == 'sqrt':
                # Apply square root transformation to the specified features
                X_transformed[:, features] = np.sqrt(X_transformed[:, features])
            elif method == 'boxcox':
                # Apply Box-Cox transformation to the specified features
                for feature in features:
                    X_transformed[:, feature], _ = stats.boxcox(X_transformed[:, feature])
            elif method == 'yeojohnson':
                for feature in features:
                    X_transformed[:, feature], _ = stats.yeojohnson(X_transformed[:, feature])
            elif method == 'cube':
                # Apply Cube transformation to the specified features
                X_transformed[:, features] = np.cbrt(X_transformed[:, features])

        return X_transformed

    def extracrt_recommeneded_features(self, X):
        skew_vals = np.abs(stats.skew(X, axis=0))
        skew_col_indices = np.where(skew_vals > self.skew_limit)[0]
        method_dict = {}

        for feature_idx in skew_col_indices:
            feature = X[:, feature_idx]

            method = self.recommend_skewness_reduction_method(feature, self.forced_fix)
            if method not in method_dict:
                method_dict[method] = [feature_idx]
            else:
                method_dict[method].append(feature_idx)

        print(method_dict)
        return method_dict

    def recommend_skewness_reduction_method(self, feature: pd.Series, forced_fix= False) -> str:

        skewness_dict = {}
        all= {}

        transformed_log = np.log1p(feature)
        _, p_value = stats.normaltest(transformed_log)

        # The p-value is a measure of the evidence against the null hypothesis of normality. 
        # A low p-value (typically less than 0.05) suggests that the data is significantly different from a normal distribution, 
        # indicating that the fix for skewness was not successful in achieving normality.
        if p_value > 0.05:
            skewness_dict['log'] = p_value
        else:
            all['log']= p_value

        transformed_sqrt = np.sqrt(feature)
        _, p_value = stats.normaltest(transformed_sqrt)
        if p_value > 0.05:
            skewness_dict['sqrt'] = p_value
        else:
            all['sqrt']= p_value

        if (feature < 0).any() or (feature == 0).any():
            transformed_yeojohnson, _ = stats.yeojohnson(feature)
            _, p_value = stats.normaltest(transformed_yeojohnson)
            if p_value > 0.05:
                skewness_dict['yeojohnson'] = p_value
            else:
                all['yeojohnson']= p_value

        else:
            transformed_boxcox, _ = stats.boxcox(feature + 0.0001)
            _, p_value = stats.normaltest(transformed_boxcox)
            if p_value > 0.05:
                skewness_dict['boxcox'] = p_value
            else:
                all['boxcox']= p_value

        transformed_cbrt = np.cbrt(feature)
        _, p_value = stats.normaltest(transformed_cbrt)
        if p_value > 0.05:
            skewness_dict['cube'] = p_value
        else:
            all['cube']= p_value

        if len(skewness_dict) > 0:
            return max(skewness_dict, key=lambda y: abs(skewness_dict[y]))
        else:
            if forced_fix:
                print('No Fix, using best transformers')
                return max(all, key=lambda y: abs(all[y]))
            else:
                return 'No Fix'

        
