import pandas as pd
import numpy as np
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm

from prophet import Prophet


class ProphetModel(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        date_col,
        target_col,
        freq,
        f_period,
        validation_size,
        prophet_params=None,
        selected_cols=None,
    ):

        self.date_col = date_col
        self.target_col = target_col
        self.prophet_params = prophet_params or {}
        self.selected_cols = selected_cols
        self.freq = freq
        self.f_period = f_period
        self.validation = validation_size

    @property
    def set_model(self):
        return "Prophet"

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col])

        self.display_df = df.copy()
        df = df.rename(columns={self.date_col: "ds", self.target_col: "y"})
        prophet_df = df.sort_values(by="ds").reset_index(drop=True)

        feature_df = self.add_features(prophet_df, self.selected_cols)
        feature_df = self.create_date_features(feature_df)

        train_size = math.ceil(len(df) * (1 - self.validation))
        train_df = feature_df.iloc[:train_size]
        self.test_df = feature_df.iloc[train_size:]

        self.m = Prophet(**self.prophet_params)

        feat_ls = [i for i in train_df.columns if i not in ["ds", "y"]]
        for f in feat_ls:
            self.m.add_regressor(f)

        self.m.fit(train_df)

        return self

    def transform(self, X=None):
        self.forecasts = self.m.predict(self.test_df)  # Make predictions on test_df
        self.calculate_errors()
        return self

    def add_features(self, prophet_df: pd.DataFrame, selected_columns):

        rows = []
        for _, row in tqdm(prophet_df.iterrows(), total=prophet_df.shape[0]):
            row_data = dict(
                day_of_week=row.ds.dayofweek,
                day_of_month=row.ds.day,
                week_of_year=row.ds.week,
                month=row.ds.month,
                y=row.y,
                ds=row.ds,
            )

            rows.append(row_data)
        _df = pd.DataFrame(rows)

        selected_columns = selected_columns or prophet_df.columns.tolist()

        selected_columns = [
            sel_col
            for sel_col in selected_columns
            if sel_col not in ["ds", "y", self.date_col, self.target_col]
        ]

        prophet_df = prophet_df[selected_columns]

        return pd.concat([prophet_df, _df], axis=1)

    def create_date_features(self, data):
        data["quarter"] = data["ds"].dt.quarter
        data["month"] = data["ds"].dt.month
        data["hour"] = data["ds"].dt.hour
        data["dayofmonth"] = data["ds"].dt.day
        data["daysinmonth"] = data["ds"].dt.daysinmonth
        data["dayofyear"] = data["ds"].dt.dayofyear

        return data

    def slide_display(self):
        self.display_df = self.display_df.sort_values(by=self.date_col).reset_index(
            drop=True
        )
        fig = px.line(
            self.display_df,
            x=self.date_col,
            y=self.target_col,
            title="Raw Time Series Data",
        )

        # slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=2, label="2m", step="month", stepmode="backward"),
                        dict(count=2, label="5m", step="month", stepmode="backward"),
                    ]
                )
            ),
        )

        return fig.update_layout(width=600, height=300)

    def prophet_plot_forecast(self):
        return self.m.plot(self.forecasts)

    def plot_test_with_actual(self):
        fig = go.Figure()
        self.display_df = self.display_df[[self.date_col, self.target_col]]

        # Actual values as a line
        fig.add_trace(
            go.Scatter(
                x=self.display_df[self.date_col],
                y=self.display_df[self.target_col],
                mode="lines",
                name="Actual",
                line=dict(color="blue"),
            )
        )

        # Forecasted values as a dashed line
        fig.add_trace(
            go.Scatter(
                x=self.test_df["ds"],
                y=self.forecasts["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color="red", dash="dot"),
            )
        )

        fig.update_layout(
            title="Actual Data vs Forecasts for Validation Data",
            xaxis_title=self.date_col,
            yaxis_title=self.target_col,
            width=600, height=300,
        )
        return fig

    def plot_component(self):
        return self.m.plot_components(self.forecasts)

    def calculate_errors(self):
        """
        Make predictions on test set, calculate and return error evaluation metrics: RMSE and MAPE.
        """
        target = self.test_df["y"]  # Actual values in test set

        rmse = root_mean_squared_error(target, self.forecasts["yhat"])
        mape = mean_absolute_percentage_error(target, self.forecasts["yhat"]) * 100

        return rmse, mape

    def inference(self, test_df=None):
        start_date = self.test_df["ds"].max() + pd.Timedelta(1, self.freq)
        end_date = start_date + pd.Timedelta(self.f_period - 1, self.freq)
        future_df = pd.DataFrame(
            {"ds": pd.date_range(start=start_date, end=end_date, freq=self.freq)}
        )
        future_df["y"] = np.nan
        if test_df is not None:
            future_df = pd.concat([future_df, test_df], axis=1)
        future_df = self.add_features(future_df, self.selected_cols)
        future_df = self.create_date_features(future_df)
        predictions = self.m.predict(future_df)

        return predictions[["ds", "yhat"]].rename(
            columns={"ds": self.date_col, "yhat": self.target_col}
        )

    def plot_predictions(self, predictions):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.display_df[self.date_col],
                y=self.display_df[self.target_col],
                mode="lines",
                name="Actual",
                line=dict(color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=predictions[self.date_col],
                y=predictions[self.target_col],
                mode="lines",
                name="Forecast",
                line=dict(color="red", dash="dot"),
            )
        )

        fig.update_layout(
            title="Actual Data and Forecasted Interval",
            xaxis_title=self.date_col,
            yaxis_title=self.target_col,
            width=600, height=300,
        )
        return fig
