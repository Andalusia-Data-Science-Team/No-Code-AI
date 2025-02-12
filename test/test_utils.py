import pytest
import pandas as pd
import numpy as np
from insight.utils import missing, resample, descriptive_analysis
from pandas.testing import assert_frame_equal


@pytest.fixture
def missing_data():
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 6],
            "Name": ["Alice", "Bob", None, "David", "Eve", "David", "David"],
            "Age": [25, np.nan, 30, 22, 28, 30, 30],
            "Salary": [50000, 60000, np.nan, 45000, np.nan, 52000, 52000],
            "City": [
                np.nan,
                np.nan,
                "Chicago",
                "Houston",
                "Chicago",
                "Houston",
                "Houston",
            ],
            "Empty_Col": [np.nan] * 7,
        }
    )  # missing categorical feature, missing numerical features, duplicate row, completely empty column


def test_missing_impute(missing_data):
    expected = pd.DataFrame(
        {
            "ID": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Name": ["Alice", "Bob", "David", "David", "Eve", "David"],
            "Age": [25.0, 27.0, 30.0, 22.0, 28.0, 30.0],
            "Salary": [50000.0, 60000.0, 51750.0, 45000.0, 51750.0, 52000.0],
            "City": ["Chicago", "Chicago", "Chicago", "Houston", "Chicago", "Houston"],
        }
    )
    cleaned_df = missing(missing_data, "Impute Missing Data")
    assert_frame_equal(cleaned_df.reset_index(drop=True), expected)


def test_missing_remove(missing_data):
    expected = pd.DataFrame(
        {
            "ID": [4, 6],
            "Name": ["David", "David"],
            "Age": [22.0, 30.0],
            "Salary": [45000.0, 52000.0],
            "City": ["Houston", "Houston"],
        }
    )
    assert_frame_equal(missing(missing_data).reset_index(drop=True), expected)


@pytest.fixture
def resample_data():
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-03", "2024-01-05"],
            "Name": ["Alice", "Bob", None],
            "Target": [25, np.nan, 30],
            "Salary": [50000, 60000, np.nan],
            "City": [np.nan, "Chicago", "Houston"],
            "Empty_Col": [np.nan] * 3,
        }
    )


@pytest.mark.parametrize(
    "input",
    [
        (
            pd.DataFrame(
                {
                    "Date": ["2024-01-01", "2024-01-03", "2024-01-05"],
                    "Target": [25, 30, 30],
                }
            )
        ),
        (
            pd.DataFrame(
                {
                    "Date": [
                        "2024-01-01 00:01",
                        "2024-01-01 00:02",
                        "2024-01-01 00:03",
                    ],
                    "Target": [25, 30, 30],
                }
            )
        ),
        (
            pd.DataFrame(
                {
                    "Date": ["01/01/2024", "02-01-2024", "3 Jan 2024"],
                    "Target": [25, 30, 30],
                }
            )
        ),
        (
            pd.DataFrame(
                {
                    "Date": ["2024-01-01 00:01:01", None, "2024-01-01 00:03:00"],
                    "Target": [25, 30, 30],
                }
            )
        ),
        (
            pd.DataFrame(
                {"Date": ["2024-01-01", np.nan, "2024-01-05"], "Target": [25, 30, 30]}
            )
        ),
        (
            pd.DataFrame(
                {
                    "Date": ["2024-01-01", "invalid", "2024-01-05"],
                    "Target": [25, 30, 30],
                }
            )
        ),
        (pd.DataFrame({"Date": [123, "not_a_date", 345], "Target": [25, 30, 30]})),
    ],
)
def test_resample_datetime_conversion(input):
    resampled_df = resample(input, "Date", "Target", "D")
    assert (
        resampled_df["Date"].dtype == "datetime64[ns]"
    ), "Date Column Wasn't Parsed Correctly"


def test_resample_same_freq(resample_data):
    expected = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-02 00:00:00",
                    "2024-01-03 00:00:00",
                    "2024-01-04 00:00:00",
                    "2024-01-05 00:00:00",
                ]
            ),
            "Target": [25.0, 26.25, 27.5, 28.75, 30.0],
            "Name": ["Alice", np.nan, "Bob", np.nan, None],
            "Salary": [50000.0, np.nan, 60000.0, np.nan, np.nan],
            "City": [np.nan, np.nan, "Chicago", np.nan, "Houston"],
            "Empty_Col": [np.nan] * 5,
        }
    )
    assert_frame_equal(resample(resample_data, "Date", "Target", "D"), expected)


def test_resample_higher_freq(
    resample_data,
):  # Case of entered frequency higher than actual data frequency
    expected = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31 00:00:00"]),
            "Target": [27.5],
            "Name": [None],
            "Salary": [np.nan],
            "City": [None],
            "Empty_Col": [np.nan],
        }
    )
    assert_frame_equal(resample(resample_data, "Date", "Target", "ME"), expected)


def test_resample_duplicate_timestamps():
    """Test behavior when duplicate timestamps exist."""
    df_with_duplicates = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-07"]
            ),
            "target": [10, 15, np.nan, 20],
        }
    )

    df_resampled = resample(df_with_duplicates, "date", "target", "D")

    # Ensure duplicates were aggregated (mean applied)
    assert (
        df_resampled.loc[df_resampled["date"] == "2024-01-01", "target"].iloc[0] == 12.5
    ), "Duplicate handling incorrect"

    # Ensure interpolation filled missing values
    assert (
        df_resampled["target"].isna().sum() == 0
    ), "Missing values were not interpolated"


def test_resample_one_day_to_hourly():
    """Test resampling from a single day ('D') to hourly ('H') frequency."""
    one_day = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "target": [10, 20]}
    )
    df_resampled = resample(one_day, "date", "target", "h")

    # Ensure correct number of timestamps: 24 hours
    assert (
        len(df_resampled) == 25
    ), "Resampling did not generate the expected number of timestamps"

    # Ensure first and last timestamps match expected range
    assert df_resampled["date"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert df_resampled["date"].iloc[-1] == pd.Timestamp("2024-01-02 00:00:00")

    # Ensure no NaNs remain after interpolation
    assert (
        df_resampled["target"].isna().sum() == 0
    ), "Interpolation did not remove all NaNs"


@pytest.fixture
def analysis_data():
    """Fixture providing a sample DataFrame with numerical and categorical columns"""
    data = {
        "num_col": [1, 2, 2, np.nan, 5],
        "cat_col": ["A", "B", "B", np.nan, "C"],
        "dup_col": ["X", "Y", "Y", "X", "Y"],
    }
    return pd.DataFrame(data)


def test_descriptive_analysis_empty_df():
    """Test that an empty DataFrame raises a ValueError"""
    with pytest.raises(ValueError):
        descriptive_analysis(pd.DataFrame())


def test_descriptive_analysis_numerical_only():
    """Test a DataFrame with only numerical columns"""
    df = pd.DataFrame({"num_col": [10, 20, np.nan, 40]})
    num_desc, cat_desc, d_types, missing, dups, unq = descriptive_analysis(df)

    assert num_desc is not None, "Numerical description should not be None"
    assert cat_desc is None, "Categorical description should be None"
    assert d_types.loc["num_col", "type"] == "int64" or "float64"
    assert missing.loc["num_col", "missing %"] == 25.0  # 1 missing out of 4


def test_descriptive_analysis_categorical_only():
    """Test a DataFrame with only categorical columns"""
    df = pd.DataFrame({"cat_col": ["A", "B", "C", "A", np.nan]})
    num_desc, cat_desc, d_types, missing, dups, unq = descriptive_analysis(df)

    assert num_desc is None, "Numerical description should be None"
    assert cat_desc is not None, "Categorical description should not be None"
    assert missing.loc["cat_col", "missing %"] == 20.0  # 1 missing out of 5


def test_descriptive_analysis_mixed_dataframe(analysis_data):
    """Test a DataFrame with both numerical and categorical columns"""
    num_desc, cat_desc, d_types, missing, dups, unq = descriptive_analysis(
        analysis_data
    )

    assert num_desc is not None, "Numerical description should exist"
    assert cat_desc is not None, "Categorical description should exist"
    assert "num_col" in num_desc.index, "num_col should be in numerical analysis"
    assert "cat_col" in cat_desc.index, "cat_col should be in categorical analysis"

    assert (
        missing.loc["num_col", "missing %"] == 20.0
    ), "Missing % for num_col incorrect"
    assert (
        missing.loc["cat_col", "missing %"] == 20.0
    ), "Missing % for cat_col incorrect"
    assert dups == 20.0, "Duplicates should be 20%"
