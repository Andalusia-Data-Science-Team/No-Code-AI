import pytest
import re
from insight.utils import format_value  # Replace with the actual import

def test_format_value_with_string():
    assert format_value("123.45000", "%.2f") == "123.45"
    assert format_value("-123.45000", "%.2f") == "\u2212123.45"
    assert format_value("123.000", "%.2f") == "123"
    assert format_value("-123.000", "%.2f") == "\u2212123"

def test_format_value_with_integer():
    assert format_value(123.45000, "%.2f") == "123.45"
    assert format_value(-123.45000, "%.2f") == "\u2212123.45"
    assert format_value(123.000, "%.2f") == "123"
    assert format_value(-123.000, "%.2f") == "\u2212123"

# def test_format_value_with_negative_zero():
#     assert format_value(-0.000, "%.2f") == "0"
#     assert format_value(-0.001, "%.3f") == "\u22120.001"

def test_format_value_with_non_string_input():
    assert format_value(123, "%d") == "123"
    assert format_value(-123, "%d") == "\u2212123"

def test_format_value_with_edge_cases():
    assert format_value("0.000", "%.3f") == "0"
    assert format_value("-0.000", "%.3f") == "\u22120"
    assert format_value("0.0000", "%.4f") == "0"
    assert format_value("-0.0000", "%.4f") == "\u22120"