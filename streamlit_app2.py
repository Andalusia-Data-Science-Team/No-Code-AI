import streamlit as st

st.set_page_config(page_title="Business AI Tool", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Analyze Data", "Train Model", "Predictions"])

if page == "Home":
    st.title("Welcome to the Business AI Tool")
    st.write("This tool enables you to analyze data, train models, and make predictions with ease.")
elif page == "Upload Data":
    st.title("Upload Your Data")
    # Add upload functionality here
elif page == "Analyze Data":
    st.title("Data Analysis")
    # Add analysis functionality here
elif page == "Train Model":
    st.title("Model Training")
    # Add model training functionality here
elif page == "Predictions":
    st.title("Make Predictions")
    # Add prediction functionality here
