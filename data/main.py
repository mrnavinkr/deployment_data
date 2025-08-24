print("updated successfully")
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Containers
header = st.container()
dataset = st.container()
feature = st.container()
modeltrainer = st.container()

@st.cache_data
def get_data(filename):
    data = pd.read_excel(filename)
    return data
    

# Header Section
with header:
    st.title("Welcome to my app")
    st.text("In this project, I look into the transaction of customers")

# Dataset Section
with dataset:
    st.header("This is a dataset")
    data = get_data("data/excel.xlsx")
    st.write(data.head())
    st.title("Data all about dscribe......!")
    st.dataframe(data.describe())
    

    st.subheader("Data visualization of Sales")
    sales_cal = pd.DataFrame(data["Sales"].value_counts()).head(100)
    st.bar_chart(sales_cal)

# Feature Section
with feature:
    st.header("The features I created")
    st.markdown("* **first feature:** I created this feature because of this... I calculated this feature using this logic...")
    st.markdown("* **second feature:** I created this feature because of this... I calculated this feature using this logic...")

# Model Training Section
with modeltrainer:
    st.header("Time to change model")
    st.text("Let's train a Random Forest Model")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=200, step=10)

    # "no limit" option add kiya
    n_estimators_input = sel_col.selectbox(
        'How many trees should be there in the model?',
        options=[100, 200, 300, 'no limit']
    )
    sel_col.text_input('here is a list of features in my data')
    sel_col.write(data.columns) 

    # handle "no limit"
    if n_estimators_input == 'no limit':
        n_estimators = 1000  # ya koi bada number
    else:
        n_estimators = n_estimators_input

    # Column name must match exactly
    input_feature = sel_col.text_input('Which feature should be used as input feature?', 'Sales')

    # X = input feature, y = target (Sales)
    if input_feature in data.columns:
        X = data[[input_feature]]
        y = data[["Sales"]].values.ravel()  # 1D array banaya

        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        regr.fit(X, y)

        prediction = regr.predict(X)

        disp_col.subheader("Mean Absolute Error:")
        disp_col.write(mean_absolute_error(y, prediction))

        disp_col.subheader("Mean Squared Error:")
        disp_col.write(mean_squared_error(y, prediction))

        disp_col.subheader("R2 Score:")
        disp_col.write(r2_score(y, prediction))
    else:
        st.error(f"⚠️ Column '{input_feature}' not found in dataset. Available columns: {list(data.columns)}")
