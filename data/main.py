import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_option_menu import option_menu  # NEW

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="World-Class Sales Dashboard", layout="wide")
st.title("🌎 Ultimate Global Sales & Profit Dashboard")
st.markdown("Interactive BI-style dashboard with **all high-level plots**, ML & Deep Learning predictions.")

# ------------------------
# Load & Clean Data
# ------------------------
@st.cache_data
def get_data(filename):
    df = pd.read_excel(filename)
    df = df.drop_duplicates(subset=["OrderID","CustomerName"])
    df = df.dropna(subset=["Sales","Profit","Quantity","Region","Category","Sub-Category","CustomerName"])
    df = df[(df["Sales"]>0)&(df["Profit"]>0)&(df["Quantity"]>0)]
    for col in ["Region","Category","Sub-Category","CustomerName"]:
        df[col] = df[col].str.strip()
    for col in ["Sales","Profit","Quantity"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

data = get_data("data/excel.xlsx")

# ------------------------
# Sidebar Filters (Iterative + Collapsible)
# ------------------------
with st.sidebar:
    st.header("✨ Interactive Filters")

    # ----- Region Filter -----
    with st.expander("Filter by Region", expanded=True):
        selected_region = st.multiselect(
            "Select Region",
            options=data["Region"].unique(),
            default=data["Region"].unique()
        )

    # Filter data iteratively for Category
    temp_data_region = data[data["Region"].isin(selected_region)]

    # ----- Category Filter -----
    with st.expander("Filter by Category", expanded=True):
        selected_category = st.multiselect(
            "Select Category",
            options=temp_data_region["Category"].unique(),
            default=temp_data_region["Category"].unique()
        )

    # Filter data iteratively for Sub-Category
    temp_data_category = temp_data_region[temp_data_region["Category"].isin(selected_category)]

    # ----- Sub-Category Filter -----
    with st.expander("Filter by Sub-Category", expanded=True):
        selected_subcat = st.multiselect(
            "Select Sub-Category",
            options=temp_data_category["Sub-Category"].unique(),
            default=temp_data_category["Sub-Category"].unique()
        )

# ------------------------
# Apply final filtered data
# ------------------------
filtered_data = data[
    (data["Region"].isin(selected_region)) &
    (data["Category"].isin(selected_category)) &
    (data["Sub-Category"].isin(selected_subcat))
]

# ------------------------
# Tabs
# ------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Basic Visuals", "Advanced Visuals",
    "Top Customers/Products", "Sankey & Treemap", "Random Forest", "Neural Network"
])

# ------------------------
# Tab1: KPI Overview
# ------------------------
with tab1:
    st.header("🌟 KPIs Overview")
    total_sales = filtered_data['Sales'].sum()
    total_profit = filtered_data['Profit'].sum()
    avg_sales = filtered_data['Sales'].mean()
    avg_profit = filtered_data['Profit'].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
    col3.metric("📊 Average Sales", f"${avg_sales:,.2f}")
    col4.metric("💹 Average Profit", f"${avg_profit:,.2f}")

    st.subheader("📋 Filtered Data Preview")
    with st.expander("Show Full Filtered Data"):
        st.dataframe(filtered_data)

# ------------------------
# Tab2: Basic Visuals
# ------------------------
with tab2:
    st.header("📊 Basic Visualizations")
    fig = px.bar(filtered_data.groupby("Region")["Sales"].sum().reset_index(), x="Region", y="Sales",
                 color="Region", text="Sales")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(filtered_data.groupby("Category")["Profit"].sum().reset_index(), x="Category", y="Profit",
                 color="Category", text="Profit")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for col, color in zip(["Sales","Profit","Quantity"], ["skyblue","lightgreen","salmon"]):
        fig.add_trace(go.Histogram(x=filtered_data[col], name=col, marker_color=color))
    fig.update_layout(barmode='overlay', title="Distribution of Sales, Profit, Quantity")
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Tab3: Advanced Visuals
# ------------------------
with tab3:
    st.header("🌐 Advanced Visualizations")
    fig = px.scatter_matrix(filtered_data, dimensions=["Sales","Profit","Quantity"], color="Category")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for cat in filtered_data['Category'].unique():
        fig.add_trace(go.Box(y=filtered_data[filtered_data['Category']==cat]['Sales'], name=f"Box-{cat}"))
        fig.add_trace(go.Violin(y=filtered_data[filtered_data['Category']==cat]['Sales'], name=f"Violin-{cat}", box_visible=True))
    st.plotly_chart(fig, use_container_width=True)

    region_sum = filtered_data.groupby("Region")["Sales"].sum().reset_index()
    fig = px.pie(region_sum, values='Sales', names='Region', title='Sales Share by Region', hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter_3d(filtered_data, x='Sales', y='Profit', z='Quantity',
                        color='Category', size='Profit', hover_data=['CustomerName'])
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Tab4: Top Customers/Products
# ------------------------
with tab4:
    st.header("🏆 Top Customers & Products")
    top_customers = filtered_data.groupby("CustomerName")["Sales"].sum().nlargest(10).reset_index()
    fig = px.bar(top_customers, x="CustomerName", y="Sales", color="CustomerName", text="Sales")
    st.plotly_chart(fig, use_container_width=True)

    top_products = filtered_data.groupby("Sub-Category")["Sales"].sum().nlargest(10).reset_index()
    fig = px.bar(top_products, x="Sub-Category", y="Sales", color="Sub-Category", text="Sales")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Tab5: Sankey & Treemap
# ------------------------
with tab5:
    st.header("🔗 Sankey & Treemap Plots")
    fig = px.treemap(filtered_data, path=["Category","Sub-Category"], values="Sales", color="Profit", color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.sunburst(filtered_data, path=["Region","Category","Sub-Category"], values="Sales", color="Profit", color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)

    region_category = filtered_data.groupby(["Region","Category"])["Sales"].sum().reset_index()
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=list(region_category['Region'].unique()) + list(region_category['Category'].unique())),
        link=dict(
            source=[0,0,1,1,2,2],
            target=[3,4,3,5,4,5],
            value=region_category['Sales'][:6]
        )
    )])
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Tab6: Random Forest
# ------------------------
with tab6:
    st.header("🤖 Random Forest Prediction")
    numeric_cols = ["Quantity","Profit"]
    sel_col, disp_col = st.columns(2)
    selected_features = sel_col.multiselect("Select features for RF", numeric_cols, default=numeric_cols)
    X_rf = filtered_data[selected_features]
    y_rf = filtered_data["Sales"]

    max_depth = sel_col.slider("Max Depth", 10, 200, 50, key="rf_depth")
    n_estimators_input = sel_col.selectbox("Number of Trees", [100,200,300,'no limit'], key="rf_trees")
    n_estimators = 1000 if n_estimators_input=='no limit' else n_estimators_input

    rf_model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_rf, y_rf)
    rf_pred = rf_model.predict(X_rf)

    disp_col.metric("MAE", f"{mean_absolute_error(y_rf, rf_pred):.2f}")
    disp_col.metric("MSE", f"{mean_squared_error(y_rf, rf_pred):.2f}")
    disp_col.metric("R2", f"{r2_score(y_rf, rf_pred):.2f}")

    st.subheader("Actual vs Predicted Sales (RF)")
    fig = px.scatter(x=y_rf, y=rf_pred, labels={"x":"Actual Sales","y":"Predicted Sales"}, color=y_rf)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predict Custom Sales (RF)")
    custom_input = {}
    for feat in selected_features:
        custom_input[feat] = sel_col.number_input(f"Enter {feat}", value=100.0, key=f"rf_{feat}")
    custom_df = pd.DataFrame([custom_input])
    predicted_sales = rf_model.predict(custom_df)[0]
    st.success(f"Predicted Sales (RF): ${predicted_sales:,.2f}")

# ------------------------
# Tab7: Neural Network
# ------------------------
with tab7:
    st.header("🧠 Neural Network Prediction")
    sel_col2, disp_col2 = st.columns(2)
    selected_features_nn = sel_col2.multiselect("Select features for NN", numeric_cols, default=numeric_cols)
    X_nn = filtered_data[selected_features_nn]
    y_nn = filtered_data["Sales"]

    nn_model = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)
    nn_model.fit(X_nn, y_nn)
    nn_pred = nn_model.predict(X_nn)

    disp_col2.metric("MAE", f"{mean_absolute_error(y_nn, nn_pred):.2f}")
    disp_col2.metric("MSE", f"{mean_squared_error(y_nn, nn_pred):.2f}")
    disp_col2.metric("R2", f"{r2_score(y_nn, nn_pred):.2f}")

    st.subheader("Actual vs Predicted Sales (NN)")
    fig = px.scatter(x=y_nn, y=nn_pred, labels={"x":"Actual Sales","y":"Predicted Sales"}, color=y_nn)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predict Custom Sales (NN)")
    custom_input_nn = {}
    for feat in selected_features_nn:
        custom_input_nn[feat] = sel_col2.number_input(f"Enter {feat}", value=100.0, key=f"nn_{feat}")
    custom_df_nn = pd.DataFrame([custom_input_nn])
    predicted_sales_nn = nn_model.predict(custom_df_nn)[0]
    st.success(f"Predicted Sales (NN): ${predicted_sales_nn:,.2f}")
