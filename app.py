import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv('data/pharma_sales.csv')

# Load model
model = joblib.load('models/sales_model.pkl')

# Title
st.title("AI-Powered Pharmaceutical Sales Forecasting Dashboard")

# Sidebar
st.sidebar.header("Forecast Inputs")

product = st.sidebar.selectbox(
    "Product",
    df['Product'].unique()
)

region = st.sidebar.selectbox(
    "Region",
    df['Region'].unique()
)

price = st.sidebar.slider(
    "Price",
    1,
    100,
    20
)

marketing = st.sidebar.slider(
    "Marketing Spend",
    100,
    5000,
    1000
)

inventory = st.sidebar.slider(
    "Inventory Level",
    100,
    5000,
    1000
)

season = st.sidebar.selectbox(
    "Season",
    df['Season'].unique()
)

# Encode manually
product_map = {
    v: i for i, v in enumerate(df['Product'].unique())
}

region_map = {
    v: i for i, v in enumerate(df['Region'].unique())
}

season_map = {
    v: i for i, v in enumerate(df['Season'].unique())
}

# Input data
input_data = np.array([[
    product_map[product],
    region_map[region],
    price,
    marketing,
    inventory,
    season_map[season],
    2026,
    5,
    1
]])

# Predict
prediction = model.predict(input_data)[0]

st.metric(
    "Predicted Units Sold",
    round(prediction)
)

# Revenue chart
st.subheader("Revenue by Product")

revenue = df.groupby(
    'Product'
)['Revenue'].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=revenue,
    x='Product',
    y='Revenue',
    ax=ax
)

plt.xticks(rotation=45)

st.pyplot(fig)

# Sales Trend
st.subheader("Sales Trend")

trend = df.groupby(
    'Date'
)['Units_Sold'].sum().reset_index()

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    trend['Date'],
    trend['Units_Sold']
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Units Sold")

st.pyplot(fig2)

# Heatmap
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['number'])

fig3, ax3 = plt.subplots(figsize=(8, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax3
)

st.pyplot(fig3)

st.success("Dashboard Loaded Successfully!")