import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("GLOBAL CO2 EMISSION DATA ANALYSIS")
st.subheader("by Caspar Ibel and Luis Manosas")

df = pd.read_csv('owid-co2-data.csv')
st.write(df.head())

# Step 3: Drop rows with null values and display info
df = df.dropna(how='any', axis=0)
st.write("DataFrame with dropped null values:")
st.write(df.head())

# Step 4: Print unique countries
unique_countries = df['country'].unique()
st.write("Unique countries:")
for country in unique_countries:
    st.write(country)

# Step 9: Plot CO2 per capita emissions
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2_per_capita')

# Create a Streamlit line chart for CO2 per capita emissions
st.line_chart(pivot_df, use_container_width=True)

# Step 10: Plot top countries
grouped_data = df.groupby('country')['co2_per_capita'].sum()
top_20_countries = grouped_data.nlargest(20)

# Create a Streamlit bar chart for top countries
st.bar_chart(top_20_countries, use_container_width=True)
