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

# Drop nonfloat variables
df_float = df.drop(['country', 'year', 'iso_code'], axis=1)

x = df_float.iloc[:, :]

# Evolution CO2 emissions aggregate (1950 - 2018)
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
# Filter the dataframe for the specified countries and years between 1950 and 2018
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]
# Pivot the dataframe to have countries as columns and years as index
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2')
# Plot the time series with grid
fig, ax = plt.subplots(figsize=(12, 8))
pivot_df.plot(ax=ax)
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
# Set plot title and labels
plt.title('CO2 Aggregate Emissions (1950-2018)')
plt.xlabel('Year')
plt.ylabel('CO2 Aggregate Emissions')
# Add country labels to each line with reduced font size
for country in countries:
    last_value = pivot_df[country].iloc[-1]
    label = country[:3]  # Abbreviate country name
    x_pos = pivot_df.index[-1]
    y_pos = last_value
    # Adjust label position to avoid overlapping
    if country == 'United Kingdom':
        y_pos -= 0.5
    elif country == 'France':
        x_pos -= 0
        y_pos -= 100
    elif country == 'Germany':
        x_pos -= 2
        y_pos -= -100
    elif country == 'Spain':
        x_pos -= 0
        y_pos -= 200
    elif country == 'Brazil':
        x_pos -= 2
        y_pos -= -50
    elif country == 'South Korea':
        x_pos -= 0
        y_pos -= 50
    plt.text(x_pos, y_pos, label, fontsize=8)
# Shrink the legend size to half
ax.legend(prop={'size': 7})
# Show the plot
st.pyplot(fig)
# Evolution CO2 emissions per capita (1950 - 2018)
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
# Filter the dataframe for the specified countries and years between 1950 and 2018
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]
# Pivot the dataframe to have countries as columns and years as index
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2_per_capita')
# Plot the time series with grid
fig, ax = plt.subplots(figsize=(12, 8))
pivot_df.plot(ax=ax)
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
# Set plot title and labels
plt.title('CO2 Per Capita Emissions (1950-2018)')
plt.xlabel('Year')
plt.ylabel('CO2 Per Capita Emissions')
# Add country labels to each line with reduced font size
for country in countries:
    last_value = pivot_df[country].iloc[-1]
    label = country[:3]  # Abbreviate country name
    x_pos = pivot_df.index[-1]
    y_pos = last_value
    # Adjust label position to avoid overlapping
    if country == 'United Kingdom':
        y_pos -= 0.5
    elif country == 'South Korea':
        x_pos -= 2
        y_pos += 0.3
    elif country == 'France':
        x_pos -= 2
        y_pos -= 0.2
    elif country == 'Germany':
        x_pos -= 2
        y_pos -= 0.2
    plt.text(x_pos, y_pos, label, fontsize=8)
# Shrink the legend size to half
ax.legend(prop={'size': 7})
# Show the plot
st.pyplot(fig)
# CO2 PER CAPITA AND GDP PER CAPITA RELATIONSHIP 1990
# Create new figure
fig = plt.figure(figsize=(8, 6))
# Set up line chart
sns.regplot(x=df_1990['gdp_per_capita'], y=df_1990['co2_per_capita'], color='blue')
plt.xlabel('gdp_per_capita')
plt.ylabel('co2 per capita')
plt.title('Relationship between GDPpc and CO2 per capita (1990)')
# Show plot
st.pyplot(fig)
# CO2 PER CAPITA AND GDP PER CAPITA RELATIONSHIP 2018
# Create new figure
fig = plt.figure(figsize=(8, 6))
# Set up line chart
sns.regplot(x=df_2018['gdp_per_capita'], y=df_2018['co2_per_capita'], color='blue')
plt.xlabel('gdp_per_capita')
plt.ylabel('co2 per capita')
plt.title('Relationship between GDPpc and CO2 per capita (2018)')
# Show plot
st.pyplot(fig)
# PLOT TOP COUNTRIES 2018 (per capita emissions)
# Group the data by country and sum the co2 values for each country
grouped_data = df_2018.groupby('country')['co2_per_capita'].sum()
# Select the top 20 countries by CO2 emissions
top_20_countries = grouped_data.nlargest(20)
# Plot the histogram with country names on the x-axis
fig = plt.figure(figsize=(15, 8))
plt.bar(top_20_countries.index, top_20_countries.values, width=0.5)
plt.xlabel('Country')
plt.ylabel('Per capita CO2 emissions in 2018')
plt.title('Per capita CO2 emissions by country in 2018')
plt.xticks(rotation=50, fontsize=10)
# Show plot
st.pyplot(fig)
# PLOT TOP COUNTRIES 2018 (per unit of energy)
# Group the data by country and sum the co2 values for each country
grouped_data = df_2018.groupby('country')['co2_per_unit_energy'].sum()
# Select the top 20 countries by CO2 emissions
top_20_countries = grouped_data.nlargest(20)
# Plot the histogram with country names on the x-axis
fig = plt.figure(figsize=(15, 8))
plt.bar(top_20_countries.index, top_20_countries.values, width=0.5)
plt.xlabel('Country')
plt.ylabel('CO2 per unit of energy in 2018')
plt.title('CO2 per unit of energy by country in 2018')
plt.xticks(rotation=50, fontsize=10)
# Show plot
st.pyplot(fig)
# GDPpc and CO2 per unit of energy relationship
# Create new figure
fig = plt.figure(figsize=(8, 6))
# Set up line chart
sns.regplot(x=df_2018['gdp_per_capita'], y=df_2018['co2_per_unit_energy'], color='blue')
plt.xlabel('gdp_per_capita')
plt.ylabel('co2 per unit energy')
plt.title('Relationship between GDPpc and CO2 per unit of energy (2018)')
# Show plot
st.pyplot(fig)

