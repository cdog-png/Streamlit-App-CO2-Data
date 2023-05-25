import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from PIL import Image  # Added PIL import
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # Added FigureCanvas import

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

# Step 11: Create GDP per capita variable
df['gdp_per_capita'] = df['gdp'] / df['population']

# Step 12: Create df_2018 DataFrame for 2018 data
df_2018 = df[df['year'] == 2018]

# Plot relationship between GDP per capita and CO2 per unit of energy
fig = plt.figure(figsize=(8, 6))
sns.regplot(x=df_2018['gdp_per_capita'], y=df_2018['co2_per_unit_energy'], color='blue')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 per unit energy')
plt.title('Relationship between GDP per capita and CO2 per unit of energy (2018)')

# Convert Matplotlib figure to PIL image
canvas = FigureCanvas(fig)
image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

# Display the image using Streamlit
st.image(image)

# Step 13: Print unique countries and create continental lists
unique_countries = df['country'].unique()
continents = {
    'africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde',
               'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Democratic Republic of Congo', 'Djibouti',
               'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
               'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali',
               'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe',
               'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania',
               'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
    'asia': ['Afghanistan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'East Timor',
             'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan',
             'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman',
             'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka',
             'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Turkey', 'Turkmenistan', 'United Arab Emirates',
             'Uzbekistan', 'Vietnam', 'Yemen'],
    'europe': ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
               'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia',
               'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein',
               'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia',
               'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia',
               'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City'],
    'north_america': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba',
                      'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras',
                      'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia',
                      'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
    'oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand',
                'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
    'south_america': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay',
                      'Peru', 'Suriname', 'Uruguay', 'Venezuela']
}

# Print unique countries
st.write("Unique countries:")
for country in unique_countries:
    st.write(country)

# Step 14: Create continent select box and filter DataFrame
continent = st.selectbox('Select a continent', list(continents.keys()))
continent_countries = continents[continent]
filtered_df = df[df['country'].isin(continent_countries)]

# Create a Streamlit line chart for CO2 per capita emissions by continent
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2_per_capita')
st.line_chart(pivot_df, use_container_width=True)

# Display the modified DataFrame
st.write("DataFrame for 2018 with continent column:")
st.write(df_2018)

# Step 15: Compute mean share of global oil CO2 emissions by continent (2018)
co2_by_continent = df_2018.groupby('continent')['share_global_oil_co2'].mean()

# Create bar plot
fig = plt.figure(figsize=(8, 6))
co2_by_continent.plot(kind='bar', color='blue')
plt.xlabel('Continent')
plt.ylabel('Share of global oil CO2 emissions')
plt.title('Mean share of global oil CO2 emissions by continent in 2018')
st.image(fig)

# Step 16: Replicate graph for 1990
mask = df['year'] == 1990
df_1990 = df[mask]

# Create new continent column
continent = []
for country in df_1990['country']:
    if country in asia:
        continent.append('Asia')
    elif country in europe:
        continent.append('Europe')
    elif country in north_america:
        continent.append('North America')
    elif country in south_america:
        continent.append('South America')
    elif country in africa:
        continent.append('Africa')
    else:
        continent.append('Oceania')
df_1990['continent'] = continent

# Compute mean share of global oil CO2 emissions by continent
co2_by_continent = df_1990.groupby('continent')['share_global_oil_co2'].mean()

# Create bar plot
fig = plt.figure(figsize=(8, 6))
co2_by_continent.plot(kind='bar', color='blue')
plt.xlabel('Continent')
plt.ylabel('Share of global oil CO2 emissions')
plt.title('Mean share of global oil CO2 emissions by continent in 1990')
st.image(fig)

# Step 17: Plot top countries by CO2 emissions
grouped_data = df_1990.groupby('country')['co2_per_capita'].sum()
top_20_countries = grouped_data.nlargest(20)

fig = plt.figure(figsize=(15, 8))
plt.bar(top_20_countries.index, top_20_countries.values, width=0.5)
plt.xlabel('Country')
plt.ylabel('Per capita CO2 emissions in 1990')
plt.title('Per capita CO2 emissions by country in 1990')
plt.xticks(rotation=50, fontsize=10)
st.image(fig)

# Step 18: Show relationship between CO2 per capita and GDP per capita in 1990
fig = plt.figure(figsize=(8, 6))
sns.regplot(x=df_1990['gdp_per_capita'], y=df_1990['co2_per_capita'], color='blue')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 per capita')
plt.title('Relationship between GDP per capita and CO2 per capita (1990)')
st.image(fig)

# Step 19: Show relationship between CO2 per capita and GDP per capita in 2018
fig = plt.figure(figsize=(8, 6))
sns.regplot(x=df_2018['gdp_per_capita'], y=df_2018['co2_per_capita'], color='blue')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 per capita')
plt.title('Relationship between GDP per capita and CO2 per capita (2018)')
st.image(fig)
