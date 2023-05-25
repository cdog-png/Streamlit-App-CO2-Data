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


# Drop vars with under 10K non-NAs
df = df.drop(['co2_including_luc_per_unit_energy', 'consumption_co2', 
              'consumption_co2_per_capita', 'consumption_co2_per_gdp', 'cumulative_other_co2', 'energy_per_capita', 
              'energy_per_gdp', 'ghg_excluding_lucf_per_capita', 'ghg_per_capita', 'methane', 'methane_per_capita', 
              'nitrous_oxide', 'nitrous_oxide_per_capita', 'other_co2_per_capita', 'other_industry_co2', 
              'share_global_cumulative_other_co2', 'share_global_other_co2', 'total_ghg', 'total_ghg_excluding_lucf', 
              'trade_co2', 'trade_co2_share', 'primary_energy_consumption'], axis = 1)
df.info()

# Drop rows with null values
df = df.dropna(how='any', axis=0)

# Filter the dataframe for the specified countries and years
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]

# Pivot the dataframe
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2')

# Create a Streamlit line chart
st.line_chart(pivot_df, use_container_width=True)

# Customize the chart
plt.figure(figsize=(12, 8))
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.title('CO2 Aggregate Emissions (1950-2018)')
plt.xlabel('Year')
plt.ylabel('CO2 Aggregate Emissions')

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

plt.legend(prop={'size': 7})
st.pyplot(plt)

# Step 7: Keep only 2018 observations in a new df
mask = df['year'] == 2018
df_2018 = df[mask]
st.write("DataFrame for 2018:")
st.write(df_2018.head())

# Step 8: Print unique countries
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
st.pyplot(plt.gcf())  # Show the line chart in Streamlit

# Step 10: Plot top countries
grouped_data = df_2018.groupby('country')['co2_per_unit_energy'].sum()
top_20_countries = grouped_data.nlargest(20)

# Create a Streamlit bar chart for top countries
st.bar_chart(top_20_countries, use_container_width=True)
st.pyplot(plt.gcf())  # Show the bar chart in Streamlit

# Step 11: Create GDP per capita variable
df['gdp_per_capita'] = df['gdp'] / df['population']

# Step 12: Plot relationship between GDP per capita and CO2 per unit of energy
fig = plt.figure(figsize=(8, 6))
sns.regplot(x=df_2018['gdp_per_capita'], y=df_2018['co2_per_unit_energy'], color='blue')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 per unit energy')
plt.title('Relationship between GDP per capita and CO2 per unit of energy (2018)')
st.pyplot(fig)

# Step 13: Print unique countries and create continental lists
unique_countries = df['country'].unique()
continents = {
    'africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde',
               'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Democratic Republic of Congo', 'Djibouti',
               'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
               'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali',
               'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda',
               'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
               'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'],
    'asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China',
             'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan',
             'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal',
             'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Russia', 'Saudi Arabia',
             'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste',
             'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen'],
    'europe': ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia',
               'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
               'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta',
               'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal',
               'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland',
               'Ukraine', 'United Kingdom', 'Vatican City'],
    'north_america': ['Canada', 'Mexico', 'United States'],
    'south_america': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru',
                      'Suriname', 'Uruguay', 'Venezuela'],
    'oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau',
                'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu']
}

# Step 14: Create new continent column (2018)
df_2018['continent'] = df_2018['country'].apply(
    lambda x: next((continent for continent, countries in continents.items() if x in countries), 'Unknown')
)

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
st.pyplot(fig)

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
st.pyplot(fig)

# Step 17: Plot top countries by CO2 emissions
grouped_data = df_1990.groupby('country')['co2_per_capita'].sum()
top_20_countries = grouped_data.nlargest(20)

fig = plt.figure(figsize=(15, 8))
plt.bar(top_20_countries.index, top_20_countries.values, width=0.5)
plt.xlabel('Country')
plt.ylabel('Per capita CO2 emissions in 1990')
plt.title('Per capita CO2 emissions by country in 1990')
plt.xticks(rotation=50, fontsize=10)
st.pyplot(fig)

# Step 18: Show relationship between CO2 per capita and GDP per capita in 1990
fig = plt.figure(figsize=(8, 6))
sns.regplot(x=df_1990['gdp_per_capita'], y=df_1990['co2_per_capita'], color='blue')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 per capita')
plt.title('Relationship between GDP per capita and CO2 per capita (1990)')
st.pyplot(fig)

# Step 19: Show relationship between CO2 per capita and GDP per capita in 2018
fig = plt.figure(figsize=(8, 6))
sns.regplot(x=df_2018['gdp_per_capita'], y=df_2018['co2_per_capita'], color='blue')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 per capita')
plt.title('Relationship between GDP per capita and CO2 per capita (2018)')
st.pyplot(fig)

