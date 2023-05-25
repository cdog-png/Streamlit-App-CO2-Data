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

#### 3
# Drop rows with null values
df = df.dropna(how = 'any', axis = 0)
df.info()

#### 4
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
# Filter the dataframe for the specified countries and years between 1950 and 2018
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]
# Pivot the dataframe to have countries as columns and years as index
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2')
# Plot the time series with grid
ax = pivot_df.plot(figsize=(12, 8))
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
# Set plot title and labels
plt.title('CO2 Aggregate Emissions (1950-2018)')
plt.xlabel('Year')
plt.ylabel('CO2 Aggregate Emissions')
# Add country labels to each line
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
# Modify legend
ax.legend(prop={'size': 7})
# Show the plot
st.pyplot(fig)

#### 5
# Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Drop nonfloat variables 
df_float = df.drop(['country', 'year', 'iso_code'], axis = 1)
# Create function called to compute VIF
def calc_VIF(x):
  vif= pd.DataFrame()
  vif['variables'] = x.columns
  vif["VIF"] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
  return(vif)
x = df_float.iloc[:,:]
# Display VIF
calc_VIF(x)

#### 6
# Drop vars with high VIF (>5) which are not essential to the model 
df = df.drop(['iso_code','cement_co2', 'co2','co2_including_luc', 'cumulative_cement_co2', 'cumulative_co2', 
              'cumulative_co2_including_luc', 'cumulative_coal_co2', 'cumulative_luc_co2', 'cumulative_oil_co2', 
              'land_use_change_co2', 'share_global_co2', 'share_global_co2_including_luc', 'share_global_cumulative_co2', 
              'share_global_cumulative_co2_including_luc', 'share_global_cumulative_coal_co2', 
              'share_global_cumulative_gas_co2', 'share_global_cumulative_luc_co2', 'share_global_cumulative_oil_co2'], 
               axis = 1)
df.info()

#### 7
# Keep only 2018 observations in a new df
mask = df['year'] == 2018
df_2018 = df[mask]
df_2018.head()

#### 8
# Assuming you have a DataFrame named 'df' with a column named 'country'
unique_countries = df['country'].unique()
# Print the unique values
for country in unique_countries:
    print(country)

#### 9
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
# Filter the dataframe for the specified countries and years between 1950 and 2018
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]
# Pivot the dataframe to have countries as columns and years as index
pivot_df = filtered_df.pivot(index = 'year', columns = 'country', values = 'co2_per_capita')
# Plot the time series with grid
ax = pivot_df.plot(figsize=(12, 8))
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
# Set plot title and labels
plt.title('CO2 Per Capita Emissions (1950-2018)')
plt.xlabel('Year')
plt.ylabel('CO2 Per Capita Emissions')
# Add country labels to each line with reduced font size
# Add country labels to each line with reduced font size and no overlapping
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
ax.legend(prop = {'size': 7})
# Show the plot
plt.show()

#### 10
## Plot top countries
# Group the data by country and sum the co2 values for each country
grouped_data = df_2018.groupby('country')['co2_per_unit_energy'].sum()
# Select the top 20 countries by CO2 emissions
top_20_countries = grouped_data.nlargest(20)
# Plot the histogram with country names on the x-axis
fig = plt.figure(figsize = (15, 8))
plt.bar(top_20_countries.index, top_20_countries.values, width = 0.5)
plt.xlabel('Country')
plt.ylabel('CO2 per unit of energy in 2018')
plt.title('CO2 per unit of energy by country in 2018')
plt.xticks(rotation = 50, fontsize = 10)
plt.show()

#### 11
# CREATE GDPpc VARIABLE
df['gdp_per_capita'] = df['gdp'] / df['population']
# REPEAT FOR 2018
df_2018['gdp_per_capita'] = df_2018['gdp'] / df_2018['population']

#### 12
# Plot relationship between GDPpc and CO2 per unit of energy
fig = plt.figure(figsize = (8, 6))
# Set up line chart
sns.regplot(x = df_2018['gdp_per_capita'], y = df_2018['co2_per_unit_energy'], color = 'blue')
plt.xlabel('gdp_per_capita')
plt.ylabel('co2 per unit energy')
plt.title('Relationship between GDPpc and CO2 per unit of energy (2018)')
# Show plot
plt.show()

#### 13
# PRINT UNIQUE COUNTRIES
print(df.country.unique())
# CREATE CONTINENTAL LISTS
africa = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 
          'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Democratic Republic of Congo', 'Djibouti', 'Egypt', 
          'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 
          'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 
          'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 
          'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 
          'Zimbabwe']
asia = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 
        'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 
        'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 
        'Palestine', 'Philippines', 'Qatar', 'Russia', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 
        'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 
        'Vietnam', 'Yemen']
europe = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 
          'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 
          'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 
          'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 
          'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City']
north_america = ['Canada', 'Mexico', 'United States']
south_america = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 
                 'Uruguay', 'Venezuela']
oceania = ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 
           'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu']

#### 14
# Create new continent column (2018)
continent = []
for country in df_2018['country']:
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
df_2018['continent'] = continent
df_2018.head()

#### 15
# COMPUTE MEAN SHARE OF GLOBAL OIL C02 EMISSIONS BY CONTINENT (2018)
co2_by_continent = df_2018.groupby('continent')['share_global_oil_co2'].mean()
# CREATE BARPLOT
co2_by_continent.plot(kind='bar', color='blue')
plt.xlabel('Continent')
plt.ylabel('Share of global oil CO2 emissions')
plt.title('Mean share of global oil CO2 emissions by continent in 2018')
plt.show()

#### 16
# REPLICATE GRAPH FOR 1990
mask = df['year'] == 1990
df_1990 = df[mask]
# CREATE NEW CONTINENT COLUMN
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
# COMPUTE MEAN SHARE OF GLOBAL OIL C02 EMISSIONS BY CONTINENT
co2_by_continent = df_1990.groupby('continent')['share_global_oil_co2'].mean()
# CREATE BARPLOT
co2_by_continent.plot(kind='bar', color='blue')
plt.xlabel('Continent')
plt.ylabel('Share of global oil CO2 emissions')
plt.title('Mean share of global oil CO2 emissions by continent in 1990')
plt.show()

#### 17
## PLOT TOP COUNTRIES
# Group the data by country and sum the co2 values for each country
grouped_data = df_1990.groupby('country')['co2_per_capita'].sum()
# Select the top 20 countries by CO2 emissions
top_20_countries = grouped_data.nlargest(20)
# Plot the histogram with country names on the x-axis
fig = plt.figure(figsize = (15, 8))
plt.bar(top_20_countries.index, top_20_countries.values, width = 0.5)
plt.xlabel('Country')
plt.ylabel('Per capita CO2 emissions in 1990')
plt.title('Per capita CO2 emissions by country in 1990')
plt.xticks(rotation = 50, fontsize = 10)
plt.show()

#### 18
## SHOW CO2 PER CAPITA AND GDP PER CAPITA RELATIONSHIP
# Create new figure
fig = plt.figure(figsize = (8, 6))
# Set up line chart
sns.regplot(x = df_1990['gdp_per_capita'], y = df_1990['co2_per_capita'], color = 'blue')
plt.xlabel('gdp_per_capita')
plt.ylabel('co2 per capita')
plt.title('Relationship between GDPpc and CO2 per capita (1990)')
# Show plot
plt.show()

#### 19
## SHOW CO2 PER CAPITA AND GDP PER CAPITA RELATIONSHIP
# Create new figure
fig = plt.figure(figsize = (8, 6))
# Set up line chart
sns.regplot(x = df_2018['gdp_per_capita'], y = df_2018['co2_per_capita'], color = 'blue')
plt.xlabel('gdp_per_capita')
plt.ylabel('co2 per capita')
plt.title('Relationship between GDPpc and CO2 per capita (2018)')
# Show plot
plt.show()

#### 20
# Perform an Explanatory Data Analysis
sns.pairplot(df, x_vars=['co2_per_capita'], y_vars=['gdp_per_capita'], height = 5)
# Use correlation analysis to identify potential features
corr = df.corr()
print(corr)
# Remove string variables from df
df_numeric = df.select_dtypes(exclude = ['object'])
# Use Lasso regression to select features
X = df_numeric.drop(['gdp_per_capita'], axis = 1)
y = df_numeric['gdp_per_capita']
from sklearn.linear_model import Lasso
model = Lasso(alpha = 0.1)
model.fit(X, y)
# Print coefficients
coefficients = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_})
print(coefficients)
# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('R-squared:', r2)
print('MSE:', mse)

#### 21
# Filter the dataframe for the years between 1950 and 2018 and the specified countries
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
df_filtered = df[(df['year'].between(1950, 2018)) & (df['country'].isin(countries))]
# Group the data by country and year
grouped_df = df_filtered.groupby(['country', 'year']).sum()
# Get the variables for CO2 emissions
variables = ['co2', 'coal_co2', 'oil_co2', 'gas_co2']
# Create a figure and subplots for each variable
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes.flatten()
# Iterate through each variable and plot the evolution of CO2 emissions for each country
for i, var in enumerate(variables):
    ax = axes[i]  # Get the current subplot
    for country in countries:
        values = grouped_df.loc[country, var]
        ax.plot(values.index, values.values, label=country)
    ax.set_title(f'{var.capitalize()} Emissions')
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions')
    ax.legend()
    ax.grid(True)  # Add grid to the subplot
# Adjust the spacing between subplots
plt.tight_layout()
# Show the plot
plt.show()

