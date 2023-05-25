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


st.title("GLOBAL CO2 EMISSION DATA ANALYSIS")
st.subheader("by Caspar Ibel and Luis Manosas")

# Read the CSV file into a DataFrame
df = pd.read_csv('owid-co2-data.csv')

# Display the DataFrame
st.header("Preview of the Dataset")
st.write(df.head())
st.markdown('#')

df = df.drop(['co2_including_luc_per_unit_energy', 'consumption_co2', 
              'consumption_co2_per_capita', 'consumption_co2_per_gdp', 'cumulative_other_co2', 'energy_per_capita', 
              'energy_per_gdp', 'ghg_excluding_lucf_per_capita', 'ghg_per_capita', 'methane', 'methane_per_capita', 
              'nitrous_oxide', 'nitrous_oxide_per_capita', 'other_co2_per_capita', 'other_industry_co2', 
              'share_global_cumulative_other_co2', 'share_global_other_co2', 'total_ghg', 'total_ghg_excluding_lucf', 
              'trade_co2', 'trade_co2_share', 'primary_energy_consumption'], axis = 1)
df.info()

df = df.dropna(how = 'any', axis = 0)
df.info()

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

# Drop vars with high VIF (>5) which are not essential to the model 
df = df.drop(['cement_co2', 'co2','co2_including_luc', 'cumulative_cement_co2', 'cumulative_co2', 
              'cumulative_co2_including_luc', 'cumulative_coal_co2', 'cumulative_luc_co2', 'cumulative_oil_co2', 
              'land_use_change_co2', 'share_global_co2', 'share_global_co2_including_luc', 'share_global_cumulative_co2', 
              'share_global_cumulative_co2_including_luc', 'share_global_cumulative_coal_co2', 
              'share_global_cumulative_gas_co2', 'share_global_cumulative_luc_co2', 'share_global_cumulative_oil_co2'], 
               axis = 1)
df.info()

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

df=df.drop(['country', 'year', 'iso_code'], axis=1)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1

# Identify the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Replace outliers with the 25th and 75th percentiles
df_cleaned = df.mask((df < lower_bound) | (df > upper_bound), (Q1 + Q3) / 2, axis=1)

# Print the cleaned DataFrame
#print(df_cleaned)

df_cleaned.info()

from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()

# Fit the scaler to the data and transform the data
df_normalized = scaler.fit_transform(df_cleaned)

# Convert the standardized array back to a DataFrame
df_normalized = pd.DataFrame(df_normalized, columns=df_cleaned.columns)

# Print the standardized data
df_normalized.describe()

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
scaler = MinMaxScaler()

# Fit the scaler to the data and transform the data
df_normalized = scaler.fit_transform(df_cleaned)

# Convert the normalized array back to a DataFrame
df_normalized = pd.DataFrame(df_normalized, columns=df_cleaned.columns)

# Print the normalized data
df_normalized.describe()

correlation_list = df_normalized.corrwith(df_normalized['co2_per_capita'])
correlation_list = correlation_list.drop('co2_per_capita')  # Remove correlation with itself
correlation_list = correlation_list.sort_values(ascending=True)

st.header("Correlation List CO2 per capita")
st.write(correlation_list)
st.markdown('#')

df_normalized = df_normalized
target = df_normalized['co2_per_capita']
data = df_normalized.drop('co2_per_capita', axis=1)
#use all variables with a higher correlation than 50%
signif_feats = df_normalized[['co2_per_gdp', 'oil_co2_per_capita', 'cement_co2_per_capita', 'gas_co2_per_capita',
                            'co2_including_luc_per_capita']]

EPSILON = 1e-8  # Small constant value to avoid zero/negative values

signif_feats_log = np.log(signif_feats + EPSILON)

# Convert the transformed array back to a DataFrame
signif_feats_log = pd.DataFrame(signif_feats_log, columns=signif_feats.columns)
signif_feats_log.describe()

X_train, X_test, y_train, y_test = train_test_split(signif_feats, target, test_size=0.2, random_state=789)
fig, ax = plt.subplots(figsize=(50, 30))

# Create the heatmap using seaborn
heatmap = sns.heatmap(df_normalized.corr(), annot=True, cmap="RdBu_r", center=0)

# Display the heatmap using Streamlit
st.pyplot(fig)
st.markdown('#')

X_train, X_test, y_train, y_test = train_test_split(signif_feats, target, test_size=0.2, random_state=789)

slr = LinearRegression()
slr.fit(X_train, y_train)

coeffs = list(slr.coef_)
coeffs.insert(0, slr.intercept_)

feats = list(signif_feats.columns)
feats.insert(0, 'intercept')

# Sort coefficients in ascending order
sorted_coeffs = sorted(coeffs)

# Create DataFrame with sorted coefficients
df_coeffs = pd.DataFrame({'Estimated Value': sorted_coeffs}, index=feats)

st.write(df_coeffs)
st.write(slr.score(X_train, y_train))
st.write(cross_val_score(slr,X_train, y_train).mean())
st.write(slr.score(X_test, y_test))

pred_test = slr.predict(X_test)

# Create a scatter plot
fig, ax = plt.subplots()
ax.scatter(pred_test, y_test)
ax.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()))

# Display the scatter plot using Streamlit
st.pyplot(fig)

pred_train = slr.predict(X_train)
residuals = pred_train - y_train

# Create a scatter plot
fig, ax = plt.subplots()
ax.scatter(y_train, residuals, s=15)
ax.plot((y_train.min(), y_train.max()), (0, 0), lw=3, color='#0a5798')

# Display the scatter plot using Streamlit
st.pyplot(fig)


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming you have already defined X_train as the feature matrix

# Add a constant column to X_train
X_train_with_constant = sm.add_constant(X_train)

# Calculate the VIF for each feature
vif = pd.DataFrame()
vif["Feature"] = X_train_with_constant.columns
vif["VIF"] = [variance_inflation_factor(X_train_with_constant.values, i) for i in range(X_train_with_constant.shape[1])]

print(vif)

residuals_norm = (residuals - residuals.mean()) / residuals.std()

# Create a probability plot
fig, ax = plt.subplots()
stats.probplot(residuals_norm, plot=plt)

# Display the probability plot using Streamlit
st.pyplot(fig)


columns = ['co2_per_gdp', 'oil_co2_per_capita', 'cement_co2_per_capita', 'gas_co2_per_capita', 'co2_including_luc_per_capita', 'co2_per_capita']

# Set the desired figure size
fig = plt.figure(figsize=(8, 8))

# Create the pairplot
sns.pairplot(df_normalized[columns], height=1.75)

# Display the plot using Streamlit
st.pyplot(fig)

signif_feats_log = np.log(signif_feats)

# Convert the transformed array back to a DataFrame
signif_feats_log = pd.DataFrame(signif_feats_log, columns=signif_feats.columns)

# Print the transformed data
print(signif_feats_log.describe())
from sklearn.linear_model import LinearRegression

lr2 = LinearRegression()
lr2.fit(X_train.loc[:, signif_feats_log], y_train)
train_score = lr2.score(X_train.loc[:, signif_feats_log], y_train)
test_score = lr2.score(X_test.loc[:, signif_feats_log], y_test)

print("Training score:", train_score)
print("Test score:", test_score)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(signif_feats, target, test_size=0.2, random_state=42)

# Initialize the Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.write("Mean Squared Error:", mse)


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, c='blue', alpha=0.5)

# Add labels and title
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Scatter Plot of Actual vs. Predicted Values')

# Add a diagonal line for reference
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

# Display the plot using Streamlit
st.pyplot(fig)
