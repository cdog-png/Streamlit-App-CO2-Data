import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
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

# Display the plot using Streamlit
st.pyplot(fig)

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

# Step 14: Create continent select box and filter DataFrame
continent = st.selectbox('Select a continent', list(continents.keys()))
continent_countries = continents[continent]
filt_df = df[df['country'].isin(continent_countries)]

# Create a Streamlit line chart for CO2 per capita emissions by continent
pivot_df = filt_df.pivot(index='year', columns='country', values='co2_per_capita')
st.line_chart(pivot_df, use_container_width=True)

# Display the modified DataFrame
st.write("DataFrame for 2018 with continent column:")
st.write(df_2018)

asia = continents['asia']
europe = continents['europe']
north_america = continents['north_america']
south_america = continents['south_america']
africa = continents['africa']
oceania = continents['oceania']

# Step 15: Compute mean share of global oil CO2 emissions by continent (2018)
if 'continent' in df_2018.columns:
    co2_by_continent = df_2018.groupby('continent')['share_global_oil_co2'].mean()
    # Create bar plot
    fig = plt.figure(figsize=(8, 6))
    co2_by_continent.plot(kind='bar', color='blue')
    plt.xlabel('Continent')
    plt.ylabel('Share of global oil CO2 emissions')
    plt.title('Mean share of global oil CO2 emissions by continent in 2018')
    st.image(fig)
else:
    st.write("The 'continent' column is not available in the DataFrame.")


fig = go.Figure(data=[go.Bar(x=top_20_countries.index, y=top_20_countries.values)])
fig.update_layout(
    xaxis_title='Country',
    yaxis_title='Per capita CO2 emissions in 1990',
    title='Per capita CO2 emissions by country in 1990',
    xaxis_tickangle=-45
)
st.plotly_chart(fig)

# Step 19: Show relationship between CO2 per capita and GDP per capita in 2018
fig = px.scatter(df_2018, x='gdp_per_capita', y='co2_per_capita', trendline='ols')
fig.update_layout(
    xaxis_title='GDP per capita',
    yaxis_title='CO2 per capita',
    title='Relationship between GDP per capita and CO2 per capita (2018)'
)
st.plotly_chart(fig)

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
