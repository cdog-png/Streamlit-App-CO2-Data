import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import pandas as pd
from PIL import Image  # Added PIL import
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
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

countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']

filtered_df = df[(df['Country Name'].isin(countries)) & (df['Year'].between(1950, 2018))]
pivot_df = filtered_df.pivot(index='Year', columns='Country Name', values='CO2 per capita')

fig, ax = plt.subplots(figsize=(12, 8))
pivot_df.plot(ax=ax)
ax.grid(True, which='both', linestyle='-', linewidth=0.5)

plt.title('CO2 Per Capita Emissions (1950-2018)')
plt.xlabel('Year')
plt.ylabel('CO2 Per Capita Emissions')

for country in countries:
    if country not in pivot_df.columns:
        continue

    last_value = pivot_df[country].iloc[-1]
    label = country[:3]
    x_pos = pivot_df.index[-1]
    y_pos = last_value

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

ax.legend(prop={'size': 7})
st.pyplot(fig)

# Drop unnecessary columns
df = df.drop(['country', 'year', 'iso_code'], axis=1)

# Calculate quartiles and IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers and replace with percentiles
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df.mask((df < lower_bound) | (df > upper_bound), (Q1 + Q3) / 2, axis=1)

# Display cleaned DataFrame information
st.write("Cleaned DataFrame:")
st.write(df_cleaned.info())

# Standardize data
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Display standardized data summary
st.write("Standardized Data Summary:")
st.write(df_normalized.describe())

# Normalize data
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Display normalized data summary
st.write("Normalized Data Summary:")
st.write(df_normalized.describe())

# Compute correlations
correlation_list = df_normalized.corrwith(df_normalized['co2_per_capita']).drop('co2_per_capita').sort_values(ascending=True)

# Display correlation list
st.header("Correlation List CO2 per capita")
st.write(correlation_list)

# Perform linear regression with significant features
signif_feats = df_normalized[['co2_per_gdp', 'oil_co2_per_capita', 'cement_co2_per_capita', 'gas_co2_per_capita',
                              'co2_including_luc_per_capita']]

X_train, X_test, y_train, y_test = train_test_split(signif_feats, df_normalized['co2_per_capita'], test_size=0.2, random_state=789)
slr = LinearRegression()
slr.fit(X_train, y_train)

# Display coefficients and scores
coeffs = list(slr.coef_)
coeffs.insert(0, slr.intercept_)
feats = ['intercept'] + list(signif_feats.columns)
df_coeffs = pd.DataFrame({'Estimated Value': coeffs}, index=feats)
st.write("Coefficients:")
st.write(df_coeffs)
st.write("Training score:", slr.score(X_train, y_train))
st.write("Cross-validation score:", cross_val_score(slr, X_train, y_train).mean())
st.write("Test score:", slr.score(X_test, y_test))

# Predictions and residuals
pred_test = slr.predict(X_test)
pred_train = slr.predict(X_train)
residuals = pred_train - y_train

# Scatter plot of predicted vs actual values (test set)
fig = go.Figure(data=go.Scatter(x=pred_test, y=y_test, mode='markers'))
fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal Line'))
fig.update_layout(title="Scatter Plot (Test Set)", xaxis_title="Predicted Values", yaxis_title="Actual Values")
st.plotly_chart(fig)

# Scatter plot of residuals vs actual values (train set)
fig = go.Figure(data=go.Scatter(x=y_train, y=residuals, mode='markers', marker=dict(size=7)))
fig.add_shape(type="line", x0=y_train.min(), y0=0, x1=y_train.max(), y1=0, line=dict(color='#0a5798', width=3))
fig.update_layout(title="Scatter Plot (Train Set)", xaxis_title="Actual Values", yaxis_title="Residuals")
st.plotly_chart(fig)

# VIF (Variance Inflation Factor)
X_train_with_constant = sm.add_constant(X_train)
vif = pd.DataFrame()
vif["Feature"] = X_train_with_constant.columns
vif["VIF"] = [variance_inflation_factor(X_train_with_constant.values, i) for i in range(X_train_with_constant.shape[1])]
st.write("Variance Inflation Factor (VIF):")
st.write(vif)

# Pairplot of selected columns
columns = ['co2_per_gdp', 'oil_co2_per_capita', 'cement_co2_per_capita', 'gas_co2_per_capita',
           'co2_including_luc_per_capita', 'co2_per_capita']
fig = px.scatter_matrix(df_normalized[columns], title="Pairplot", dimensions=columns)
st.plotly_chart(fig)

# Linear regression with log-transformed features
signif_feats_log = np.log(signif_feats + 1e-8)
signif_feats_log = pd.DataFrame(signif_feats_log, columns=signif_feats.columns)
lr2 = LinearRegression()
lr2.fit(X_train.loc[:, signif_feats_log.columns], y_train)
train_score = lr2.score(X_train.loc[:, signif_feats_log.columns], y_train)
test_score = lr2.score(X_test.loc[:, signif_feats_log.columns], y_test)
st.write("Linear Regression with Log-Transformed Features:")
st.write("Training score:", train_score)
st.write("Test score:", test_score)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write("Random Forest Regression:")
st.write("Mean Squared Error:", mse)

# Scatter plot of predicted vs actual values (test set) - Random Forest
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Data'))
fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Identity'))
fig.update_layout(xaxis_title="Actual Values", yaxis_title="Predicted Values", title="Scatter Plot of Actual vs. Predicted Values (Random Forest)")
st.plotly_chart(fig)
