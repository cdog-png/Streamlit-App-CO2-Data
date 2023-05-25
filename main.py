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
st.title("Owid CO2 Initial Dataset")
st.write(df.head(20))

# Step 3: Drop rows with null values and display info
df = df.dropna(how='any', axis=0)
st.title("DataFrame with dropped null values:")
st.write(df.head(20))

# Step 9: Plot CO2 per capita emissions
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2_per_capita')

# Create a Streamlit line chart for CO2 per capita emissions
chart_data = pd.DataFrame(pivot_df.values, columns=pivot_df.columns, index=pivot_df.index)
chart_data.index.name = 'Year'
chart_data.columns.name = 'Country'
chart = go.Figure()

for column in chart_data.columns:
    chart.add_trace(go.Scatter(x=chart_data.index, y=chart_data[column], name=column))

chart.update_layout(
    xaxis=dict(title='Year'),
    yaxis=dict(title='CO2 per capita Emissions')
)

# Display the chart
st.title("CO2 per capita Emissions")
st.plotly_chart(chart)

# Step 10: Plot top countries
grouped_data = df.groupby('country')['co2_per_capita'].sum()
top_20_countries = grouped_data.nlargest(20)

# Create a Streamlit bar chart for top countries
st.subheader("Top 20 Countries by CO2 per capita")
st.bar_chart(top_20_countries, use_container_width=True)

# Step 11: Create GDP per capita variable
df['gdp_per_capita'] = df['gdp'] / df['population']

# Step 12: Create df_2018 DataFrame for 2018 data
df_2018 = df[df['year'] == 2018]

fig = go.Figure(data=go.Scatter(
    x=df_2018['gdp_per_capita'],
    y=df_2018['co2_per_unit_energy'],
    mode='markers',
    marker=dict(color='blue'),
    line=dict(color='blue'),
    name='Data',
))

fig.update_layout(
    xaxis=dict(title='GDP per capita'),
    yaxis=dict(title='CO2 per unit energy')
)

# Display the graph
st.title("Relationship between GDP per capita and CO2 emissions (2018)")
st.plotly_chart(fig)

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
st.title("CO2 per capita by continent")
continent = st.selectbox('Select a continent', list(continents.keys()))
continent_countries = continents[continent]
filt_df = df[df['country'].isin(continent_countries)]

# Create a Streamlit line chart for CO2 per capita emissions by continent
pivot_df = filt_df.pivot(index='year', columns='country', values='co2_per_capita')
st.line_chart(pivot_df, use_container_width=True)

asia = continents['asia']
europe = continents['europe']
north_america = continents['north_america']
south_america = continents['south_america']
africa = continents['africa']
oceania = continents['oceania']

# Step 19: Show relationship between CO2 per capita and GDP per capita in 2018
fig = px.scatter(df_2018, x='gdp_per_capita', y='co2_per_capita', trendline='ols')
fig.update_layout(
    xaxis_title='GDP per capita',
    yaxis_title='CO2 per capita',
)
st.title("Relationship CO2 per capita and GDP per capita in 2018")
st.plotly_chart(fig)

filtered_df = df[(df['country'].isin(countries)) & (df['year'].isin([2018, 1990]))]

# Create separate traces for each year
traces = []
for year in [2018, 1990]:
    trace = go.Scatter(
        x=filtered_df[filtered_df['year'] == year]['country'],
        y=filtered_df[filtered_df['year'] == year]['co2_per_capita'],
        mode='lines',
        name=str(year)
    )
    traces.append(trace)

# Create the layout for the graph
layout = go.Layout(
    xaxis=dict(title='Country'),
    yaxis=dict(title='CO2 per Capita')
)

# Create the figure
fig = go.Figure(data=traces, layout=layout)

# Display the graph
st.title("CO2 per capita 1990 vs 2018")
st.plotly_chart(fig)

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
st.title("Correlation List CO2 per capita")
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
fig.update_layout(xaxis_title="Predicted Values", yaxis_title="Actual Values")
st.subheader("Predicted vs Actual Values")
st.plotly_chart(fig)

# Scatter plot of residuals vs actual values (train set)
fig = go.Figure(data=go.Scatter(x=y_train, y=residuals, mode='markers', marker=dict(size=7)))
fig.add_shape(type="line", x0=y_train.min(), y0=0, x1=y_train.max(), y1=0, line=dict(color='#0a5798', width=3))
fig.update_layout(xaxis_title="Actual Values", yaxis_title="Residuals")
st.subheader("Residuals vs Actual Values")
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
