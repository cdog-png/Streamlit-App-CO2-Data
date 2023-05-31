#step 1 import applications
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

#Step 2: Import Dataset and insert text
st.title("GLOBAL CO2 EMISSION DATA ANALYSIS")
st.write('\n')

df = pd.read_csv('owid-co2-data.csv')
st.subheader("1) The Dataset")
st.header("Initial Dataset")
left_column, right_column = st.beta_columns(2)

with left_column:
    st.write("Summary Statistics")
    st.write(df.describe())

with right_column:
    st.write("Raw Data")
    st.write(df.head(8))
    st.write('\n')

# Step 3: Drop rows with null values and display info
df = df.drop(['co2_including_luc_per_unit_energy', 'consumption_co2', 
              'consumption_co2_per_capita', 'consumption_co2_per_gdp', 'cumulative_other_co2', 'energy_per_capita', 
              'energy_per_gdp', 'ghg_excluding_lucf_per_capita', 'ghg_per_capita', 'methane', 'methane_per_capita', 
              'nitrous_oxide', 'nitrous_oxide_per_capita', 'other_co2_per_capita', 'other_industry_co2', 
              'share_global_cumulative_other_co2', 'share_global_other_co2', 'total_ghg', 'total_ghg_excluding_lucf', 
              'trade_co2', 'trade_co2_share', 'primary_energy_consumption'], axis = 1)


df = df.dropna(how='any', axis=0)
st.header("DataFrame with dropped null values:")
left_column, right_column = st.beta_columns(2)

with left_column:
    st.write("Summary Statistics")
    st.write(df.describe())

with right_column:
    st.write("Raw Data")
    st.write(df.head(8))


# Step 4: Plot CO2 per capita emissions 1950-2018
countries = ['China', 'United States', 'United Kingdom', 'Germany', 'Spain', 'India', 'Brazil', 'Russia', 'France', 'Japan', 'South Korea']
filtered_df = df[(df['country'].isin(countries)) & (df['year'].between(1950, 2018))]
pivot_df = filtered_df.pivot(index='year', columns='country', values='co2_per_capita')

chart_data = pd.DataFrame(pivot_df.values, columns=pivot_df.columns, index=pivot_df.index)
chart_data.index.name = 'Year'
chart_data.columns.name = 'Country'
chart = go.Figure()

for column in chart_data.columns:
    chart.add_trace(go.Scatter(x=chart_data.index, y=chart_data[column], name=column))

chart.update_layout(
    xaxis=dict(title='Year'),
    yaxis=dict(title='CO2 per capita emissions'))

st.write('\n')
st.write('\n')
st.subheader("2) Visualizations")
st.header("CO2 per capita Emissions 1950 - 2018")
st.plotly_chart(chart)

# Step 5: Plot top countries
grouped_data = df.groupby('country')['co2_per_capita'].sum()
top_20_countries = grouped_data.nlargest(20)

st.header("Top 20 Countries by CO2 per capita")
st.bar_chart(top_20_countries, use_container_width=True)

# Step 6: Create GDP per capita variable
df['gdp_per_capita'] = df['gdp'] / df['population']

# Step 7: Create df_2018 DataFrame for 2018 data
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

st.header("Relationship between GDP per capita and CO2 per unit energy (2018)")
st.plotly_chart(fig)

# Step 8: Print unique countries and create continental lists
unique_countries = df['country'].unique()
continents = {
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

st.header("Historic CO2 per capita countries by continent")
continent = st.selectbox('Select a continent', list(continents.keys()))
continent_countries = continents[continent]
filt_df = df[df['country'].isin(continent_countries)]

# Create a Streamlit line chart for CO2 per capita emissions by continent
pivot_df = filt_df.pivot(index='year', columns='country', values='co2_per_capita')

# Adjust x-axis labels to display year numbers
pivot_df.index = pivot_df.index.astype(str)

st.line_chart(pivot_df, use_container_width=True)

# Step 9: Show relationship between CO2 per capita and GDP per capita in 2018
fig = px.scatter(df_2018, x='gdp_per_capita', y='co2_per_capita', trendline='ols')
fig.update_layout(
    xaxis_title='GDP per capita',
    yaxis_title='CO2 per capita',
)
st.header("Relationship CO2 per capita and GDP per capita in 2018")
st.plotly_chart(fig)

filtered_df = df[(df['country'].isin(countries)) & (df['year'].isin([2018, 1990]))]

# Step 10: CO2 per capita 1990 vs 2018
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
st.header("CO2 per capita 1990 vs 2018")
st.plotly_chart(fig)

# Step 11: Data Processing
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
df_standardized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Normalize data
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Display Processed Dataset summaries
st.write('\n')
st.write('\n')
st.subheader("3) Data Analysis")
st.header("Data Processing")

left_column, right_column = st.beta_columns(2)

with left_column:
    st.write("Standardized Data Summary")
    st.write(df_standardized.describe())

with right_column:
    st.write("Normalized Data Summary")
    st.write(df_normalized.describe())


#Step 12: Model 1
target = df_normalized['co2_per_capita']
data = df_normalized.drop('co2_per_capita', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)
slr = LinearRegression()
slr.fit(X_train, y_train)

#12.1 Desciptiond and scores
st.header("Model 1 - 31 Variables")
left_column, right_column = st.beta_columns(2)
with left_column:
    st.write("Multiple Regression")
    st.write("Target Variable: CO2 per Capita")
    st.write("Explanatory variables: df_normalized - 31 variables")

# Second three points in the right column
with right_column:
    st.write("Training score:", slr.score(X_train, y_train))
    st.write("Cross-validation score:", cross_val_score(slr, X_train, y_train).mean())
    st.write("Test score:", slr.score(X_test, y_test))

# 12.2 Correlation and Coefficient list
correlation_list = data.corrwith(target).sort_values(ascending=True)
coefficient_list = pd.DataFrame({'Variable': data.columns, 'Coefficient': slr.coef_})
coefficient_list = coefficient_list.sort_values(by='Coefficient', ascending=False)

left_column, right_column = st.beta_columns(2)

# Left column - Correlation list
with left_column:
    st.write("CO2 per capita Correlation list")
    st.write(correlation_list)

# Right column - Coefficient list
with right_column:
    st.write("Coefficient list")
    coefficient_list_display = coefficient_list[['Variable', 'Coefficient']]
    st.write(coefficient_list_display)

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

signif_feats = df_normalized[['co2_per_gdp', 'oil_co2_per_capita', 'cement_co2_per_capita', 'gas_co2_per_capita','co2_including_luc_per_capita']]

#Step 13: Model 2
X_train, X_test, y_train, y_test = train_test_split(signif_feats, target, test_size=0.2, random_state=789)
slr = LinearRegression()
slr.fit(X_train, y_train)

st.header("Model 2 - 6 most correlated variables")

left_column, right_column = st.beta_columns(2)
with left_column:
    st.write("Multiple Regression")
    st.write("Target Variable: CO2 per Capita")
    st.write("Explanatory variables: df_normalized - 31 variables")

# Second three points in the right column
with right_column:
    st.write("Training score:", slr.score(X_train, y_train))
    st.write("Cross-validation score:", cross_val_score(slr, X_train, y_train).mean())
    st.write("Test score:", slr.score(X_test, y_test))

# VIF (Variance Inflation Factor)
X_train_with_constant = sm.add_constant(X_train)
vif = pd.DataFrame()
vif["Feature"] = X_train_with_constant.columns
vif["VIF"] = [variance_inflation_factor(X_train_with_constant.values, i) for i in range(X_train_with_constant.shape[1])]
st.write("Variance Inflation Factor (VIF):")
st.write(vif)

fig = px.scatter_matrix(df_normalized[columns], title="Pairplot", dimensions=columns)
fig.update_layout(width=900, height=900)  # Adjust the width and height as per your preference

st.plotly_chart(fig)

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
