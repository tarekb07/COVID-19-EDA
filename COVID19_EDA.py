#!/usr/bin/env python
# coding: utf-8

# # <center> Covid-19 Global Datasets - Exploratory Data Analysis 
#     
# The aim of this project is to perform exploratory data analysis (EDA) by analysing the data of countries effected by Covid-19.
# 
# We are interested to explore the following questions:
# - (Q-1) Is the Delta variant of Covid-19 highly transmissible?
# - (Q-2) Does it result in more deaths?
# - (Q-3) What is the global trend of Covid-19 vaccinations?
# - (Q-4) Is there any correlation between economic variables and the spread of Covid-19?
# - (Q-5) What is the situation in Australia compared to countries with high vaccinations rate like Israel?
#     
# We will carry out the analysis using two datasets from [Our World in Data (OWID)](https://ourworldindata.org/). 
# OWID is a scientific online publication that focuses on large global problems. The OWID datasets are comprehensive datasets and these datasets get updated daily.  
# 
# - Our World in Data - COVID-19 Dataset [Link](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv) 
# 
# - Our World in Data - Vaccinations Dataset [Link](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv) 
#     
# 
# ### Workflow
#     
# #### Part 1
# - Import Libraries
# - Create Functions
# - Import Datasets 
#     
# #### Part 2
# - Data Cleaning 
# - Data Manipulation 
#     
# #### Part 3
# - Exploratory Data Analysis and Visualisation
# - Conclusion

# # Part 1 

# ## Import Libraries

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', 100)
#pd.set_option('display.width', 1000)
#pd.set_option('display.max_rows', 50)
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns


# ## Create Functions

# In[2]:


# Read files from source
def read_file(url):
    r = requests.get(url).content
    r_file = pd.read_csv(io.StringIO(r.decode('utf-8')))
    return r_file

################################

# Look up country by name
def show_country(country_name):
    return total_cases.loc[total_cases['Country'] == country_name]

###############################

# Function to display total stats on a dashboard by country as input
def total_stats(country_name):
    x = total_merged_rename.copy()
    x = x.loc[x['Country'] == country_name]
    x['% Partially Vaccinated'] = (x['1st Dose'] / x['Population'])*100
    x['% Fully Vaccinated'] = (x['2nd Dose'] / x['Population'])*100
    x_new_cases = x[['New Cases']].sum()
    x_new_deaths = x[['New Deaths']].sum()
    x_total_cases = x[['Total Cases']].sum()
    x_total_deaths = x[['Total Deaths']].sum()
    x_1st_vax = x[['1st Dose']].sum()
    x_2nd_vax = x[['2nd Dose']].sum()
    x_part_vax = x[['% Partially Vaccinated']].sum()
    x_full_vax = x[['% Fully Vaccinated']].sum()
    x_vax_rate = x[['Vaccinations Rate']].sum()
    x_daily_vax = x[['Daily Vaccinations']].sum()

    fig = go.Figure()
    fig.update_layout(title= '{}: Covid-19 Statistics'.format(country_name))
    fig.add_trace(go.Indicator(
        mode='number', 
        value= int(x_new_cases), 
        number= {'valueformat': '0,f'}, 
        title= {'text': 'New Cases'},
        domain = {'row': 0, 'column': 0}))

    fig.add_trace(go.Indicator(
        mode='number', 
        value= int(x_new_deaths), 
        number= {'valueformat': '0,f'}, 
        title= {'text': 'New Lives Lost'},
        domain = {'row': 0, 'column': 1}))

    fig.add_trace(go.Indicator(
        mode='number', 
        value= int(x_total_cases), 
        number= {'valueformat': '0,f'}, 
        title= {'text': 'Total Cases'},
        domain = {'row': 2, 'column': 0}))

    fig.add_trace(go.Indicator(
        mode='number', 
        value= int(x_total_deaths), 
        number= {'valueformat': '0,f'}, 
        title= {'text': 'Total Mortality'},
        domain = {'row': 2, 'column': 1}))

    fig.add_trace(go.Indicator(
        mode='number', 
        value= int(x_1st_vax), 
        number= {'valueformat': '0,f'}, 
        title= {'text': 'Partially Vaccinated'},
        domain = {'row': 4, 'column': 0}))

    fig.add_trace(go.Indicator(
        
        mode='number', 
        value= float(x_part_vax), 
        number= {'valueformat': '0.2f'}, 
        title= {'text': 'Partially Vaccinated %'},
        domain = {'row': 4, 'column': 1}))

    fig.add_trace(go.Indicator(
        mode='number', 
        value= int(x_2nd_vax), 
        number= {'valueformat': '0,f'}, 
        title= {'text': 'Fully Vaccinated'},
        domain = {'row': 6, 'column': 0}))

    fig.add_trace(go.Indicator(
        mode='number', 
        value= float(x_full_vax), 
        number= {'valueformat': '0.2f'}, 
        title= {'text': 'Fully Vaccinated %'},
        domain = {'row': 6, 'column': 1}))
    
    fig.add_trace(go.Indicator(
        mode='number', 
        value= int(x_daily_vax), 
        number= {'valueformat': '0,f'}, 
        title= {'text': 'Daily Vaccinations'},
        domain = {'row': 8, 'column': 0}))
     
    fig.add_trace(go.Indicator(
        mode='number', 
        value= float(x_vax_rate), 
        number= {'valueformat': '0.2f'}, 
        title= {'text': 'Vaccinations Rate %'},
        domain = {'row': 8, 'column': 1}))

    fig.update_layout(grid={'rows': 10, 'columns': 2, 'pattern': 'independent'})
    fig.show()

###################################

# Create a single function to perform time-series analysis and 
# plot chart of cumulative data for ALL variable vs country at once
def plot_cumulative(country_name):
    fig = px.line(raw_data_rename[raw_data_rename['Country'] == country_name], 
    x = 'Date', y = 'New Cases', 
    title = '{}: Cumulative New Reported Cases'.format(country_name))
    fig.show()
    
    fig = px.line(raw_data_rename[raw_data_rename['Country'] == country_name], 
    x = 'Date', y = 'New Deaths', 
    title = '{}: Cumulative New Reported Mortality'.format(country_name))
    fig.show()
    
    fig = px.line(raw_data_rename[raw_data_rename['Country'] == country_name], 
    x = 'Date', y = 'Total Cases', 
    title = '{}: Cumulative Reported Total Cases'.format(country_name))
    fig.show()
    
    fig = px.line(raw_data_rename[raw_data_rename['Country'] == country_name], 
    x = 'Date', y = 'Total Deaths', 
    title = '{}: Cumulative Reported Total Mortality'.format(country_name))
    fig.show()

    fig = px.line(raw_vax_rename[raw_vax_rename['Country'] == country_name], 
    x = 'Date', y = 'Daily Vaccinations', 
    title = '{}: Cumulative Reported Daily Vaccinations'.format(country_name))
    fig.show()
    
    fig = px.line(raw_vax_rename[raw_vax_rename['Country'] == country_name], 
    x= 'Date', y = 'Vaccinations Rate', 
    title = '{}: Cumulative Vaccinations Rate'.format(country_name))
    fig.show()

    fig = px.line(raw_vax_rename[raw_vax_rename['Country'] == country_name], 
    x = 'Date', y = 'Total Vaccinations', 
    title = '{}: Cumulative Reported Total Vaccinations'.format(country_name))
    fig.show()

    fig = px.line(raw_vax_rename[raw_vax_rename['Country'] == country_name], 
    x= 'Date', y = '1st Dose', 
    title = '{}: Cumulative Reported First Dose Vaccinations'.format(country_name))
    fig.show()
    
    fig = px.line(raw_vax_rename[raw_vax_rename['Country'] == country_name], 
    x = 'Date', y = '2nd Dose', 
    title = '{}: Cumulative Reported 2nd Dose Vaccinations'.format(country_name))
    fig.show()
    
    fig = px.line(raw_vax_rename[raw_vax_rename['Country'] == country_name], 
    x = 'Date', y = 'Total Boosters', 
    title = '{}: Cumulative Reported Vaccine Booster'.format(country_name))
    fig.show()


# ## Import and Read files 

# ### 1- OWID COVID-19 - Dataset

# In[3]:


#Read file
raw_data = read_file('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
raw_data


# In[4]:


#Display info
raw_data.info()


# In[5]:


#Inpsect null values 
raw_data.isnull().sum()


# #### Observation
# -	The OWID dataset consist of over 64 columns 119,000 rows and growing, as it continues to be modified. 
# -	Majority of the data type is float and some are strings.
# -	Looking at columns with null values, we can see that there is a significant amount of null in many important columns needed for this analysis such as vaccinations columns. 
# -	This will impact on our analysis especially when we want to examine the rate of vaccination per country and the rate of 1st and 2nd doses of Covid vaccine. 
# -	As this data is missing important values, we will input a separate vaccination data set in which we will clean and later merge together. 
# 

# ### 2- OWID Vaccinations - Dataset

# In[6]:


#Read file
raw_vax = read_file('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv')
raw_vax


# In[7]:


#Display info
raw_vax.info()


# In[8]:


#Inspect null value
raw_vax.isnull().sum()


# #### Observation
# - The OWID vaccinations dataset consist of 14 columns and over 50,000 rows and 14 columns and growing as it continues to be modified.
# - Majority of the data type is float and few object strings.
# - Looking at columns with null values, we can see that there are null values present in this dataset.
# - Also, as we are going to be merging this dataframe with raw_data dataframe. From a quick observation I can see that the location columns in raw_data dataframe can act as a key to match the dataset with vax_raw location column. We will rename the location column : 'Country' before the merge.   

# # Part 2 - Data Cleaning and Manipulation

# ### 1- OWID COVID-19 - Dataset

# #### Drop unwanted rows and columns 

# The raw_data file consist of rows relating to data per continent, as the aim of this analysis to look at an overview of countries, I decided to drop these rows below.
# Next, I will remove unwanted columns.

# In[9]:


# Remove unwanted rows
raw_data = raw_data[raw_data.iso_code != 'OWID_AFR']
raw_data = raw_data[raw_data.iso_code != 'OWID_ASI']
raw_data = raw_data[raw_data.iso_code != 'OWID_EUR']
raw_data = raw_data[raw_data.iso_code != 'OWID_EUN']
raw_data = raw_data[raw_data.iso_code != 'OWID_INT']
raw_data = raw_data[raw_data.iso_code != 'OWID_NAM']
raw_data = raw_data[raw_data.iso_code != 'OWID_OCE']
raw_data = raw_data[raw_data.iso_code != 'OWID_SAM']
raw_data = raw_data[raw_data.iso_code != 'OWID_WRL']
raw_data = raw_data[raw_data.location != 'Lower middle income']
raw_data = raw_data[raw_data.location != 'High income']
raw_data = raw_data[raw_data.location != 'Upper middle income']


# In[10]:


# Remove unwanted columns 
raw_data_drop = raw_data.drop(columns = ['iso_code', 'continent', 'new_cases_smoothed', 'new_deaths_smoothed', 
            'reproduction_rate', 'new_vaccinations_smoothed', 'new_vaccinations_smoothed_per_million', 
            'new_deaths_smoothed_per_million','new_cases_smoothed_per_million', 'new_tests_smoothed',
            'new_tests_smoothed_per_thousand', 'total_vaccinations','people_vaccinated','people_fully_vaccinated',
            'total_boosters', 'new_vaccinations','total_vaccinations_per_hundred','people_vaccinated_per_hundred',
            'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred','icu_patients', 'icu_patients_per_million', 
            'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions', 'weekly_icu_admissions_per_million', 
            'weekly_hosp_admissions', 'weekly_hosp_admissions_per_million', 'new_tests','total_tests',
            'total_tests_per_thousand', 'new_tests_per_thousand', 'positive_rate', 'tests_per_case',
            'excess_mortality_cumulative', 'excess_mortality','tests_units', 'female_smokers', 'male_smokers',
            'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'excess_mortality_cumulative_absolute', 
            'new_people_vaccinated_smoothed', 'new_people_vaccinated_smoothed_per_hundred', 'excess_mortality_cumulative_per_million'])
                                                                               
raw_data_drop.columns


# #### Rename Columns

# In[11]:


#human_development_index = HDI
#gdp_per_capita = GDP
#diabetes_prevalence = Diabetes prev
raw_data_rename = raw_data_drop.rename(columns={'location': 'Country', 'date': 'Date',
                            'total_cases': 'Total Cases','new_cases': 'New Cases', 'new_deaths': 'New Deaths', 'total_deaths': 'Total Deaths',  
                            'total_cases_per_million': 'Total Cases per million','total_deaths_per_million': 'Total Deaths per million',
                            'new_deaths_per_million': 'New Deaths per million ',                    
                            'new_cases_per_million': 'New Cases per million',
                            'stringency_index': 'Stringency Index','population': 'Population', 
                            'population_density': 'Population Density','median_age': 'Median Age',
                            'aged_65_older': '65 or Older', 'aged_70_older': '70 or Older',
                            'gdp_per_capita': 'GDP', 'human_development_index': 'HDI', 'extreme_poverty': 'Extreme Poverty', 'cardiovasc_death_rate': 'Cardiovasc Death Rate',
                            'diabetes_prevalence': 'Diabetes Prev'})                                                 
                                                     


# In[12]:


raw_data_rename.head()


# #### Fill missing value

# In[13]:


# drop all NaN and repalce with (0)
raw_data_rename.fillna(0, inplace = True)


# In[14]:


raw_data_rename.sample(10)


# #### Convert Date to datetime 

# In[15]:


# Convert date from str object to datetime
raw_data_rename['Date'] = pd.to_datetime(raw_data_rename['Date']) 


# #### Combine into Data Structure

# In[16]:


# Sort values by date, then groupby and aggregate then sum date and country,
# then groupby and aggregate by country and last() date input, reset index and drop duplicates 
raw_data_rename.sort_values(by = 'Date', inplace = True) #sort values by date

raw_data_sort = raw_data_rename.groupby(['Date', 'Country'],as_index = False).sum().groupby(['Country']).last().reset_index().drop_duplicates()
raw_data_sort.head(5)


# ### 2- OWID  Vaccinations - Dataset 

# #### Drop unwanted rows and columns 

# Similar to raw_data df, The raw_vax df consist of rows relating to data per continent, as the aim of this analysis to look at an overview of countries, I decided to drop these rows below.
# Next, I will remove unwanted columns.

# In[17]:


# Remove unwanted rows
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_AFR']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_EUR']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_EUN']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_INT']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_HIC']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_LIC']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_LMC']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_NAM']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_OCE']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_SAM']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_SAM']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_UMC']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_WRL']
raw_vax = raw_vax[raw_vax.iso_code != 'OWID_ASI']
raw_vax.head()


# #### Drop Columns

# In[18]:


# Drop unwanted columns 
raw_vax_drop = raw_vax.drop(columns = ['iso_code','daily_vaccinations_raw', 
            'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred', 
            'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred', 
            'daily_vaccinations_per_million', 'daily_people_vaccinated', 'daily_people_vaccinated_per_hundred'])


# #### Rename Columns

# In[19]:


# Rename columns 
raw_vax_rename = raw_vax_drop.rename(columns={'location': 'Country','date': 'Date', 'total_vaccinations': 'Total Vaccinations',
       'people_vaccinated': '1st Dose', 'people_fully_vaccinated': '2nd Dose', 'total_boosters': 'Total Boosters',
       'daily_vaccinations': 'Daily Vaccinations'})
raw_vax_rename.head()


# #### Convert Date to datetime, Total Vaccinations and Daily Vaccinations to float

# In[20]:


# Convert Date from str object to datetime
raw_vax_rename['Date'] = pd.to_datetime(raw_vax_rename['Date'])


# In[21]:


# Covert str objects to float 
raw_vax_rename['Total Vaccinations'] = pd.to_numeric(raw_vax_rename['Total Vaccinations'], downcast= 'float')
raw_vax_rename['Daily Vaccinations'] = pd.to_numeric(raw_vax_rename['Daily Vaccinations'], downcast= 'float')


# In[22]:


raw_vax_rename.dtypes


# In[23]:


# Create a columns for Vaccinations Rate for time-series analysis
raw_vax_rename['Vaccinations Rate'] = raw_vax_rename['Daily Vaccinations']/raw_vax_rename['Total Vaccinations']*100
raw_vax_rename.head()


# In[24]:


raw_vax_rename.fillna(0, inplace = True)
raw_vax_rename.head()


# #### Combine into Data Structure

# In[25]:


# Sort values by date, then groupby and aggregate then sum date and country,
# then groupby and aggregate by country and last() date input, reset index and drop duplicates 
raw_vax_rename.sort_values(by = 'Date', inplace = True)
raw_vax_sort = raw_vax_rename.groupby(['Date', 'Country'],as_index = False).sum().groupby(['Country']).last().reset_index().drop_duplicates()


# In[26]:


raw_data_sort.head()


# In[27]:


raw_vax_sort.head()


# ## Merge both datasets together

# #### Observation
# So far, we have two cleaned datasets:
# - raw_data_sort (OWID covid19 dataset)
# - raw_vax_sort (OWID Vaccination dataset)
# 
# Next, we will perform a merge using the pd.merg function. The merge will use 'Country' as a key between both datasets.

# In[28]:


# Create a copy of each file then perform pd.merge()
#openprice.merge(wkhigh, on='Symbol')
raw_data = raw_data_sort.copy()
raw_vax = raw_vax_sort.copy()
total_merged = pd.merge(raw_data, raw_vax,left_on = 'Country', right_on= 'Country', how = 'left')


# In[29]:


total_merged.head()


# #### Drop and Rename Date column

# In[30]:


# Rename new columns
total_merged_rename = total_merged.rename(columns={'Date_x': 'Date'})  
total_merged_rename.drop(columns = ['Date_y'], inplace = True)


# In[31]:


total_merged_rename


# #### Observation
# -	We have successfully merged both dataframes together and matched the date with the related country column. In total we have 27 columns 233 rows and over 233 columns.
# -	Finally, we performed random checks of values in  total vaccination, 1st dose and 2nd dose data per country against current reported values and were satisfied that the information we have is accurate. It is also important to note, for example vaccination data for China contained the total vaccinations and it is missing 1st does and 2nd dose information, this may result in inaccurate visual representations.

# ### Create new columns 

# In this section we will create new columns to provide us with more insights:
# - Total Recovered 
# - Recovered Rate
# - Death Rate   
# - % Partially Vaccinated
# - % Fully Vaccinated

# In[32]:


# Make a copy of df
total_cases = total_merged_rename.copy() 

# Create new columns 
total_cases['Total Recovered'] = total_cases['Total Cases'] - total_cases['Total Deaths']
total_cases['Recovered Rate'] = (total_cases['Total Recovered'] / total_cases['Total Cases'])*100

total_cases['Death Rate'] = (total_cases['Total Deaths'] / total_cases['Total Cases'])*100

total_cases['% Partially Vaccinated'] = (total_cases['1st Dose'] / total_cases['Population'])*100
total_cases['% Fully Vaccinated'] = (total_cases['2nd Dose'] / total_cases['Population'])*100


# In[33]:


total_cases.head()


# In[34]:


# replace infinite values with NaN
total_cases.replace([np.inf, -np.inf], np.nan, inplace=True)
# then replace NaN with 0
total_cases.fillna(0, inplace = True)


# In[35]:


# Inspect a country using def show_country(country_name)
aus = show_country('Australia')
aus


# After completing the merge some rows showed 'inf' values and as a result we have replaced these values with NaN and later with '0'.

# # Part 3 - Data Exploration and Visualisation  

# ### Summary Statistics

# In[36]:


total_cases.describe().round()


# ### Observation
# - As both datasets updates every few hours, negative value can occur which indicated an adjustment had taken place, negative value may require further investigation, and in most cases, it not unsual. 
# - The max values for the numeric data are far off from the 75 percentile which could indicate that we may have outliers.

# ## Total  Summary Dashboard - Global

# In[37]:


# Create a new variable for each column to sum up total 
covid_cases = total_cases[['Total Cases']].loc[0:].sum()
covid_deaths = total_cases[['Total Deaths']].loc[0:].sum()
covid_recovered = total_cases[['Total Recovered']].loc[0:].sum()
covid_1st_vax = total_cases[['1st Dose']].loc[0:].sum()
covid_2nd_vax = total_cases[['2nd Dose']].loc[0:].sum()
covid_total_vax = total_cases[['Total Vaccinations']].loc[0:].sum()

# Create total for each column and display as numbers
fig = go.Figure()
fig.update_layout(title= 'Global Covid-19 Statistics')
fig.add_trace(go.Indicator(
    delta = {'reference': 228180749},
    mode='number + delta', 
    value= int(covid_cases), 
    number= {'valueformat': '0,f'}, 
    title= {'text': 'Global Cases'},
    domain = {'row': 0, 'column': 0}))

fig.add_trace(go.Indicator(
    delta = {'reference': 4685823},
    mode='number + delta', 
    value= int(covid_deaths), 
    number= {'valueformat': '0,f'}, 
    title= {'text': 'Global Morality'},
    domain = {'row': 1, 'column': 0}))

fig.add_trace(go.Indicator(
    delta = {'reference': 223494926},
    mode='number + delta', 
    value= int(covid_recovered), 
    number= {'valueformat': '0,f'}, 
    title= {'text': 'Global Recovered'},
    domain = {'row': 2, 'column': 0}))

fig.add_trace(go.Indicator(
    delta = {'reference': 2268378273},
    mode='number + delta', 
    value= int(covid_1st_vax), 
    number= {'valueformat': '0,f'}, 
    title= {'text': 'Partially Vaccinated'},
    domain = {'row': 0, 'column': 1}))

fig.add_trace(go.Indicator(
    delta = {'reference': 1458113528},
    mode='number + delta', 
    value= int(covid_2nd_vax), 
    number= {'valueformat': '0,f'}, 
    title= {'text': 'Fully Vaccinated'},
    domain = {'row': 1, 'column': 1}))

fig.add_trace(go.Indicator(
    delta = {'reference': 5903889408},
    mode='number + delta', 
    value= int(covid_total_vax), 
    number= {'valueformat': '0,f'}, 
    title= {'text': 'Global Vaccinations'},
    domain = {'row': 2, 'column': 1}))

fig.update_layout(grid={'rows': 3, 'columns': 2, 'pattern': 'independent'})
fig.show()


# ## Analysis and Visualisation - Global

# In[38]:


# Generate total summary df and plto in a bar graph
cases = total_cases[['Total Cases', 'Total Deaths', 'Total Recovered']].loc[0:].sum() # sum up all cases 
cases_df = pd.DataFrame(cases).reset_index() # create a new df to show total cases/deaths/recovered summary
cases_df.columns = ['Status','Total'] #rename index columns
cases_df


# In[39]:


#plot bar graph of total summary 
fig = px.bar(cases_df, x = 'Status', y= 'Total', 
             title = 'Total Cases vs Total Lives Lost vs Total Recovered', 
             text = 'Total', hover_data=['Status', 'Total'])
fig.update_traces(texttemplate='%{text:.3s}', textposition='auto')

fig.show()


# In[40]:


# Generate  summary of new cases V new deaths of max date, & plot graph
cases = total_cases[['New Cases','New Deaths']].loc[0:].sum() # sum up all cases 
cases_df = pd.DataFrame(cases).reset_index() # create a new df to show total cases/deaths/recovered summary
cases_df.columns = ['Status','Total'] #rename index columns
cases_df


# In[41]:


#plot bar chart 
fig = px.bar(cases_df, x ='Status', y = 'Total', 
             title = 'Total New Cases vs Total New Lives Lost', 
             text = 'Total', hover_data=['Status', 'Total'])
fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')

fig.show() 


# #### Observation
# Looking at the three bar graphs which display worldwide Covid-19 data of total and new cases and deaths. We can see that although the number of cases is significantly high, the total number of deaths is significantly low. The main concern is the high number of new cases which can be related to the new Covid-19 Delta variant. On the other hand, there is a positive sign that people are recovering.

# In[42]:


# Generate Total summary & plot graph
vax = total_cases[['1st Dose', '2nd Dose', 'Total Vaccinations']].loc[0:].sum() # sum up all cases

vax_df = pd.DataFrame(vax).reset_index() # create a new df to show total vax
vax_df.columns = ['Status','Total'] #rename index columns
vax_df


# In[43]:


fig = px.bar(vax_df, x ='Status', y ='Total', 
             title = '1st Dose vs 2nd Dose vs Total Vaccinations', 
             hover_data=['Status', 'Total'])
fig.show()


# #### Observation 
# The bar graph shows the total number of vaccinations including 1st dose and fully vaccinated people worldwide. 
# Overall, we can observe that total vaccination exceeded 6 billion vaccines worldwide and it is expected that the number will continue to increase as we progress in time.
# Furthermore, it is important to highlight that China 1st and 2nd dose data was not available which will impact the numbers displayed for 1st and 2nd dose bars, however the total vaccination data was available and therefore that data is accurate. 

# ## Analysis and Visualisation - Comparison by Country 

# In[44]:


# Create a copy of total_cases df and sort values by total deaths, 
# then display 2 plots showing total deaths vs death rate
sort_deaths = total_cases.copy()
sort_deaths = sort_deaths.sort_values(by='Total Deaths', ascending=False).head(30)

fig = go.Figure()
fig = make_subplots(specs = [[{'secondary_y': True}]])
fig.add_trace(go.Bar(
    x= sort_deaths['Country'],
    y= sort_deaths['Total Deaths'],
    name='Total Mortality',
    marker_color='light blue',
    textposition = 'auto'), secondary_y= False)

fig.add_trace(go.Scatter(
    x= sort_deaths['Country'],
    y= sort_deaths['Death Rate'],
    name='% Mortality Rate',
    marker_color='red',
    mode = 'markers+lines',
    text= sort_deaths['Death Rate'].round(2)),
    secondary_y = True)

fig.update_layout(title_text='Total Morality vs Mortality Rate')
fig.show()


# #### Observation
# The following bar graph is showing the total number of deaths against the line graph which is showing the rate of deaths, both sorted by top 30 countries.
# When analysing both graphs we can see that mortality rate is around 2% in most countries, however few countries with high rate of deaths are Peru at 9.2%, Mexico at 7.6% and Ecuador at 6.4% which is significantly high compared to other South American countries such as; Colombia, Argentina and Chile where the rate is below 2.5%.
# 
# News article indicated that in Mexico and Peru there was slow vaccinations roll out, limited hospital capacity and shortage of oxygen supplies, where many patients with moderate COVID symptoms were being sent home. Many would later become severely ill and die at home.

# In[45]:


sort_vax = total_cases.copy()
sort_vax = sort_vax.sort_values(by = '% Fully Vaccinated', ascending=False).head(50)

fig = go.Figure()
fig.add_trace(go.Bar(
    x= sort_vax['Country'],
    y= sort_vax['% Fully Vaccinated'],
    name='% Fully Vaccinated',
    marker_color='light blue',
    text= sort_vax['% Fully Vaccinated'].round(2),
    textposition = 'auto'))
fig.update_layout(title_text='% Fully Vaccinated Globally', autosize=False, width=1100, height=600) 
fig.show()


# ### Observation
# This bar chart shows the percentage of people fully vaccinated per courntry sorted by descending order. Countries such as; Portugal, United Arab Emirates, Spain, Singapore and Chilie are amongst the most vaccinated countries.

# ## Analysis of Relationships Between Variables

# In[46]:


# Create a a new list of varibales for correlation analysis
total_cases_corr = total_cases[['Country', 'Date', 
                'Total Cases', 'New Cases', 'Total Deaths', 
                'New Deaths', 'Total Cases per million','Total Deaths per million', 
                'New Cases per million', 'Total Vaccinations', 
                '1st Dose', '2nd Dose', 'Population', 
                'Population Density', 'Median Age', '65 or Older', 
                '70 or Older', 'GDP','HDI','Extreme Poverty', 
                'Stringency Index', 'Cardiovasc Death Rate',
                'Diabetes Prev', 'Total Recovered', 'Vaccinations Rate']]


# ### Correlation Heatmap

# In[47]:


# matplot a heatmap to show relationships between variables
plt.figure(figsize=(20, 8))
mask = np.triu(np.ones_like(total_cases_corr.corr())) #define the mask to set the values in the upper triangle to True
heatmap = sns.heatmap(total_cases_corr.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='RdYlBu')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
heatmap


# #### Observation:
# - The correlation heatmap shows moderate to low degree of correlation between variables. The darker the colour the stronger the correlation is and vice versa. 
# - It is worth exploring some variables such as Median Age, HDI and GDP vs Total Cases per million, Total Deaths per million and Vaccinations Rate.

# ### Scatter Plots Analysis

# In[48]:


# Create scatter plots for IV and DV in 3x3 format
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(16,12))
                                        
ax1.scatter(total_cases_corr['Median Age'], total_cases_corr['Total Cases per million'])
ax1.title.set_text('Median Age vs Total Cases per million')
ax1.set_xlabel('Median Age')
ax1.set_ylabel('Total Cases per million')

ax2.scatter(total_cases_corr['Median Age'], total_cases_corr['Total Deaths per million'])
ax2.title.set_text('Median Age vs Total Deaths per million')
ax2.set_xlabel('Median Age')
ax2.set_ylabel('Total Deaths per million')

ax3.scatter(total_cases_corr['Median Age'], total_cases_corr['Vaccinations Rate'])
ax3.title.set_text('Median Age vs Vaccinations Rate')
ax3.set_xlabel('Median Age')
ax3.set_ylabel('Vaccinations Rate')

ax4.scatter(total_cases_corr['GDP'], total_cases_corr['Total Cases per million'])
ax4.title.set_text('GDP vs Total Cases per million')
ax4.set_xlabel('GDP')
ax4.set_ylabel('Total Cases per million')

ax5.scatter(total_cases_corr['GDP'], total_cases_corr['Total Deaths per million'])
ax5.title.set_text('GDP vs Total Deaths per million')
ax5.set_xlabel('GDP')
ax5.set_ylabel('Total Deaths per million')

ax6.scatter(total_cases_corr['GDP'], total_cases_corr['Vaccinations Rate'])
ax6.title.set_text('GDP vs Vaccinations Rate')
ax6.set_xlabel('GDP')
ax6.set_ylabel('Vaccinations Rate')

ax7.scatter(total_cases_corr['HDI'], total_cases_corr['Total Cases per million'])
ax7.title.set_text('HDI vs Total Cases per million')
ax7.set_xlabel('HDI')
ax7.set_ylabel('Total Cases per million')

ax8.scatter(total_cases_corr['HDI'], total_cases_corr['Total Deaths per million'])
ax8.title.set_text('HDI vs Total Deaths per million')
ax8.set_xlabel('HDI')
ax8.set_ylabel('Total Deaths per million')

ax9.scatter(total_cases_corr['HDI'], total_cases_corr['Vaccinations Rate'])
ax9.title.set_text('HDI vs Vaccinations Rate')
ax9.set_xlabel('HDI')
ax9.set_ylabel('Vaccinations Rate')


fig.tight_layout(pad=3) # to adj subplots params to fit the area
plt.show()


# #### Observation:
# The scatter plots used to observe the relationship between number of variables  of interest. For exmaple: 
# - Median age vs Total cases and deaths per million shows moderate positive correlation where median age between 20-40 have higher rates of total cases and deaths recorded per million, whereas Median age vs Vaccinations plot was moderately negative.
# - When looking at the Gross Domestic Product (GDP) relationships, we can observe an unclear due to scattered data but almost low positive relationship with total number of new cases er million, followed by total number of deaths per million where in countries with higher GDP the vaccinations rate is unexpectedly low.
# - Furthermore, in observing the Human Development Index (HDI) relationships with total number of cases, deaths per million and vaccinations rate, we observe an unexpected result where countries with high HDI showed poor performance in dealing with Covid-19 and lower vaccinations rate compared to countries with lower HDI.

# ## Time-Series Analysis of Covid-19:  Australia vs Israel

# ### Overview

# In[49]:


# plot total_stats first
Australia = total_stats('Australia')
Israel  = total_stats('Israel')


# In[50]:


# Select a country for comparison 
Australia = plot_cumulative('Australia')


# #### Observation
# In Australia, the number of reported new cumulative Covid-19 cases was stable and low between November 2020 and July 2021. However, the number of new cases has shown an exponential growth where the growth has become rapidly fast over a short period of time between July 2021 and September 2021.
# 
# We can also observe the same exponential growth in the total number of cases, where the number of cumulative cases stayed stable between September 2020 - July 2021 and later started to grow very rapidly and it is too early to say if we have reached the peak. On the other hand, the mortality rete is significantly lower than last year. 
# 
# The vaccinations also showing a rapid growth in the number of 1st and 2nd dose vaccinations in Australia. This is important to note that upward and downward spikes in the vaccination charts, this variation can be indicative of daily reports not being recorded. This could occur for many reasons such as, data not sent and entered to the dataset due to falling on a weekend, or public holiday.

# In[51]:


Israel  = plot_cumulative('Israel')


# #### Observation
# Israel has conducted the fastest campaign to vaccinate its population against COVID-19. In fact, over 60% of its population were fully vaccinated by July 2021. 
# In analysing the line charts, we can observe that there has been a spike in new Covid-19 and new deaths despite high number of vaccinations in the population. As a result, Israel introduced a booster vaccine dose around August 2021 to try and restore protection from Covid-19 infection. We can also confirm this when looking at the daily vaccinations data which spiked around August in line with the booster shot spike around same the time. 
# It will be interesting to monitor this progress to understand the impact of the booster dose and the level of protection it provides for future outbreaks. 

# ### =========================================================================================
# 
# 
# ## Conclusion

# The aim of this project is to perform exploratory data analysis by analysing the data of countries effected by Covid-19.
# 
# #### (Q-1) The Delta variant of COVID-19 highly transmissible?
# Overall data shows an upward trend and rapid increase of new cases which can be related to the new Covid-19 Delta variant and we can conclude that this variant can be 2x more transmissible than previous types and this is evident in the number of daily cases globally.
# #### (Q-2) Does it result in more deaths compared to other variants?
# On the positive side people are recovering as the rate of death from infection is very low and the numbers are also low compared to last year in 2020. It is still unclear if whether we can conclude that the Delta variant is less deadly and unlikely to cause severe illness. Awareness, hygiene, social distancing, lock downs and high vaccinations could perhaps contribute to drop in the number of deaths.  
# #### (Q-3) What is the global trend of Covid-19 vaccinations?
# With regards to the vaccinations we can observe that the countries worldwide are pushing towards achieving high percentage of full vaccinations of its population. This is to provide a level of protection against Covid-19 infection. Countries such as; Portugal, United Arab Emirates, Spain, Singapore and Chile are amongst the most vaccinated countries.
# #### (Q-4) Is there any correlation between economic variables and the spread of Covid-19?
# When examining relationships between economic variables we found that Median Age, HDI and GDP correlated with Total Cases per million, Total Deaths per million and Vaccinations Rate. This correlation was evident in a number of scatter plots showing various degrees of positive and negative correlations. Some unexpected correlations were evident in countries with higher GDP where the vaccinations rate is unexpectedly low and countries with high HDI showed poorer performance in dealing with Covid-19 and lower vaccinations rate compared to countries with lower HDI. 
# #### (Q-5) What is the situation in Australia compared to countries with high vaccinations rate like Israel?
# When observing time-series analysis, we looked at Australia and Israel. In Australia, the number of new cases is growing rapidly and very fast since July 2021 and over a short period of time, it is too early to say if we have reached the peak. While th rate of mortlity is signifcantly less than what it was 12 months ago. The vaccination rate is also showing a steady daily rate of vaccinations of ~1% and a rapid growth in the number of vaccinations.
# 
# In contrast, Israel has conducted the fastest campaign to vaccinate its population against COVID-19, however we observed a spike in new Covid-19 cases and new deaths since August 2021 despite high number of full vaccinations in the population. As a result, Israel introduced a booster vaccine dose to try and restore protection from Covid-19 infection. 
# 
# Finally, further analysis to investigate the vaccinations status of hospitalised patients and their health status i.e. whether they suffer from systematic condition and/or their immune system is compromised due to medication to better understand how vaccinations is protecting us from becoming severely ill.  
