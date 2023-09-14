#!/usr/bin/env python
# coding: utf-8

# Unicorn Companies Dataset
# Unicorn companies are private startups valued at over $1 billion as of March 2022, including each company's current valuation,funding,country of origin,industry,select investors and the years they were founded and became unicorns.
# 
# In thi data analysis, I will perform basic exploratory analysis and also create insights which will project recommendationsto help the business models in making decisions that will focus on the high growth potential and also help diversify investment. My analysis will also give an overall recommendationto solve  problems . 
# 
# These problems include;
# 
# Which unicorn companies have had the biggest return on investment?
# How long does it usually take for a company to become a unicorn? Has it always been this way?
# Which countries have the most unicorns? Are there any cities that appear to be industry hubs?
# Which investors have funded the most unicorns?

# In[122]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[123]:


unicorn_companies = pd.read_csv(r'C:\Users\Cartw\Downloads\Unicorn_Companies.csv') 
unicorn_companies


# In[124]:


unicorn_companies.head()


# In[125]:


unicorn_companies.shape


# In[126]:


unicorn_companies.columns


# In[127]:


unicorn_companies.dtypes


# In[128]:


unicorn_companies.info


# In[129]:


unicorn_companies.isna


# In[130]:


unicorn_companies.isnull().sum()


# In[131]:


plt.figure(figsize=(10,5))
sns.heatmap(unicorn_companies.isnull(),cbar=True,cmap='magma_r')
plt.show()


# In[132]:


df=pd.DataFrame(unicorn_companies)
print(df)
df.City.value_counts()


# In[133]:


# Replace 'unknown' with NaN
df['Funding'].replace('unknown', np.nan, inplace=True)

# Drop rows with NaN values
df.dropna(subset=['Funding'], inplace=True)


# In[134]:


df.describe().astype('int')


# In[135]:


##INSIGHTS
unicorn_companies['Year Founded'].mean()



# # Exploratory Data  Analysis

# In[136]:


##UNIVARIATE ANALYSIS= How many companies per country?
count_countries= unicorn_companies['Country'].value_counts().head().sort_values(ascending=False)
count_countries


# In[137]:


count_Industry= unicorn_companies['Industry'].value_counts().head().sort_values(ascending=False)
count_Industry


# In[138]:


ax=count_countries.plot(kind='bar',figsize=(12,8),title='Country Per Company',xlabel='Country',
                ylabel='No Of Company',legend=False)
ax.bar_label(ax.containers[0],label_type='edge')
ax.margins(y=0.1)
plt.show()


# #OBSERVATION==United States has the highest number of companies with a total value of 562 while 
# Germany has the lowest number of company with a value of 26.

# In[139]:


## SUMMARY STATISTICS PER INDUSTRY AND FUNDING
unicorn_companies.groupby('Continent').Valuation.describe()



# In[140]:


count_Industries= unicorn_companies['Industry'].value_counts().head().sort_values(ascending=False)
count_Industries


# In[141]:


ax=count_Industries.plot(kind='barh',figsize=(12,8),title='Industry Per Funding',xlabel='Funding',
                ylabel='Industry',legend=False)
ax.bar_label(ax.containers[0],label_type='edge')
ax.margins(y=0.1)
plt.show()


# #OBSERVATIONS===FINTECH IS THE INDUSTRY WHICH WAS MOST FUNDED WITH THE AMOUNT OF 224  WHILE ARTIFICIAL 
# INTERLIGENCE IS THE INDUSTRY  LEAST FUNDED WITH AN AMOUNT OF 73. 

# In[142]:


## ANALYSIS====WHAT IS THE TOP 5 MOST FREQUENT SELECT INVESTORS?


# In[143]:


investors = unicorn_companies['Select Investors'].str.split(',', expand=True).stack().str.strip().value_counts()
investors.head()


# In[144]:


import matplotlib.pyplot as plt

# Extract the investor frequencies
investors = unicorn_companies['Select Investors'].str.split(',', expand=True).stack().str.strip().value_counts()

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
investors.head().plot(kind='barh', ax=ax)
ax.set_xlabel('Frequency')
ax.set_ylabel('Investor')
ax.set_title('Top 5 Most Frequent Select Investors')
plt.show()


# ##OBSERVATION===The resulting chart shows the top 5 most frequent select investors and their frequency in the unicorn companies.
#  #From this chart, it can be observed that Accel is the most frequent select investor, followed by Tiger Global Management,Andreessen 
# Horowotz, then Sequioa Capital China and Sequioa Capital.
# 

# In[145]:


#ANALYSIS=== What is the the distribution of unicorn companies across different industry categories?


# In[146]:


import matplotlib.pyplot as plt

industry_counts = unicorn_companies['Industry'].value_counts()

fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(industry_counts.values, bins=20)
ax.set_xlabel('Number of Companies')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Unicorn Companies by Industry')
plt.show()


# ###OBSERVATION===The resulting histogram shows that the majority of unicorn companies are in most industries. 
# The distribution appears to be indicating that, there are relatively few unicorn companies in some industries 
# and a larger number in others.
# 
# 

# In[151]:


###Relationship between funding and valuation
plt.scatter(x=df['Funding'].head(20), y=df['Valuation'].head(20))
plt.xlabel('Funding (millions of USD)')
plt.ylabel('Valuation (millions of USD)')
plt.title('Relationship between Funding and Valuation')
plt.show()


# In[152]:


#Which countries have the most unicorns? Are there any cities that appear to be industry hubs?
# Count unicorns by country
# Count unicorns by country
country_counts = df['Country'].value_counts()

# Select countries with at least 25 unicorns
top_countries = country_counts[country_counts >= 25].index

# Filter unicorns by top countries
df_top_countries = df[df['Country'].isin(top_countries)]

# Group unicorns by industry and country
industry_country = df_top_countries.groupby(['Industry', 'Country']).size().reset_index(name='count')

# Create pivot table of unicorn distribution by industry and country
pivot = industry_country.pivot('Industry', 'Country', 'count')

# Create heatmap of unicorn distribution by industry and country
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap='Blues', annot=True, fmt='g', cbar=False, linewidths=1)
plt.xlabel('Country')
plt.ylabel('Industry')
plt.title('Unicorn Distribution by Industry and Country')
plt.show()



# In[121]:


# Count unicorns by city
city_counts = df['City'].value_counts()

# Select cities with at least 10 unicorns
top_cities = city_counts[city_counts >= 10].index

# Filter unicorns by top cities
df_top_cities = df[df['City'].isin(top_cities)]

# Group unicorns by industry and city
industry_city = df_top_cities.groupby(['Industry', 'City']).size().reset_index(name='count')

# Create pivot table of unicorn distribution by industry and city
pivot = industry_city.pivot('Industry', 'City', 'count')

# Create heatmap of unicorn distribution by industry and city
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap='Oranges', annot=True, fmt='g', cbar=False, linewidths=1)
plt.xlabel('City')
plt.ylabel('Industry')
plt.title('Unicorn Distribution by Industry and City')
plt.show()


#  Recommendations for unicorn companies:
# 
# - Invest in startups that have a clear and scalable business model, and a product or service that solves a real problem for their target market.
# - Conduct thorough due diligence on potential investments, including researching the market, competitors, and management team, and analyzing financial statements and projections.
# - Consider investing in startups that have a social or environmental mission, as these companies may have a strong brand and customer loyalty.
# - Build a diverse team with a range of skills and backgrounds to better understand and serve a diverse range of startups.
# - Provide value-add services to portfolio companies, such as mentorship, networking opportunities, and access to resources and expertise.
# - Stay up-to-date with industry trends and changes in regulations that may impact the startup ecosystem.
# - Maintain a long-term perspective and be patient with investments, as startups often take time to reach their full potential.
