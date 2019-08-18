#!/usr/bin/env python
# coding: utf-8

# Hello,
# 
# In this lesson, we try to do etl using Python's Pandas library.
# 
# You will learn about:
# 
#  * What is ETL?
#  * What is PANDAS and Dataframe?
#  * Analyze data
#  * Merge dataframes
#  * Visualize data
#  * Load result into output file
#  
# 
# So, let's get started...

# # <b>WHAT IS ETL?</b>
# 
# ![ETL-3.png](attachment:ETL-3.png)In computing, extract, transform, load (ETL) is the general procedure of copying data from one or more sources into a destination system which represents the data differently from the source(s) or in a different context than the source(s). The ETL process became a popular concept in the 1970s and is often used in data warehousing[1].
# 
# Data extraction involves extracting data from homogeneous or heterogeneous sources; data transformation processes data by data cleansing and transforming them into a proper storage format/structure for the purposes of querying and analysis; finally, data loading describes the insertion of data into the final target database such as an operational data store, a data mart, or a data warehouse.
# 
# A properly designed ETL system extracts data from the source systems, enforces data quality and consistency standards, conforms data so that separate sources can be used together, and finally delivers data in a presentation-ready format so that application developers can build applications and end users can make decisions.
# 
# http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvRXh0cmFjdCxfdHJhbnNmb3JtLF9sb2Fk
# 
# https://www.talend.com/resources/what-is-etl/

# # <b>WHAT IS PANDAS?</b>
# 
# ![Panda-icon.png](attachment:Panda-icon.png)
# 
# Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license. The name is derived from the term "panel data", an econometrics term for data sets that include observations over multiple time periods for the same individuals.
# 
# http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvUGFuZGFzXyhzb2Z0d2FyZSk
# 
# Pandas can handle virtually any data file format, like:
# 
# * Comma-separated values (CSV)
# * XLSX
# * ZIP
# * Plain Text (txt)
# * JSON
# * XML
# * HTML
# * Images
# * Hierarchical Data Format
# * PDF
# * DOCX
# * MP3
# * MP4
# * SQL
# 
# 
# # <b>WHAT IS DATAFRAME?</b>
# 
# 
# Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns. Pandas DataFrame consists of three principal components, the <b>data</b>, <b>rows</b>, and <b>columns</b>.
# 
# 
# ![finallpandas.png](attachment:finallpandas.png)
# 
# 
# 
# 

# # <b>ABOUT DATASET</b>
# 
# I took this dataset from kaggle. Let's look into dataset to understand what kind of datas we have. 
# 
# 
# https://www.kaggle.com/shrutimehta/zomato-restaurants-data
# 
# 
# # from zomato.csv file
# 
# * Restaurant ID: Identification Number
# * Restaurant Name: Name Of the Restaurant
# * Country Code: 216
# * City: City Name of the Restaurant
# * Address
# * Locality:Shot Address Of the Restaurant
# * Locality Verbose: Long Address of the Restaurant
# * Longitude: Longitude
# * Latitude: Latitude
# * Cuisines:Types Of Cuisines Served
# * Average Cost for two: Average Cost if two people visit the Restaurant
# * Currency: Dollars
# * Has Table booking: Can we book tables in Restaurant? Yes/No
# * Has Online delivery: Can we have online delivery ? Yes/No
# * Is delivering now: Is the Restaurant delivering food now? Yes/No
# * Switch to order menu: Switch to order menu ? Yes/ No
# * Price range: Categorized price between 1 -4
# * Aggregate rating: Categorizing ratings between 1-5
# * Rating color: Different colors representing Customer Rating
# * Rating text: Different Rating like Excellent, Very Good ,Good, Avg., Poor, Not Rated
# * Votes: No.Of Votes received by restaurant from customers.
# 
# # from Country-Code.xlsx file
# 
# * Country Code
# * Country: Country Name

# In[1]:


#import libraries

import pandas as pd # data manipulation and analysis library
import matplotlib.pyplot as plt # visualization library


# In[2]:


#read datafiles and create dataframe

zomato_df=pd.read_csv('zomato.csv',encoding = "Latin") #or encoding = "latin"

country_df=pd.read_excel('Country-Code.xlsx')

zomato_df.head()
#zomato_df['Average Cost for two'].head(10)


# In[3]:


#DATA ANALYZING
print('ZOMATO DATA ANALYZE\n\n')
print(zomato_df.head())
print(zomato_df.tail())
print(zomato_df.shape)
# computes various summary statistics, excluding NaN values 
zomato_df.describe() 
# for computing correlations 
zomato_df.corr() 
# computes numerical data ranks 
zomato_df.rank() 


print('COUNTRY DATA ANALYZE\n\n')

print(country_df.head())
print(country_df.tail())
print(country_df.shape)
# computes various summary statistics, excluding NaN values 
country_df.describe() 
# for computing correlations 
country_df.corr() 
# computes numerical data ranks 
country_df.rank() 


# In[4]:


#join,merge, null value handing, remove column, transforming

result = pd.merge(zomato_df,
                 country_df,
                 on='Country Code', 
                 how='left')
result.head()


# In[5]:


#DATA VISUALIZING
# according to country and restaurant note

# the Python Graph Gallery --> https://python-graph-gallery.com/
# shows presence of a lot of outliers/extreme values 

result.boxplot(column='Aggregate rating', by = 'Country',rot=90,fontsize=15) 

plt.show() 


# In[6]:


#Loading to output file - df.to_csv
result.to_csv('targetfile.csv')


# # CONCLUSION
# 
# As you see, we import, analyze, join, visualize and load our datas into target in this notebook.
# 
# We concluded this lesson. Thank you...

# In[7]:


result.head(5)


# In[8]:


result= result[(result['Aggregate rating'].astype(float) > 4.8)  & (result['Votes'].astype(int)> 1000)]

result.head()


# In[9]:


# BONUS :)

# HELLO WORLD FOR MACHINE LEARNING
from sklearn import tree
features=[[140,1],[130,1],[150,0],[170,0]]
labels=[0,0,1,1]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print (clf.predict([[150,0]]))

