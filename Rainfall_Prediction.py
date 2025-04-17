#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the libraries
import numpy as np
import pandas as pd


# In[2]:


# Read the Data
data = pd.read_csv("D:/Python/austin_weather.csv")
data


# In[3]:


# Drop the unnecessary Columns
data = data.drop(["Events", "Date", "SeaLevelPressureLowInches"], axis = 1)


# In[4]:


data = data.replace('T', 0.0)


# In[5]:


data = data.replace('-', 0.0)


# In[6]:


data


# In[7]:


data.to_csv("D:/Python/austin_weather_final.csv")


# In[8]:


# Import the libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[9]:


data = pd.read_csv("D:/Python/austin_weather_final.csv")
data


# In[10]:


X = data.drop(['PrecipitationSumInches'], axis =1)


# In[11]:


Y = data["PrecipitationSumInches"]


# In[12]:


# Reshaping it into second Vector
Y = Y.values.reshape(-1,1)


# In[13]:


Y


# In[14]:


day_index = 798
days = [ i for i in range(Y.size)]


# In[15]:


# Initialise the Linear Regression Classifier
clf = LinearRegression()


# In[16]:


# Train the Classifier
clf.fit(X,Y)


# In[17]:


# Plot a Graph
print("The Percipitation Trend Graph")
plt.scatter(days, Y, color = 'g')
plt.scatter(days[day_index], Y[day_index], color = 'r')
plt.title("Percipitation level")
plt.xlabel("Days")
plt.ylabel("Percipitation in Inches")
plt.show()
x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'] )


# In[18]:


print("Percipitation VS Attribute Trend Graph")

for i in range(x_vis.columns.size):
    plt.subplot(3,2,i+1)
    plt.scatter(days, x_vis[x_vis.columns.values[i][:100]], color = 'g')
    plt.scatter(days[day_index], x_vis[x_vis.columns.values[i]][day_index], color = 'r')
    plt.title(x_vis.columns.values[i])
plt.show()


# In[ ]:




