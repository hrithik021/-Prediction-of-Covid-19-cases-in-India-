#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt


# In[2]:


data=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/10-19-2020.csv")
data.tail(10)


# In[3]:


#data=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/10-19-2020.csv")
#data.tail(10)
covid=pd.read_csv("https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv")
#updated_cases=pd.read_csv("https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv")
#updated_cases.head(10)
covid.tail()


# In[4]:


print("size/shape of the dataset",covid.shape)
print("checking for null values",covid.isnull().sum())
print("data type",covid.dtypes )


# <h1> hide index here<h1>
# #country_cases.style.hide_index()
# #country_cases.head() 
# #print(country_cases.to_string(index=False))

# In[5]:


covid["Date"]= pd.to_datetime(covid["Date"],dayfirst=True)
covid["Date"]


# In[6]:


#grouping diff types of cases as per date
datewise=covid.groupby(["Date"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})


# In[7]:


print("Basic information at this time",covid["Date"].iloc[-1])
print("total number of confirmed cases around the world",datewise["Confirmed"].iloc[-1])
print("total number of Recovered cases around the world",datewise["Recovered"].iloc[-1])
print("total number of Deaths cases around the world",datewise["Deaths"].iloc[-1])
print("total number of active cases",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))
print("total number of closed cases",(datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1]))
#print(covid["Date"].iloc[-1])


# In[8]:


plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date,y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])
plt.title("Distributions plot for Active Cases by TANISH SAINI")
plt.xticks(rotation=90)


# In[9]:


plt.figure(figsize=(15,5))
sns.barplot(x=datewise.index.date,y=datewise["Recovered"]+datewise["Deaths"])
plt.title("Distributions plot for closed  Cases by TANISH SAINI")
plt.xticks(rotation=90)


# In[10]:


datewise["weekofyear"]=datewise.index.weekofyear
week_num=[]
weekwise_confirmed=[]
weekwise_recovered=[]
weekwise_deaths=[]
w = 1
for i in list(datewise["weekofyear"].unique()):
    weekwise_confirmed.append(datewise[datewise["weekofyear"]==i]["Confirmed"].iloc[-1])
    weekwise_recovered.append(datewise[datewise["weekofyear"]==i]["Recovered"].iloc[-1])
    weekwise_deaths.append(datewise[datewise["weekofyear"]==i]["Deaths"].iloc[-1])
    week_num.append(w)
    w=w+1
plt.figure(figsize=(8,5))
plt.plot(week_num,weekwise_confirmed,linewidth=3)
plt.plot(week_num,weekwise_recovered,linewidth=3)
plt.plot(week_num,weekwise_deaths,linewidth=3)
plt.xlabel("weekNumber")
plt.ylabel("Number of cases")
plt.title("weekly progress of different types of cases by TANISH SAINI ")


# In[11]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(22,6))
sns.barplot(x=week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)
#sns.barplot(x=week_num,y=pd.series(weekwise_recovered).diff().fillna(0),ax=ax2)
sns.barplot(x=week_num,y=pd.Series(weekwise_deaths).diff().fillna(0),ax=ax2)
ax1.set_xlabel("week_Number")
ax2.set_xlabel("week_Number")
ax1.set_ylabel("Number of confimed cases")
ax2.set_ylabel("Number of deaths cases")
ax1.set_title("weekly increases in number of confirmed cases")
ax2.set_title("weekly increases in number of deaths cases")
plt.show()


# In[12]:


print("Average increases in number of confirmed cases everyday:",np.round(datewise["Confirmed"].diff().fillna(0).mean()))
print("Average increases in number of recovered cases everyday:",np.round(datewise["Recovered"].diff().fillna(0).mean()))
print("Average increases in number of deaths cases everyday:",np.round(datewise["Deaths"].diff().fillna(0).mean()))


plt.figure(figsize=(15,6))
plt.plot(datewise["Confirmed"].diff().fillna(0),label="daily increase in confirmed cases",linewidth=3)
plt.plot(datewise["Recovered"].diff().fillna(0),label="daily increase in recovered cases",linewidth=3)
plt.plot(datewise["Deaths"].diff().fillna(0),label="daily increase in deaths cases",linewidth=3)
plt.xlabel("Timestamp")
plt.ylabel("daily cases")
plt.title("daily cases")
plt.legend()
plt.xticks(rotation=90)
plt.show


# In[13]:


#country wise analysis
#calculating country wise Mortality rate
countrywise=covid[covid["Date"]==covid["Date"].max()].groupby(["Country"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"}).sort_values(["Confirmed"],ascending=False)
countrywise["Mortality"]=(countrywise["Deaths"]/countrywise["Recovered"])*100
countrywise["Recovered"]=(countrywise["Recovered"]/countrywise["Confirmed"])*100


# In[14]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(25,10))
top_15confirmed=countrywise.sort_values(["Confirmed"],ascending=False).head(15)
top_15deaths=countrywise.sort_values(["Deaths"],ascending=False).head(15)
sns.barplot(x=top_15confirmed["Confirmed"],y=top_15confirmed.index,ax=ax1)
ax1.set_title("top 15 countries as per number of confirmed cases")
sns.barplot(x=top_15deaths["Deaths"],y=top_15deaths.index,ax=ax2)
ax2.set_title("top 15 coutries as per number of deaths cases")


# In[15]:


#data analysis for india
india_data=covid[covid["Country"]=="India"]
datewise_india = india_data.groupby(["Date"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
print(datewise_india.iloc[-1])
print("total number of Active acses",datewise_india["Confirmed"].iloc[-1]-datewise_india["Recovered"].iloc[-1]-datewise_india["Deaths"].iloc[-1])
print("total number of closed cases",datewise_india["Recovered"].iloc[-1]+datewise_india["Deaths"].iloc[-1])                                                   


# <h1> weekly report of india <h1>

# In[16]:


datewise_india["weekofyear"]=datewise_india.index.weekofyear
india_week_num=[]
india_weekwise_confirmed=[]
india_weekwise_recovered=[]
india_weekwise_deaths=[]
w = 1
for i in list(datewise["weekofyear"].unique()):
    india_weekwise_confirmed.append(datewise_india[datewise_india["weekofyear"]==i]["Confirmed"].iloc[-1])
    india_weekwise_recovered.append(datewise_india[datewise_india["weekofyear"]==i]["Recovered"].iloc[-1])
    india_weekwise_deaths.append(datewise_india[datewise_india["weekofyear"]==i]["Deaths"].iloc[-1])
    india_week_num.append(w)
    w=w+1
plt.figure(figsize=(8,5))
plt.plot(india_week_num,india_weekwise_confirmed,linewidth=3)
plt.plot(india_week_num,india_weekwise_recovered,linewidth=3)
plt.plot(india_week_num,india_weekwise_deaths,linewidth=3)
plt.xlabel("weekNumber")
plt.ylabel("Number of cases")
plt.title("weekly progress of different types of cases by TANISH SAINI ")


# <h1>US Analysis with weekly report <h1>

# In[17]:


#data analysis for US
US_data=covid[covid["Country"]=="US"]
datewise_US = US_data.groupby(["Date"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
print(datewise_US.iloc[-1])
print("total number of Active acses",datewise_US["Confirmed"].iloc[-1]-datewise_US["Recovered"].iloc[-1]-datewise_US["Deaths"].iloc[-1])
print("total number of closed cases",datewise_US["Recovered"].iloc[-1]+datewise_US["Deaths"].iloc[-1])


# In[18]:


datewise_US["weekofyear"]=datewise_US.index.weekofyear
US_week_num=[]
US_weekwise_confirmed=[]
US_weekwise_recovered=[]
US_weekwise_deaths=[]
w = 1
for i in list(datewise["weekofyear"].unique()):
    US_weekwise_confirmed.append(datewise_US[datewise_US["weekofyear"]==i]["Confirmed"].iloc[-1])
    US_weekwise_recovered.append(datewise_US[datewise_US["weekofyear"]==i]["Recovered"].iloc[-1])
    US_weekwise_deaths.append(datewise_US[datewise_US["weekofyear"]==i]["Deaths"].iloc[-1])
    US_week_num.append(w)
    w=w+1
plt.figure(figsize=(8,5))
plt.plot(US_week_num,US_weekwise_confirmed,linewidth=3)
plt.plot(US_week_num,US_weekwise_recovered,linewidth=3)
plt.plot(US_week_num,US_weekwise_deaths,linewidth=3)
plt.xlabel("weekNumber")
plt.ylabel("Number of cases")
plt.title("weekly progress of different types of cases by TANISH SAINI ")


# In[19]:


max_ind =datewise_india["Confirmed"].max()
china_data=covid[covid["Country"]=="China"]
Italy_data=covid[covid["Country"]=="Italy"]
US_data=covid[covid["Country"]=="US"]
spain_data=covid[covid["Country"]=="Spain"]
datewise_china=china_data.groupby(["Date"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
datewise_Italy=Italy_data.groupby(["Date"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
datewise_US=US_data.groupby(["Date"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
datewise_spain=spain_data.groupby(["Date"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
print("it took ",datewise_india[datewise_india["Confirmed"]>0].shape[0],"days in india to reach",max_ind,"Confirmed cases")
#print("it took ",datewise_china[(datewise_china["Confirmed"]>0)&(datewise_china["Confirmed"]<=max_ind)].shape[0],"days in china to reach Confirmed cases")
#print("it took ",datewise_Italy[(datewise_Italy["Confirmed"]>0)&(datewise_Italy["Confirmed"]<=max_ind)].shape[0],"days in italy to reachConfirmed cases")
print("it took ",datewise_US[(datewise_US["Confirmed"]>0)&(datewise_US["Confirmed"]<=max_ind)].shape[0],"days in US to reachConfirmed cases")
#print("it took ",datewise_spain[(datewise_spain["Confirmed"]>0)&(datewise_spain["Confirmed"]<=max_ind)].shape[0],"days in spain to reach Confirmed cases")


# <h1> model implementation for prediction <h1>

# In[20]:


datewise["Days Since"]=datewise.index-datewise.index[0]
datewise["Days Since"] = datewise["Days Since"].dt.days
train_ml = datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml = datewise.iloc[:int(datewise.shape[0]*0.95):]
model_scores=[]


# <h1> linear regression <h1>

# In[21]:


lin_reg = LinearRegression(normalize=True)
#svm = SVR(C=1,degree=5,kernel='poly',epsilon=0.001)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
#svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))


# In[23]:


prediction_valid_lin_reg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))
#prediction_valid_svm=svm.predict(np.array(valid_ml["Days since"]).reshape(-1,1))


# In[47]:


new_date =[]
new_prediction_lr=[]
#new_prediction_svm=[]
for i in range(1,20):
    new_date.append(datewise.index[-1]+timedelta(days=i))
    new_prediction_lr.append(lin_reg.predict(np.array(datewise["Days Since"].max()+i).reshape(-1,1))[0][0])
    #new_prediction_svm.append(svm.predict(np.array(datewise["Days since"].max()+i).reshape(-1,1))[0])
pd.set_option("display.float_format",lambda x: '%.f'  % x)
model_predictions= pd.DataFrame(zip(new_date,new_prediction_lr),columns=["Dates","LR"])
model_predictions.head(20)                               


# In[54]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.90)]
valid=datewise.iloc[int(datewise.shape[0]*0.90):]


# In[55]:


holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=1.1,smoothing_slope=0.2)
y_pred=valid.copy()
y_pred["Holt"]=holt.forecast(len(valid))


# In[57]:


holt_new_date=[]
holt_new_prediction=[]
for i in range(1,20):
    holt_new_date.append(datewise.index[-1]+timedelta(days=1))
    holt_new_prediction.append(holt.forecast((len(valid)+1))[-1])
model_predictions["holts linear model prediction"]=holt_new_prediction
model_predictions.head(15)


# In[ ]:





# In[ ]:




