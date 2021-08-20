#!/usr/bin/env python
# coding: utf-8

# # IMPORTING NECESSARY LIBRARIES

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# # IMPORTING DATA

# In[5]:


df=pd.read_csv("C:/Users/Acer/Downloads/telecom_users.csv")


# In[6]:


df.head()


# In[7]:



#customerID -customer id
#gender - client gender (male / female)
#SeniorCitizen - is the client retired (1, 0)
#Partner - is the client married (Yes, No)
#tenure - how many months a person has been a client of the company
#PhoneService - is the telephone service connected (Yes, No)
#MultipleLines - are multiple phone lines connected (Yes, No, No phone service)
#InternetService - client's Internet service provider (DSL, Fiber optic, No)
#OnlineSecurity - is the online security service connected (Yes, No, No internet service)
#OnlineBackup - is the online backup service activated (Yes, No, No internet service)
#DeviceProtection - does the client have equipment insurance (Yes, No, No internet service)
#TechSupport - is the technical support service connected (Yes, No, No internet service)
#treamingTV - is the streaming TV service connected (Yes, No, No internet service)
#StreamingMovies - is the streaming cinema service activated (Yes, No, No internet service)
#Contract - type of customer contract (Month-to-month, One year, Two year)
#PaperlessBilling - whether the client uses paperless billing (Yes, No)
#PaymentMethod - payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
#MonthlyCharges - current monthly payment
#TotalCharges - the total amount that the client paid for the services for the entire time
#Churn - whether there was a churn (Yes or No)


# In[8]:


df.shape


# # TO CHECK THE DATA TYPE OF EACH COLUMN

# In[9]:


df.dtypes


# Here we can see that the column "TotalCharges" is detected as object but it's a float, hence later we need to identify and resolve this issue. 
# also the column "SeniorCitizen" is an integer but we need it as a catogery i.e yes/no type.

# # CHECKING FOR UNIQUE CATOGERIES IN EACH COLUMN

# In[10]:


print(df['gender'].unique())


# In[11]:


print(df['Partner'].unique())
print(df['Dependents'].unique())
print(df['PhoneService'].unique())
print(df['MultipleLines'].unique())
print(df['InternetService'].unique())
print(df['OnlineSecurity'].unique())
print(df['OnlineBackup'].unique())
print(df['DeviceProtection'].unique())
print(df['TechSupport'].unique())
print(df['StreamingTV'].unique())
print(df['StreamingMovies'].unique())
print(df['Contract'].unique())
print(df['PaperlessBilling'].unique())
print(df['PaymentMethod'].unique())
print(df['TotalCharges'].unique())
print(df['Churn'].unique())


# # IDENTIFYING COLUMNS WITH MISSING VALUES 

# In[12]:


df.isnull().sum()


# In[13]:


df.isnull().count()


# # DROPPING UNNECESSARY COLUMNS

# In[14]:


df.drop('customerID', axis=1, inplace=True)


# In[15]:


df.drop('Unnamed: 0', axis=1, inplace=True)


# In[16]:


df.head()


# In[17]:


df['TotalCharges'].dtypes


# # we can see that the column named total charges is mistaken for the type "object" . It happened because there was missing value at 356 th position. lets just drop the entire row and change the column to numeric so that there is no issue during one hot encoding.

# In[18]:


df['TotalCharges']=pd.to_numeric(df.TotalCharges, errors='coerce')


# In[19]:


df.info()


# # changing data type of column SeniorCitizen

# In[20]:


df["SeniorCitizen"].replace([0,1],["No","Yes"], inplace=True)
df["SeniorCitizen"].unique()


# In[21]:


df.describe()


# # Dropping the rows with missing values

# In[22]:


df.dropna(axis=0,how='any', inplace= True)


# In[23]:


df.head()


# In[24]:


df.shape


# In[25]:


df.describe()


# Hence our final cleaned data set consists of 5976 rows and 20 columns

# # Viewing a random observation

# In[26]:


df.iloc[346]


# # getting basic insights from our data through visualization

# In[27]:


sns.catplot(x="gender", kind="count", data=df)


# In[28]:


sns.catplot(x="Partner", kind="count", data=df)
sns.catplot(x="Dependents", kind="count", data=df)
sns.catplot(x="PhoneService", kind="count", data=df)
sns.catplot(x="MultipleLines", kind="count", data=df)
sns.catplot(x="InternetService", kind="count", data=df)
sns.catplot(x="OnlineSecurity", kind="count", data=df)
sns.catplot(x="OnlineBackup", kind="count", data=df)
sns.catplot(x="DeviceProtection", kind="count", data=df)
sns.catplot(x="TechSupport", kind="count", data=df)
sns.catplot(x="StreamingTV", kind="count", data=df)
sns.catplot(x="StreamingMovies", kind="count", data=df)
sns.catplot(x="Contract", kind="count", data=df)
sns.catplot(x="PaperlessBilling", kind="count", data=df)
sns.catplot(y="PaymentMethod", kind="count", data=df)
sns.catplot(x="Churn", kind="count", data=df)


# In[29]:


sns.catplot(y="PaymentMethod", kind="count", data=df)


# # All the catogerical variables in our data are nominal hence we move ahead with one hot encoding

# In[30]:


df=pd.get_dummies(df, drop_first=True)


# In[31]:


df.head()


# In[ ]:


run last


# In[80]:


from pandas_profiling import ProfileReport 
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True) 
 
profile.to_widgets() 
 
import sweetviz as sv 
my_report = sv.analyze(df,target_feat= 'Churn_Yes',pairwise_analysis= 'auto') 
my_report.show_html("df_report.html")


# # Feature Selection

# In[32]:


df.corr()


# now the variables (which are least correlated with target variable) i.e whose correlation with the target variable is between -0.1 and 0.1 we will consider those variables to be removed from our data.
# we can see that the variables:
#     1)gender_Male, 2)PhoneService_Yes, 3)MultipleLines_No phone service, 4) MultipleLines_Yes, 5) OnlineBackup_Yes, 6)DeviceProtection_Yes,7) StreamingTV_Yes, 8)StreamingMovies_Yes, 9)PaymentMethod_Mailed check 
#     have a very less correlation with our target variable.

# In[33]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


corrmat= df.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# (multicollinearity)finding those independent variables which are highly correlated with eachother, those with higher correlation will be removed from our data

# In[35]:


threshold=0.85


# In[36]:


# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[j]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# i.e if the correlation between two independent variables is > 0.85 we consider that they contribute the same role hence we can drop one of them

# In[37]:


correlation(df.iloc[:,:-1],threshold)


# # Using Multicollinearity to remove the independent variable which are highly correlated with each other, hence we use variance infliation factor (variables with vif>5 will be remove)

# In[38]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[39]:


df.head()


# In[40]:


df.shape


# In[41]:


x=df.iloc[:,:30]
y=df.iloc[:,30]


# In[42]:


y.head()


# In[43]:


x.head()


# In[44]:


vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
  
print(vif_data)


# In[45]:


# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(x):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

    return(vif)


# In[46]:


calc_vif(x)


# # feature importance using extra trees regressor

# In[47]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)


# In[48]:


feat_imp =pd.Series(model.feature_importances_, index=x.columns)
feat_imp.nlargest(30).plot(kind='barh')
plt.rcParams['figure.figsize']=(10,7)
plt.show()


# # Using chi-squared k-best method

# In[49]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[50]:


### Apply SelectKBest Algorithm
ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(x,y)


# In[51]:


dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(x.columns)


# In[52]:


features_rank=pd.concat([dfcolumns,dfscores],axis=1)


# In[53]:


features_rank.columns=['Features','Score']
features_rank
features_rank.nlargest(30,'Score')


# For all the feature selection techniques applied above, there are few features that regularly showed up to be removed from our data hence we remove the following 10 less useful variables: PhoneService_Yes, MultipleLines_No phone service, StreamingMovies_Yes, DeviceProtection_No internet service, OnlineBackup_No internet service, TechSupport_No internet service, OnlineSecurity_No internet service, StreamingTV_No internet service, InternetService_No, StreamingMovies_No internet service.
# 

# In[54]:


df.drop(['PhoneService_Yes', 'MultipleLines_No phone service', 'StreamingMovies_Yes', 'DeviceProtection_No internet service', 'OnlineBackup_No internet service', 'TechSupport_No internet service', 'OnlineSecurity_No internet service', 'StreamingTV_No internet service', 'InternetService_No', 'StreamingMovies_No internet service'], axis=1, inplace=True)


# In[55]:


df.head()


# # Model creation

# In[56]:


x1=df.iloc[:,:20]
y1=df.iloc[:,20]


# In[57]:


y1.head()


# In[58]:


x1.head()


# # Normalizing 'tenure','MonthlyCharges','TotalCharges' using min max scalar

# In[59]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1[cols_to_scale] = scaler.fit_transform(x1[cols_to_scale])


# In[60]:


y1.value_counts()


# In[61]:


x1.head()


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2)


# In[64]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x1_train,y1_train)
LR


# In[65]:


y1hat = LR.predict(x1_test)
y1hat


# In[66]:


y1hat_prob = LR.predict_proba(x1_test)
y1hat_prob


# In[67]:


confusion_matrix(y1_test,y1hat)


# In[68]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y1_test,y1hat))
pd.crosstab(y1_test,y1hat)


# Our dataset is quiet imbalenced hence we will use SMOTE to balance our dataset

# # SMOTE 

# In[69]:


pip install --upgrade scikit-learn


# In[70]:


pip install imblearn


# In[71]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')


# In[72]:


x1_train_smote, y1_train_smote = smote.fit_resample(x1_train,y1_train)


# In[73]:


from collections import Counter
print("Before SMOTE :" , Counter(y1_train))
print("After SMOTE :" , Counter(y1_train_smote))


# In[74]:


LR = LogisticRegression(C=0.01, solver='liblinear').fit(x1_train_smote, y1_train_smote)
LR


# In[75]:


y1hat = LR.predict(x1_test)
y1hat


# In[76]:


y1hat_prob = LR.predict_proba(x1_test)
y1hat_prob


# In[77]:


print(accuracy_score(y1_test,y1hat))
pd.crosstab(y1_test,y1hat)


# # Using KNN

# In[78]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x1_train_smote, y1_train_smote)
y_predict = model.predict(x1_test)


# In[79]:


print(accuracy_score(y1_test,y_predict))
pd.crosstab(y1_test,y_predict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




