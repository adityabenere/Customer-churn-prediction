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


# # We can see that the column named total charges is mistaken for the type "object" . It happened because there was missing value at 356 th position. lets just drop the entire row and change the column to numeric so that there is no issue during one hot encoding.

# In[18]:


df['TotalCharges']=pd.to_numeric(df.TotalCharges, errors='coerce')


# In[19]:


df.info()


# # Changing data type of column SeniorCitizen

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


# # Hence our final cleaned data set consists of 5976 rows and 20 columns

# # Viewing a random observation

# In[26]:


df.iloc[346]


# # Getting basic insights from our data through visualization

# In[27]:


int_feat = df.select_dtypes(exclude=['object','category']).columns
fig, ax = plt.subplots(nrows=2, ncols = 2, figsize=(15,8), constrained_layout=True)
ax=ax.flatten()
for c,i in enumerate(int_feat):
    sns.histplot(df[i], ax=ax[c], bins=10)
    ax[c].set_title(i)


# In[28]:


cat_cols = df.select_dtypes(include=['object','category']).columns
fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(20,20), constrained_layout=True)
ax=ax.flatten()
for x,i in enumerate(cat_cols):
    sns.countplot(x=df[i], ax=ax[x])


# # Bivariate Analysis

# In[29]:


fig, ax = plt.subplots(nrows=2, ncols = 2, figsize=(15,8), constrained_layout=True)
ax=ax.flatten()
for c,i in enumerate(int_feat):
    sns.histplot(data=df, x=i, ax=ax[c], bins=10, hue='Churn', kde=True)
    ax[c].set_title(i)


# 1) As the tenure increases the number of customers churning is less
# 
# 2) For monthly charges between 70 to 100 there is a rise in the number of customers churning.

# In[30]:


fig, ax = plt.subplots(nrows=2, ncols = 2, figsize=(15,8), constrained_layout=True)
ax=ax.flatten()
for c,i in enumerate(int_feat):
    sns.boxplot(data=df, x=i, ax=ax[c], y='Churn')
    ax[c].set_title(i)


# In most of the cases there are no outliers present, But in case of Total charges higher than 5500 where customers are leaving the service, there are outliers present, Hence we will replace those values by the value of right Whisker.

# In[31]:


cat_cols = df.select_dtypes(include=['object','category']).columns.to_list()
cat_cols.remove('Churn')
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(15,15), constrained_layout=True)
ax=ax.flatten()
for x,i in enumerate(cat_cols):
    sns.countplot(data=df,x=i, ax=ax[x], hue='Churn')


# In[32]:


sns.countplot(x=df["InternetService"], hue=df['Churn'])


# In[33]:


sns.countplot(hue=df['Churn'], x=df["PaymentMethod"])


# 1) Fiber Optic service have more customers churning.
# 
# 2) Payment method electronic check also have more customers churning.
# 
# Hence special attention should be paid to these customers.

# # All the catogerical variables in our data are nominal hence we move ahead with one hot encoding

# In[34]:


df=pd.get_dummies(df, drop_first=True)


# In[35]:


df.head()


# # Feature Selection

# # 1) Correlation

# In[36]:


df.corr()


# Results: Now the variables (which are least correlated with target variable) i.e whose correlation with the target variable is between -0.1 and 0.1 we will consider those variables to be removed from our data.
# we can see that the variables:
#     1)gender_Male, 2)PhoneService_Yes, 3)MultipleLines_No phone service, 4) MultipleLines_Yes, 5) OnlineBackup_Yes, 6)DeviceProtection_Yes,7) StreamingTV_Yes, 8)StreamingMovies_Yes, 9)PaymentMethod_Mailed check 
#     have a very less correlation with our target variable.

# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


corrmat= df.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # 2) Multicollinearity

# (Multicollinearity)If 2 independent variables are highly correlated with eachother, it means they produce the same effect, one of them can be dropped from our data.

# In[39]:


threshold=0.85


# In[40]:


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

# In[41]:


correlation(df.iloc[:,:-1],threshold)


# Result: The following valiables can be dropped from our dataset. {'DeviceProtection_No internet service',
#  'InternetService_No',
#  'OnlineBackup_No internet service',
#  'OnlineSecurity_No internet service',
#  'PhoneService_Yes',
#  'StreamingTV_No internet service',
#  'TechSupport_No internet service'}

# # 3) Variance infliation factor (variables with vif>5 can be dropped)

# In[42]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[43]:


df.head()


# In[44]:


df.shape


# In[45]:


x=df.iloc[:,:30]
y=df.iloc[:,30]


# In[46]:


y.head()


# In[47]:


x.head()


# In[48]:


vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
  
print(vif_data)


# In[49]:


# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(x):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

    return(vif)


# In[50]:


calc_vif(x)


# # 4) Feature importance using extra trees regressor

# In[51]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)


# In[52]:


feat_imp =pd.Series(model.feature_importances_, index=x.columns)
feat_imp.nlargest(30).plot(kind='barh')
plt.rcParams['figure.figsize']=(10,7)
plt.show()


# # 5) Using chi-squared k-best method

# Those features with a less score can be dropped

# In[53]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[54]:


### Apply SelectKBest Algorithm
ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(x,y)


# In[55]:


dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(x.columns)


# In[56]:


features_rank=pd.concat([dfcolumns,dfscores],axis=1)


# In[57]:


features_rank.columns=['Features','Score']
features_rank
features_rank.nlargest(30,'Score')


# # Overall conclusion from feature selection.

# For all the feature selection techniques applied above, there are few features that regularly showed up to be removed from our data hence we remove the following 10 less useful variables: PhoneService_Yes, MultipleLines_No phone service, StreamingMovies_Yes, DeviceProtection_No internet service, OnlineBackup_No internet service, TechSupport_No internet service, OnlineSecurity_No internet service, StreamingTV_No internet service, InternetService_No, StreamingMovies_No internet service.
# 

# In[58]:


df.drop(['PhoneService_Yes', 'MultipleLines_No phone service', 'StreamingMovies_Yes', 'DeviceProtection_No internet service', 'OnlineBackup_No internet service', 'TechSupport_No internet service', 'OnlineSecurity_No internet service', 'StreamingTV_No internet service', 'InternetService_No', 'StreamingMovies_No internet service'], axis=1, inplace=True)


# In[59]:


df.head()


# # Model creation

# In[60]:


x1=df.iloc[:,:20]
y1=df.iloc[:,20]


# In[61]:


y1.head()


# In[62]:


x1.head()


# # Normalizing 'tenure','MonthlyCharges','TotalCharges' using min max scalar

# In[63]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1[cols_to_scale] = scaler.fit_transform(x1[cols_to_scale])


# In[64]:


y1.value_counts()


# In[65]:


x1.head()


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2)


# # 1) Logistic regression

# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x1_train,y1_train)
LR


# In[69]:


y1hat = LR.predict(x1_test)
y1hat


# In[70]:


y1hat_prob = LR.predict_proba(x1_test)
y1hat_prob


# In[71]:


y1hat_prob[:,1]


# In[72]:


confusion_matrix(y1_test,y1hat)


# In[73]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y1_test,y1hat))
pd.crosstab(y1_test,y1hat)


# In[74]:


from sklearn.metrics import classification_report


# In[75]:


print(classification_report(y1_test, y1hat))


# # Since our target is imbalanced i.e the number of customers who are churned is pretty less than the customers not churned. We will use Precision recall curve.
# *Our goal is to choose a threshold where False Positives are reduced.

# In[76]:


from sklearn.metrics import precision_recall_curve


# In[77]:


precision, recall, threshold = precision_recall_curve(y1_test,y1hat_prob[:,1])


# In[78]:


precision


# In[79]:


recall


# In[80]:


threshold


# In[81]:


import plotly.offline as py
import plotly.express as px


# In[82]:


fig = px.line(x=recall[:-1], y = precision[:-1], hover_name=threshold)
fig.update_xaxes(title="Recall")
fig.update_yaxes(title="Precision")
fig


# In[83]:


from sklearn.metrics import auc
auc_1= auc(recall, precision)


# In[84]:


auc_1


# # Random Forest 

# In[85]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x1_train, y1_train)


# In[86]:


y_pred = rf.predict(x1_test)


# In[87]:


print(classification_report(y1_test, y_pred))


# In[88]:


pd.crosstab(y1_test,y_pred)


# In[89]:


print(accuracy_score(y1_test,y_pred))


# # SMOTE (Synthetic Minority Over-sampling Technique)

# In[90]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')


# In[91]:


x1_train_smote, y1_train_smote = smote.fit_resample(x1_train,y1_train)


# In[92]:


from collections import Counter
print("Before SMOTE :" , Counter(y1_train))
print("After SMOTE :" , Counter(y1_train_smote))


# # Logistic Regression (Smote)

# In[93]:


LR = LogisticRegression(C=0.01, solver='liblinear').fit(x1_train_smote, y1_train_smote)
LR


# In[94]:


y1hat = LR.predict(x1_test)
y1hat


# In[95]:


y1hat_prob = LR.predict_proba(x1_test)
y1hat_prob


# In[96]:


from sklearn.metrics import classification_report


# In[97]:


print(classification_report(y1_test, y1hat))


# In[98]:


pd.crosstab(y1_test,y1hat)


# In[99]:


print(accuracy_score(y1_test,y1hat))


# # Random Forest (Smote)

# In[100]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x1_train_smote, y1_train_smote)


# In[101]:


y2pred = rf.predict(x1_test)


# In[102]:


print(classification_report(y1_test, y2pred))


# In[103]:


pd.crosstab(y1_test,y2pred)


# In[104]:


print(accuracy_score(y1_test,y2pred))


# CONCLUSION : 1)lOGISTIC REGRESSION(SMOTE) IS REDDUCING THE FALSE POSITIVES TO 60 BUT THE TOTAL MISCLASSIFICATION DONE BY THE MODEL IS 301.
#              2)ON THE OTHER HAND RANDOM FOREST (SMOTE) IS REDUCING THE FALSE POSITIVES TO 97 AND THE MISCLASSIFICATION DONE BY THE MODEL IS 270

# In[ ]:




