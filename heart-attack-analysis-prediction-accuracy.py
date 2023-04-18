#!/usr/bin/env python
# coding: utf-8

# # Heart Attack Analysis & Prediction Dataset

# Age : Age of the patient
# 
# Sex : Sex of the patient
# 
# exang: exercise induced angina (1 = yes; 0 = no)
# 
# ca: number of major vessels (0-3)
# 
# cp : Chest Pain type chest pain type
# Value 1: typical angina
# Value 2: atypical angina
# Value 3: non-anginal pain
# Value 4: asymptomatic
# 
# trtbps : resting blood pressure (in mm Hg)
# 
# chol : cholestoral in mg/dl fetched via BMI sensor
# 
# fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 
# rest_ecg : resting electrocardiographic results
# Value 0: normal
# Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# 
# thalach : maximum heart rate achieved
# 
# target : 0= less chance of heart attack 1= more chance of heart attack

# # Importing Libraries

# In[47]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np


# In[48]:


df = pd.read_csv('heart.csv')


# In[49]:


df.head()


# In[50]:


df.describe()


# In[51]:


df.info()
print('Number of rows are',df.shape[0], 'and number of columns are ',df.shape[1])


# # Checking For Null Values 

# In[52]:


df.isnull().sum()


# No missing data found

# # Check for duplicates

# In[53]:


df[df.duplicated()]


# In[54]:


# Remove duplicate rows from the df
df.drop_duplicates(keep='first',inplace=True)


# In[55]:


# verify if the duplicate row is removed
df[df.duplicated()]


# In[56]:


df = df.reset_index()


# In[57]:


df = df.drop("index", axis=1)


# In[58]:


df.head()


# In[59]:


#check for new shape after the removal of duplicate row
print('Number of rows are',df.shape[0], 'and number of columns are ',df.shape[1])


# In[60]:


df.describe()


# # Data Visualization

# # Age

# In[61]:


age_prob = sns.kdeplot(data = df, hue = 'output', x = 'age')


# I Find it strange, the fact that you're unlikely to get a heart attack if you survived your mid fifties. Though the analysis so, but I am not convinved yet.

# In[62]:


plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu');


# # Sex

# In[63]:


pd.crosstab(df.sex,df.output).plot(kind="bar", stacked=True, figsize=(5,5), color=['teal','cyan'])
plt.title('Gender comparison')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()


# Males tend to have Heart Attack more than females

# # cp

# In[64]:


pie = df.groupby('cp')["output"].count().plot(kind="pie",autopct='%1.1f%%',figsize=(5,5),title="Chest Pain type chest pain type")


# In[65]:


g = sns.FacetGrid(df, col = 'output', row = 'sex', height=3) 
g.map(sns.histplot, "cp", binwidth=0.3 )   


# The probability of men getting chest pain type 1 is 4 times higher than women getting it, No high correlation between chest pain type 1 and actually getting a heart attack

# # trtbps

# In[66]:


g = sns.FacetGrid(df, col = 'output', height=5) 
g.map(sns.histplot, "trtbps", binwidth = 2 ) 


# For blood pressure: overall, it's a weak coorelation, but I managed to get the following information from the data.
# 
# At its normal state: there is no coorelation between getting a heart attack. 
# Above 120: (Which is the bloob pressure normal state) it's more likely to get a heart attack.
# Above 160: there is no certanity a person would get a heart attack. 

# # chol

# In[67]:


chol_prob = sns.kdeplot(data = df, hue = 'output', x = 'chol')


# People with a higher amount of cholestrol in their blood are more likely to get a heart attack.

# # fbs

# In[68]:


g = sns.FacetGrid(df, col = 'output', height=3) 
g.map(sns.histplot, "fbs", binwidth=0.1 )  


# There is a significant indication that higher blood sugar indicates a heart attack #Weak Coorelation.

# # restecg

# In[69]:


g = sns.FacetGrid(df, col = 'output', height=4) 
g.map(sns.histplot, "restecg", binwidth = 0.1 ) 


# We spoke earlier about three results you can get from the "restecg" column, idicating that the second type shows a problem in the heart. Well, the data also says so, as there is a higher chance of getting a heart attack with type 2(represnted by 1 in the dataset).

# # thalachh

# In[70]:


g = sns.FacetGrid(df, col = 'output', height=5) 
g.map(sns.histplot, "thalachh", binwidth = 2 ) 


# There is a strong coorelation between achieving a heart rate that is higher than 140 and getting a heart attack.

# # exng

# In[71]:


g = sns.FacetGrid(df, col = 'output', height=3) 
g.map(sns.histplot, "exng", binwidth=0.1 )   


# People who survived a previous stroke before has a higher chance of 50% to get a heart attack.

# # thall

# In[72]:


g = sns.FacetGrid(df, col = 'output') 
g.map(sns.histplot, "thall", binwidth=0.3 )   


# for the "thall", type 2 indicates a higher probability of getting a heart attack.

# In[73]:


df


# In[74]:


df.dtypes


# In[75]:


df[["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","slp","caa","thall","output"]]=df[["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","slp","caa","thall","output"]].astype(object)


# In[76]:


df.describe(include=object)


# In[77]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df.select_dtypes(include=numerics)
catdf=df.drop(newdf.columns,axis=1)
catdf.drop(["output"],axis=1,inplace=True)
print("continous attributes")
print(newdf.columns)
print("categorical attributes")
print(catdf.columns)


# # Feature selection

# In[78]:


import matplotlib.pyplot as plt
import seaborn as sns
for i in catdf.columns:
    plt.figure(i)
    sns.barplot(df["output"],df[i])


# # Anova Test

# In[79]:


from scipy import stats
import scipy.stats
def anova_test(cat,nem):
    grouped_test2=df[[cat, nem]].groupby([cat])
    lst=[]
    for i in df[cat].unique():
        lst.append(list(grouped_test2.get_group(i)[nem]))
    f_val, p_val = stats.f_oneway(*lst) 
    print("f_statistics\n",f_val,p_val)
    dfn1=len(df[cat].unique())-1
    dfd1=len(df[cat].index)-len(df[cat].unique())
    print("degree of freedom\n",dfn1,dfd1)
    f_crit=scipy.stats.f.ppf(q=1-.05, dfn=dfn1, dfd=dfd1)
    print("f_critical\n",f_crit)
    if f_val>f_crit and p_val<0.05:
        print(cat,"attribute accepted")
        print("_______________________________")
        return True
    else:
        print(cat,"attribute rejected")
        print("_______________________________")
        return False
    


# In[80]:


lst_imp=[]
for i in catdf.columns:
    if(anova_test("output",i)):
        lst_imp.append(i)
        


# In[81]:


lst_imp


# In[82]:


for i in catdf.columns:
    print("cross tab for",i)
    ct=pd.crosstab(index=catdf[i],columns=df["output"],normalize='index',dropna=True)
    print(ct)
    print("___________")


# # Chi Test

# In[83]:


lst_cat_imp=[]
for i in catdf.columns:   
    chi_data=pd.crosstab(index=df[i],columns=df['output'])
    chi_lst=[]
    for ind,row in chi_data.iterrows():
        chi_lst.append(row.values)
    (chi2,p,dof,_)=stats.chi2_contingency(chi_lst)
    import scipy
    crit=scipy.stats.chi2.ppf(1-0.05, dof)
    if p<0.05:
        print(i,"accepted")
        lst_cat_imp.append(i)
        lst_imp.append(i)
    else:
        pass
print(lst_imp)


# In[84]:


df[["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","slp","caa","thall","output"]]=df[["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","slp","caa","thall","output"]].astype(float)


# In[85]:


df.corr()["output"]


# df.corr()["output"] calculates the Pearson correlation coefficient between the "output" numerical attribute and all other numerical attributes in the DataFrame df  The Pearson correlation coefficient measures the strength and direction of the linear relationship between two variables, with values ranging from -1 to 1. A value of 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation.

# In[86]:


lst_imp=['age','sex','cp','trtbps','restecg','thalachh','exng','slp','caa','thall']


# # Data Splitting

# In[87]:


x=df[lst_imp]
y=df["output"]


# In[88]:


from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

std  = MinMaxScaler()
X = pd.DataFrame(std.fit_transform(x) , columns=x.columns)
X


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 40)


# # Feature Scalling

# In[90]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Modeling

# In[91]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')
import xgboost
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
get_ipython().system('pip3 install catboost')
from catboost import CatBoostClassifier
get_ipython().system('pip install lightgbm')
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

classifiers = [[CatBoostClassifier(verbose=0),'CatBoost Classifier'],[XGBClassifier(),'XGB Classifier'], [RandomForestClassifier(),'Random Forest'], 
    [KNeighborsClassifier(), 'K-Nearest Neighbours'], [SGDClassifier(),'SGD Classifier'], [SVC(),'SVC'],[LGBMClassifier(),'LGBM Classifier'],
              [GaussianNB(),'GaussianNB'],[DecisionTreeClassifier(),'Decision Tree Classifier'],[LogisticRegression(),'Logistic Regression']]


# In[92]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score , confusion_matrix
from sklearn.inspection import permutation_importance

for cls in classifiers:
    model = cls[0]
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("-----------------")
    print(cls[1])
    print ('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) *  100)
    print("Recall : ", recall_score(y_test, y_pred) *  100)
    print("Precision : ", precision_score(y_test, y_pred) *  100)
    print("F1 score : ", f1_score(y_test, y_pred) *  100)


# # Conclusion

# It appears that the Support Vector Classifier (SVC) has the highest accuracy, recall, precision, and F1 score among the classifiers evaluated.
# 
# * Accuracy :  92.10526315789474
# * Recall :  97.61904761904762
# * Precision :  89.13043478260869
# * F1 score :  93.18181818181817

# # Pickle

# In[47]:


from sklearn.svm import SVC
import pickle

# Create and train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Save the SVM model to a pickle file
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Load the saved SVM model from the pickle file
with open('svm_model.pkl', 'rb') as f:
    loaded_svm_model = pickle.load(f)

# Use the loaded SVM model for prediction tasks
y_pred = loaded_svm_model.predict(X_test)

# Print the results
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred) *  100)
print("Recall: ", recall_score(y_test, y_pred) *  100)
print("Precision: ", precision_score(y_test, y_pred) *  100)
print("F1 score: ", f1_score(y_test, y_pred) *  100)


# In[ ]:




