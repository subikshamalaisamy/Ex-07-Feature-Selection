# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df.head()

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df.head()

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df.head()

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):
  
  p= []
  
  X_1 = X[cols]
  
  X_1 = sm.add_constant(X_1)
  
  model = sm.OLS(y,X_1).fit()
  
  p = pd.Series(model.pvalues.values[1:],index = cols)  
  
  pmax = max(p)
  
  feature_with_p_max = p.idxmax()
  
  if(pmax>0.05):
    
    cols.remove(feature_with_p_max)    
  
  else: 
    
    break

selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):
  
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
  
  model = LinearRegression()
  
  rfe = RFE(model,step=nof_list[n])
  
  X_train_rfe = rfe.fit_transform(X_train,y_train)
  
  X_test_rfe = rfe.transform(X_test)
  
  model.fit(X_train_rfe,y_train)
  
  score = model.score(X_test_rfe,y_test)
  
  score_list.append(score)
  
  if(score>high_score):
    
    high_score = score    
    
    nof = nof_list[n]
    
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()


# OUPUT
<img width="853" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/861c1f85-82d4-47a7-b9ee-23f0459f6b63">
<img width="122" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/40612a9a-8d65-4c60-a8e9-89d514f1c7ab">
<img width="344" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/fcc17d65-5bef-4092-b6f0-a2a66f6e20b3">
<img width="331" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/996bd37f-95c0-4916-996d-d66480622115">
<img width="379" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/d5f928ea-471b-48b7-8f4d-2541417f4d99">
<img width="396" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/356c0ce4-b2d8-4be1-90c5-226cf6ba6e23">
<img width="345" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/30ad5ecd-468e-4a8b-be22-e458d24439a6">
<img width="302" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/47a04bb0-d7a6-4542-b4d4-5e95e2b17b89">
<img width="298" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/032326a2-b4d9-44c0-bb57-df07b7bb482e">
<img width="139" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/024238cf-d5c1-472c-9793-d53e2279724c">
<img width="368" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/f0724d3e-4f9e-45cc-8707-3d94a24353ff">
<img width="229" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/ce944fb8-d123-46a0-8a75-8819eb15a016">
<img width="162" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/6f76ad48-20d5-4fda-b964-c550dc0ef98a">
<img width="181" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/459aad08-fbe7-4cfe-b4c5-d487115fabd8">
<img width="193" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/f1be8ee9-5651-4f0a-8a82-05ae2e5b1806">
<img width="276" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/eed30116-0cee-495d-9fb1-6a6ea9d445b5">
<img width="244" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/c3141472-9bd3-4a3a-bf96-7eded4b41acb">
<img width="284" alt="image" src="https://github.com/subikshamalaisamy/Ex-07-Feature-Selection/assets/87276633/0500646c-3d51-4c01-a4d0-faaae767dd3e">

# RESULT

Thus the various feature selection techniques for the given dataset is performed and saved successfully.
