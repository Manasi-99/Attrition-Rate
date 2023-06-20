#!/usr/bin/env python
# coding: utf-8

# In[2]:


# IST 652 Scripting for Data Analysis
# FINAL PROJECT
# Company Attrition Prediction

# Author: Manasi Todankar

# Reference for overview of the Project: https://ai.plainenglish.io/iris-flower-classification-step-by-step-tutorial-c8728300dc9e


# In[74]:


# Importing libraries

import numpy as np
import pandas as pd
import seaborn as sns


# In[75]:


# Loading the CSV file
df = pd.read_csv(r'C:\Users\todan\OneDrive\IST 652\Data\Company Atrrition Data.csv')
df


# In[ ]:


# ------------------------- DATA EXPLORATION ------------------------------- #


# In[76]:


# Printing the first 7 rows of data to get a brief idea about the dataset.
df.head(7)


# In[77]:


# Dimensions of the dataframe
df.shape


# In[78]:


# Column datatypes
# Printing the column dataset to get to know the data. To understand with what kind of data I am working with.
# If I need to make any changes to initial data types for data processing.

df.dtypes


# In[79]:


# Checking for empty cells in every column.

df.isna().sum()


# In[ ]:


# We can observe that 'NumCompaniesWorked', 'EnvironmentSatisfaction', 'JobStatisfaction', 'WorkLifeBalance' columns
# contain na values. As all these columns are numeric, I will use the mean of those respective columns to impute values.


# In[80]:


# Imputing the values with the mean of their respective columns

from sklearn.impute import SimpleImputer

imputeCol = SimpleImputer(strategy = 'mean')
imputeCol = imputeCol.fit(df[['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']])
df[['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']] = imputeCol.transform(df[['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']])


# In[46]:


# Confirming that there are no more null values
df.isna().sum()


# In[81]:


# Lastly checking for any blank/null values in the data
df.isnull().values.any()


# In[82]:


# Descriptive Statistics

# Getting the Descriptive Statistics to check for any abnormal data.
# To check any outliers.
df.describe()


# In[ ]:


# In the descriptive statistics, we can see that there is no abnormal data.
# There are no suspisious negative values.
# The minimum and maximum values of the columns make sense to what they represent. 
# Thus, there is no need to treat the data for outliers or false entry of data.


# In[83]:


# For the purpose of avoiding a certain error in future.
df['BusinessTravel'] = df['BusinessTravel'].str.replace('_', '.')
df['BusinessTravel'] = df['BusinessTravel'].str.replace('-', '.')
df


# In[84]:


# Getting the count of people who left the company and the people who contnued working
df['Attrition'].value_counts()


# In[85]:


# Barplot of the Attrition values

import matplotlib.pyplot as plt

sns.countplot(x = df['Attrition'])
plt.show()


# In[86]:


# We can observe in the above barplot that the values are lopsided.
# We have a great number of 'No' compared to 'Yes'. 
# To check the influence of these number on the machine learning model we perform a small calculation.

# We are subtracting the 'Yes' from 'No' and then dividing the by number of 'Yes'.

# So just by guessing 'No', the accuracy of the model will be 80.77%

# Now, we will basically look for a model that has a higher accuracy than 80.77%.

(3699 - 711) /3699


# In[87]:


# Number of employees that left and stayed by age.

plt.subplots(figsize = (16,4))
sns.countplot(x = 'Age', hue = 'Attrition', data = df)
plt.show()


# In[88]:


# Attrition number in the different departments of the Company.

plt.subplots(figsize = (16,4))
sns.countplot(x = 'Department', hue = 'Attrition', data = df)
plt.show()


# In[89]:


# Comparing the number of years spent at the company vs their ages.
# This helps us in understanding if the employee left the company near its retirement age or very early on.
# So that we can focus on the employees leaving early on in their career.

import seaborn

plt.subplots(figsize = (16,7))
seaborn.scatterplot(x = 'Age', y = 'YearsAtCompany', hue = 'Attrition', data = df, palette = 'pastel')
plt.xlabel('Employee Age')
plt.ylabel('Years at the Company')
plt.legend(title = 'Attrition')
plt.show()


# In[90]:


# Getting the range of years spent at the Company.

print("Number of Years at the company varies from {} to {} years.".format(
    df['YearsAtCompany'].min(), df['YearsAtCompany'].max()))


# In[91]:


# Reference: https://seaborn.pydata.org/generated/seaborn.kdeplot.html

# Visualing the numbers

import seaborn as sns

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(data = df, x = df['YearsAtCompany'], hue = 'Attrition', palette = 'pastel')
plt.xlabel('YearsAtCompany')
plt.xlim(left=0)
plt.xlabel('Years at the Company')
plt.ylabel('Density')
plt.title('Years At Company in Percent by Attrition Status')
plt.show()


# In[92]:


# Range of Number of years with the present manager.

print("Number of Years with the current manager varies from {} to {} years.".format(
    df['YearsWithCurrManager'].min(), df['YearsWithCurrManager'].max()))


# In[93]:


# Visualizing the number

plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(data = df, x = df['YearsWithCurrManager'], hue = 'Attrition', palette = 'pastel')
plt.xlabel('YearsWithCurrManager')
plt.xlim(left=0)
plt.ylabel('Density')
plt.xlabel('Years with the current Manager')
plt.title('Years At Company in Percent by Attrition Status')
plt.show()


# In[94]:


# Printing all the datatypes and their uniques values as a part of preprocessing the data.

for column in df.columns:
  if df[column].dtype == object:
    print(str(column) + ':' + str(df[column].unique()))
    print(df[column].value_counts())
    print('-------------------------------------------------')


# In[96]:


df['EmployeeCount'].unique()


# In[97]:


df['Over18'].unique()


# In[98]:


df['StandardHours'].unique()


# In[99]:


# Removing useless columns

# Over18: By default all the employees are above 18 years.
# EmployeeCount: Employee count does not contribute towards our study. As it does not tell naything about the data.
# Standard Hours are by default 8hr/day.
# Even the Employee ID is a redundant column.

df = df.drop('Over18', axis = 1)
df = df.drop('EmployeeCount', axis = 1)
df = df.drop('StandardHours', axis = 1)
df = df.drop('EmployeeID', axis = 1)


# In[100]:


# Confirming the drop operation on the dataframe.

df.shape


# In[101]:


# Getting the correlation between the attributes

df.corr()


# In[102]:


# Correlation in graphical form

plt.figure(figsize=(18,16))
sns.heatmap(df.corr(), annot = True, fmt = '.0%')
plt.show()


# In[103]:


# Machine Learning------------
# Data preparation

# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

# Transforming non-numerical columns into numerical columns.

from sklearn.preprocessing import LabelEncoder

for column in df.columns:
  if df[column].dtype == np.number:
    continue
  df[column] = LabelEncoder().fit_transform(df[column])


# In[104]:


# Just for the ease of dividing the dataset into training and test datasets, I want to move the age column as the first column.
# And keep the Attrition response column as the first column.

df['EmployeeAge'] = df['Age']


# In[105]:


df = df.drop('Age', axis = 1)
df


# In[106]:


# Splitting the dataset

X = df.iloc[:, 1: df.shape[1]].values
Y = df.iloc[:, 0].values


# In[107]:


# Splitting the dataset for modelling
# 75% for training
# 25% for testing

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[108]:


# Importing all the packages we will need for implementing the Machine Learning models.

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# In[109]:


# Defining the Machine learning model functions and giving parameters where necessary

# c: trust value. Keeping it low to give more weight to this complexity penalty.
logistic_reg = LogisticRegression(C = 0.1, random_state = 42, solver = 'liblinear')

decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
gaussian_nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)
svm = svm.SVC(kernel='linear')


# In[110]:


# Importing sklearn.metrics package
from sklearn.metrics import accuracy_score

# Getting the accuracy for multiple models.
for a,b in zip([logistic_reg, decision_tree, random_forest, gaussian_nb, knn, svm],["Logistic Regression","Decision Tree","Random Forest","Naive Bayes","KNN","SVM"]):
    a.fit(X_train,Y_train)
    prediction = a.predict(X_train)
    Y_pred = a.predict(X_test)
    score_train = accuracy_score(Y_train,prediction)
    score_test = accuracy_score(Y_test,Y_pred)
    training_accu = "[%s] training data accuracy is : %f" % (b,score_train)
    test_accu = "[%s] test data accuracy is : %f" % (b,score_test)
    print(training_accu)
    print(test_accu)
    
# Reference: scikit-learn: machine learning in Python


# In[111]:


# Filtering out the model scores on test data only.
model_scores={'Logistic Regression':logistic_reg.score(X_test,Y_test),
              'Decision tree':decision_tree.score(X_test,Y_test),
              'Random forest':random_forest.score(X_test,Y_test),
              'Naive Bayes':gaussian_nb.score(X_test,Y_test),
              'KNN classifier':knn.score(X_test,Y_test),
              'Support Vector Machine':svm.score(X_test,Y_test)
             }
model_scores


# In[ ]:


# We can see that Random Forest Tree has the highest accuracy followed by Decision Tree.
# Thus we will be using these two algorithms.


# In[112]:


# Using Random Forest Classifier Algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)


# In[113]:


# As previouslt predicted, the score in 1.0
forest.score(X_train, Y_train)


# In[115]:


# Confusion Matrix and accuracy score for the model on the test data

#                                    Predicted Values
#                                Negative(0)      Positive(0)    
#                    Negative(0)         TN      |       FP
#     Actual Values                  ---------------------
#                    Positive(1)        FN      |       TP


# Confusion Matrix for Random Forest Tree Algorithm
from sklearn.metrics import confusion_matrix

conf_matrix_RF = confusion_matrix(Y_test, forest.predict(X_test))

TN_RF = conf_matrix_RF[0][0]
TP_RF = conf_matrix_RF[1][1]
FN_RF = conf_matrix_RF[1][0]
FP_RF = conf_matrix_RF[0][1]

print(conf_matrix_RF)
print('Model Testing accuracy for Random Forest Algorithm = {}'.format((TP_RF + TN_RF) / (TP_RF + FP_RF + FN_RF + TN_RF))) 


# In[116]:


# Visulaizing the Confusion Matrix
plt.figure(figsize = (7,5))
sns.heatmap(conf_matrix_RF, annot=True, fmt='g')
plt.show()


# In[117]:


# Random Forest Algorithm Classification Report

from sklearn.metrics import classification_report

random_forest_CR = random_forest.predict(X_test)
print(classification_report(Y_test, random_forest_CR))


# In[ ]:


#------------------------------------------------------------------


# In[118]:


# Decision Tree algorithm

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, Y_train)


# In[119]:


# As predicted previously
classifier.score(X_train, Y_train)


# In[120]:


# Confusion Matrix for Decision Tree Algorithm

conf_matrix_DT = confusion_matrix(Y_test, classifier.predict(X_test))

TN_DT = conf_matrix_DT[0][0]
TP_DT = conf_matrix_DT[1][1]
FN_DT = conf_matrix_DT[1][0]
FP_DT = conf_matrix_DT[0][1]

print(conf_matrix_DT)
print('Model Testing accuracy for Decision Tree Algorithm = {}'.format((TP_DT + TN_DT) / (TP_DT + FP_DT + FN_DT + TN_DT))) 


# In[121]:


# Visualizing the Confusion Matrix
plt.figure(figsize = (7,5))
sns.heatmap(conf_matrix_DT, annot=True, fmt='g')
plt.show()


# In[122]:


# Decision Tree Algorithm Classification Report

from sklearn.metrics import classification_report

decision_tree_CR = decision_tree.predict(X_test)
print(classification_report(Y_test, decision_tree_CR))


# In[124]:



import graphviz
from sklearn import tree

feature_names = df.columns[:24]
target_names = df['Attrition'].unique().tolist()


viz = tree.export_graphviz(classifier, out_file=None, 
                                feature_names = feature_names,  
                                class_names = str(target_names),
                                filled=True)

# Draw graph
graph = graphviz.Source(viz, format="png") 
graph

