# -*- coding: utf-8 -*-
"""
Data Preprocessing

"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

titanic = sns.load_dataset("titanic")

#familiarize with the dataset
titanic.shape #dimension of the dataset
titanic.describe()
titanic.head() #visualize first 5 rows of data
titanic.dtypes #list out all columns and their types

#checking and handling missing data
titanic.isna().any() #columns with any NaN value
titanic.isna().sum() #total number of NaN values for each column
titanic.dropna(axis=1,inplace=True) #deletes column with any NaN value
titanic.dropna(axis=0,inplace=True) #deletes rows with any NaN value
titanic["age"].fillna(titanic["age"].mean(),inplace=True) #replace all missing age values with mean
titanic["age"].fillna(method='bfill',inplace=True) #replace all missing age values with previous valid value

#below is a function that uses a logic to replace missing ages based on name title. 
def age_plug(df):
    age_derived = [] # assigning age to misisng rows
    for i in range(len(df)):
        if np.isnan(np.array(df["Age"][i])):
            if df["Name"][i].split()[1].strip() in ["Master.","Miss."]:
                age_derived.append(np.float64(np.random.randint(1,18)))
            else:
                age_derived.append(np.float64(np.random.randint(18,60)))
        else:
            age_derived.append(df["Age"][i])
    df["Age"] = np.array(age_derived)
    return df

age_plug(titanic)

#outlier visualization
titanic["fare"].hist() #visualizing the histogram of a given feature
plt.scatter(titanic["fare"].index,titanic["fare"]) #visualizing scatter plot


#covariance matrix
titanic.cov()

#correlation matrix  
titanic.corr()

#chi square value
chi2 = chi2_contingency(titanic.iloc[:,[0,1]])[0] #select the categorical variables in the dataframe and apply the formula 