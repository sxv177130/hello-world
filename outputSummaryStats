import pandas as pd
import csv
import operator


#Read the dataset into a dataframe
df = pd.read_csv("C:/Users/sneha/Box Sync/UTD/Study Materials - UTD/Fall 2018/Amir - ML/Assignment-1/dataset.csv")
#print (df)

#Summary statistics for each variable
summaryStats = df.describe(include='all')
print(summaryStats)

#Transppose the dataframe
dfTranspose = df.describe().transpose()
print(dfTranspose)

#Median
median = df.median()
print(median)

#missing/Nan values
nanVals = df.count()
print(nanVals)

#drop the columns with more than 50% missings
finaldf = summaryStats.loc[:, df.isnull().sum() < 0.5*df.shape[0]]
print(finaldf)

#save as csv file
finaldf.to_csv('output.csv')
