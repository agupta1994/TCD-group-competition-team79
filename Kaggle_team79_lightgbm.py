# Imported the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Added data in CSV format using pandas library
df = pd.read_csv('tcd-ml-1920-group-income-train.csv')
Test_Data=pd.read_csv('tcd-ml-1920-group-income-test.csv')

# Concatinated test data and train data
df=pd.concat([df,Test_Data], sort=False)

# Renamed the Total Yearly Income [EUR] to Income
df = df.rename(index=str, columns={"Total Yearly Income [EUR]": "Income"})

# Dropped Instance column from the merged data
df = df.drop("Instance", axis=1)

# Taken log of Income column
df["Income"] = np.log(df["Income"])

# Removed all NaN values from data using forward fill for all the categorical columns and taking median for numeric columns
df['Year of Record'] = df['Year of Record'].fillna(df['Year of Record'].median())
df['Satisfation with employer'] = df['Satisfation with employer'].fillna(method='ffill')
df['Gender'] = df['Gender'].fillna(method='ffill')
df['Country'] = df['Country'].fillna(method='ffill')
df['Profession'] = df['Profession'].fillna(method='ffill')
df['University Degree'] = df['University Degree'].fillna(method='ffill')
df['Hair Color'] = df['Hair Color'].fillna(method='ffill')
df.head()

# Replaced garbage values with other values
df['Housing Situation'] = df['Housing Situation'].replace(0,'other')
df['Housing Situation'] = df['Housing Situation'].replace('nA','other')

# Replaced string with number
df['Work Experience in Current Job [years]'] = df['Work Experience in Current Job [years]'].replace('#NUM!',0)

# Converted object datatype to float datatype
df['Work Experience in Current Job [years]'] = df['Work Experience in Current Job [years]'].astype(float)

# Removed EUR from all the values of Yearly Income in addition to Salary (e.g. Rental Income) and converted it to numeric
df['Yearly Income in addition to Salary (e.g. Rental Income)'] = df['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace('([A-Za-z]+)', '')
df['Yearly Income in addition to Salary (e.g. Rental Income)'] = pd.to_numeric(df['Yearly Income in addition to Salary (e.g. Rental Income)'],errors='coerce')

# Replaced all garbage values with other value
df['Gender'] = df['Gender'].replace(0,'other')
df['Country'] = df['Country'].replace(0,'other')
df['University Degree'] = df['University Degree'].replace(0,'other')
df['Hair Color'] = df['Hair Color'].replace(0,'other')
df.head()

# Encoding of categorical columns using get_dummies of pandas library
data2 = pd.get_dummies(df, columns=["Gender"])
data2 = pd.get_dummies(data2, columns=["Country"])
data2 = pd.get_dummies(data2, columns=["Profession"])
data2 = pd.get_dummies(data2, columns=["University Degree"])
data2 = pd.get_dummies(data2, columns=["Hair Color"])
data2 = pd.get_dummies(data2, columns=["Housing Situation"])
data2 = pd.get_dummies(data2, columns=["Satisfation with employer"])

# Divided the data into training part
train = data2[0:1048574]
Y = train.Income
X = train.drop("Income", axis=1)

# Test and Train data split using sklearn library
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
X.head()

# Used Lightgbm Regressor to train the data
import lightgbm as lgb
params = {
          'max_depth': 20,
          'learning_rate': 0.04,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mse',
          "verbosity": -1,
         }
train_data = lgb.Dataset(X_train, Y_train)
test_data = lgb.Dataset(X_test, Y_test)
model=lgb.train(params, train_data, 1000, valid_sets = [train_data, test_data], verbose_eval=1000, early_stopping_rounds=500)

# Predicted using the trained model
Y_pred = model.predict(X_test)

# Found Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, Y_pred)
mae

# Used second part of data for prediction of test data
test = data2[1048574:]
X_testing = test.drop("Income", axis=1)
y_sub = model.predict(X_testing)

# Taken exponential of the predicted Income
y_final = np.exp(y_sub)

# Stored the predicted income in a csv format
test['Income'] = y_final
test.to_csv('output.csv', columns=['Income'])
from IPython.display import FileLink
FileLink("output.csv")
