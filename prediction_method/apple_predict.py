import pandas_datareader as web
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle

df = web.DataReader('AMZN', data_source='yahoo', start='2012-01-01', end='2020-01-20')
df
count = 0
for i in range(len(df)):
    if df['Close'][i] == df['Adj Close'][i]:
        count = count + 1
count
df['Adj High'] = df['High']
df['Adj Low'] = df['Low']
df['Adj Open'] = df['Open']
df['Adj Volume'] = df['Volume']

df = df[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']]

df['HL_PCT'] = (df['Adj High'] - df['Adj Close'])/ df['Adj Close'] * 100
df['PCT_change'] = (df['Adj Close'] - df['Adj Open'])/ df['Adj Open'] * 100

df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Adj Volume']]

forecast_col = 'Adj Close'
forecast_col

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['Label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
df

x = np.array(df.drop(['Label'], axis = 1))
y = np.array(df['Label'])

# x = x[:-forecast_out+1]
# df.dropna(inplace = True)
y = np.array(df['Label'])
print(len(x), len(y))

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)

clf = LinearRegression(n_jobs=10)
print(clf.fit(train_x, train_y))

print(clf.score(test_x, test_y))

print(forecast_out)

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

score = cross_val_score(clf, x, y, cv= cv)
print("score" + str(score))

np.mean(score)

#test_X = np.array(test_x)
#test_y = np.array(test_y)
#d1 = pd.DataFrame([test_x, test_y])
filename = 'amazon_predict.sav'
pickle.dump(clf, open(filename, 'wb'))

pd.DataFrame(clf.predict(test_x), test_y)