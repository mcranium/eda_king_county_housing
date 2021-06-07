import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse
import pickle

import joblib

housing = pd.read_csv('data/King_County_House_prices_dataset.csv')

# one house apperently had 33 bedrooms but only 1.75 bathrooms
housing.bedrooms.replace(33, np.nan, inplace=True)
# one house apperently was build in the birth year of Jesus Christ
housing.yr_renovated.replace(np.NaN, 0, inplace=True)

housing.sqft_basement.replace('?', 0, inplace=True)
housing.sqft_basement = pd.Series(housing.sqft_basement, dtype='float64')

housing.date = pd.to_datetime(housing.date)

housing.zipcode = pd.Series(housing.zipcode, dtype='category')

yr_built_or_renovated = []
for row in range(housing.shape[0]):
    if housing.yr_renovated[row] > housing.yr_built[row]:
        yr_built_or_renovated.append(housing.yr_renovated[row])
housing['yr_built_or_renovated'] = pd.Series(yr_built_or_renovated)

condition_binary = []
for row in range(housing.shape[0]):
    if housing.condition[row] >= 3:
        condition_binary.append(1)
    else:
        condition_binary.append(0)
housing['condition_binary'] = pd.Series(condition_binary)

X = housing[[
                # helper variables
                'price',
                'zipcode',
                # variables to fit the model
                'bedrooms',
                'bathrooms',
                'sqft_living',
                'waterfront',
                'condition_binary',
                'grade',
                'yr_built_or_renovated',
                'view',
                'sqft_lot',
                'sqft_above',
                'sqft_basement',
                'sqft_living15'
            ]]

X = X.dropna(axis=0)

y = X.price

X = X.drop('price', axis=1)

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

price_sqft = y_train / X_train.sqft_lot
X_train = X_train.assign(price_sqft=price_sqft)

avg_price_sqft_zipcode_dict = X_train.groupby('zipcode').mean().price_sqft.reindex().to_dict()
avg_price_sqft_zipcode = X_train.zipcode.map(avg_price_sqft_zipcode_dict)
X_train = X_train.assign(avg_price_sqft_zipcode = avg_price_sqft_zipcode)
X_train.drop('price_sqft', axis=1, inplace=True)

avg_price_sqft_zipcode = X_test.zipcode.map(avg_price_sqft_zipcode_dict)
X_test = X_test.assign(avg_price_sqft_zipcode = avg_price_sqft_zipcode)

X_test = X_test.dropna(axis=0)
y_test = y_test.dropna(axis=0)
X_train = X_train.dropna(axis=0)
y_train = y_train.dropna(axis=0)

X_test = X_test.join(y_test, how='inner')
X_train = X_train.join(y_train, how='inner')

y_test = X_test.price
y_train = X_train.price
X_test = X_test.drop('price', axis=1)
X_train = X_train.drop('price', axis=1)

model = sm.OLS(y_train, X_train)
fitted = model.fit()

print(fitted.summary())

y_test_pred = fitted.predict(X_test)
print('\n')
print('The RMSE is: ', rmse(y_test, y_test_pred))

joblib.dump(fitted, 'fitted_model.pickle')
