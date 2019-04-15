#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:23:47 2019

@author: Sabrina
"""

# =============================================================================
# Using KNN to predict car price
# =============================================================================

# =============================================================================
# Explore data and data cleaning
# =============================================================================
import pandas as pd
import numpy as np

headers = ['symboling', 'normalized-losses', ' make',  'fuel-type', 
           'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 
           'engine-location', 'wheel-base', 'length', 'width', 'height',
           'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 
           'fuel-system', 'bore','stroke', 'compression-ratio', 'horsepower', 
           'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

pd.options.display.max_columns = 30

cars = pd.read_csv('imports-85.data', names= headers)
print(cars.shape)
print(cars.head())

# Select numeric data
numeric_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height',
           'curb-weight', 'engine-size', 'bore','stroke', 'compression-ratio', 
           'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[numeric_cols]

print(numeric_cars.head())
print(numeric_cars.info())

num_cars = numeric_cars.copy()
num_cars = num_cars.replace("?", np.nan).astype(float)
print(num_cars.isnull().sum())

# removing missing values in price column
num_cars.dropna(subset = ['price'], inplace=True)
# fill with mean in other columns
num_cars.fillna(np.mean(num_cars), inplace=True)

print(num_cars.isnull().sum())

# Normalize data in each column except price
norm_cars = (num_cars - num_cars.min())/(num_cars.max()-num_cars.min())
norm_cars['price'] = num_cars['price']
print(norm_cars.head())

# =============================================================================
# Build Model 
# =============================================================================

# =============================================================================
# Univariate model:
#  - same features, same k values
#  - same features, different k values
# =============================================================================

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# k = 5
def knn_train_test(train_col, target_col, df):
    # randomly shuffle index 
    np.random.seed(1)
    shuffle_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffle_index)
    
    # set up train/test df
    div = int(rand_df.shape[0]/2)
    train_df = rand_df.iloc[:div].copy()
    test_df = rand_df.iloc[div:].copy()
    
    # train data
    knn = KNeighborsRegressor()
    knn.fit(train_df[[train_col]], train_df[target_col])
    
    # predictions
    predictions = knn.predict(test_df[[train_col]])
    
    # check errors
    mse = mean_squared_error(test_df[target_col], predictions)
    rmse = np.sqrt(mse)
    
    return rmse

features_rmses ={}
features = norm_cars.columns.drop('price')

for feature in features:
    rmse_per_feature = knn_train_test(feature, 'price', norm_cars)
    features_rmses[feature] = rmse_per_feature
    
features_rmses_series = pd.Series(features_rmses)
print(features_rmses_series.sort_values())

# a list of k values
def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
    shuffle_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffle_index)
    
    div = int(rand_df.shape[0]/2)
    train_df = rand_df.iloc[:div].copy()
    test_df = rand_df.iloc[div:].copy()
    
    k_values = [k for k in range(1, 10, 2)]
    k_rmse = {}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors = k)
        knn.fit(train_df[[train_col]], train_df[target_col])
        predictions = knn.predict(test_df[[train_col]])
        mse = mean_squared_error(test_df[target_col], predictions)
        rmse = np.sqrt(mse)
        k_rmse[k] = rmse
    return k_rmse

features_k_rmse = {}
features = norm_cars.columns.drop('price')
for feature in features:
    k_rmse_per_feature = knn_train_test(feature, 'price', norm_cars)
    features_k_rmse[feature] = k_rmse_per_feature
    
print(features_k_rmse)

import matplotlib.pyplot as plt
for k, v in features_k_rmse.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x, y)
    plt.xlabel('k values')
    plt.ylabel('RMSE')
    plt.title('Different k value across features')

plt.show()
    

# =============================================================================
# Multivariate Mode
#  -  same k values, different attributes
#  -  different k values, different attributes
# =============================================================================

# calculate avg_rmse when k  = [k for k in range(1, 10, 2)]
avg_rmse_per_feature = {}
for k, v in features_k_rmse.items():
    k_avg_rmse = np.mean(list(v.values()))
    avg_rmse_per_feature[k] = k_avg_rmse
    
print(avg_rmse_per_feature)

def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
    shuffle_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffle_index)
    
    div = int(rand_df.shape[0]/2)
    train_df = rand_df.iloc[:div].copy()
    test_df = rand_df.iloc[div:].copy()
    
    k_values = [5]
    k_rmses = {}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors = k)
        knn.fit(train_df[train_col], train_df[target_col])
        predictions = knn.predict(test_df[train_col])
        mse = mean_squared_error(test_df[target_col], predictions)
        rmse = np.sqrt(mse)
        k_rmses[k] = rmse
        
    return k_rmses
    
k_rmses_attributes = {}
attri2 = ['engine-size', 'horsepower']
k_rmses_attributes['two best features'] = knn_train_test(attri2, 'price', norm_cars)

attri3 = ['engine-size', 'horsepower', 'width']
k_rmses_attributes['three best features'] = knn_train_test(attri3, 'price', norm_cars)

attri4 = ['engine-size', 'horsepower', 'width', 'curb-weight']
k_rmses_attributes['four best features'] = knn_train_test(attri4, 'price', norm_cars)

attri5 = ['engine-size', 'horsepower', 'width', 'curb-weight', 'highway-mpg']
k_rmses_attributes['five best features'] = knn_train_test(attri5, 'price', norm_cars)

attri6 = ['engine-size', 'horsepower', 'width', 'curb-weight', 'highway-mpg', 'length']
k_rmses_attributes['six best features'] = knn_train_test(attri6, 'price', norm_cars)

print(k_rmses_attributes)

# Different k with different attributes
def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
    shuffle_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffle_index)
    
    div = int(rand_df.shape[0]/2)
    train_df = rand_df.iloc[:div].copy()
    test_df = rand_df.iloc[div:].copy()
    
    k_values = [k for k in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors = k)
        knn.fit(train_df[train_col], train_df[target_col])
        predictions = knn.predict(test_df[train_col])
        mse = mean_squared_error(test_df[target_col], predictions)
        rmse = np.sqrt(mse)
        k_rmses[k] = rmse
        
    return k_rmses

k_rmses_attributes = {}
attri2 = ['engine-size', 'horsepower']
k_rmses_attributes['two best features'] = knn_train_test(attri2, 'price', norm_cars)

attri3 = ['engine-size', 'horsepower', 'width']
k_rmses_attributes['three best features'] = knn_train_test(attri3, 'price', norm_cars)

attri4 = ['engine-size', 'horsepower', 'width', 'curb-weight']
k_rmses_attributes['four best features'] = knn_train_test(attri4, 'price', norm_cars)

attri5 = ['engine-size', 'horsepower', 'width', 'curb-weight', 'highway-mpg']
k_rmses_attributes['five best features'] = knn_train_test(attri5, 'price', norm_cars)

attri6 = ['engine-size', 'horsepower', 'width', 'curb-weight', 'highway-mpg', 'length']
k_rmses_attributes['six best features'] = knn_train_test(attri6, 'price', norm_cars)

print(k_rmses_attributes)

for k, v, in k_rmses_attributes.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x, y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')
    plt.title('RMSE with different features across k values')
plt.show()

# =============================================================================
# Cross Validation
# =============================================================================

# split into 5 subsets with 2 attributes
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits= 5, shuffle=True, random_state =1)
model = KNeighborsRegressor()
mses = cross_val_score(model, 
                       norm_cars[['engine-size', 'horsepower']],
                       norm_cars['price'],
                       scoring = 'neg_mean_squared_error',
                       cv = kf)
rmses = np.sqrt(np.abs(mses))
avg_kfold_rmse = np.mean(rmses)
print(rmses)
print(avg_kfold_rmse)


# 2 attributes across a list of folds
from sklearn.model_selection import KFold, cross_val_score
num_folds = [k for k in range(3, 25)]

for fold in num_folds:
    kf = KFold(n_splits = fold, shuffle=True, random_state=1)
    model = KNeighborsRegressor()
    mses = cross_val_score(model, 
                           norm_cars[['engine-size', 'horsepower']],
                           norm_cars['price'],
                           scoring='neg_mean_squared_error',
                           cv=kf)
    rmses = np.sqrt(np.abs(mses))
    avg_rmse_per_fold = np.mean(rmses)
    std_rmse_per_fold = np.std(rmses)
    print(str(fold), 'folds:', 'avg_rmse_per_fold:', avg_rmse_per_fold, 'std_rmse_per_fold:',  std_rmse_per_fold)