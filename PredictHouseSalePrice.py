#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:23:37 2019

@author: Sabrina
"""

# =============================================================================
# PREDICT HOUSE SALE PRICE
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

pd.options.display.max_columns = 999
df = pd.read_csv('AmesHousing.tsv', sep = '\t')
print(df.shape)
print(df.info())


# =============================================================================
# General Workframe: 
# -> Feature Engineer -> Feature Selection  -> Train n' Test
# =============================================================================

def transform_features (df):
    return df

def features_selection (df):
    features = df[['Gr Liv Area', 'SalePrice']]
    return features

def train_and_test (df):
    train = df.iloc[:1460]
    test = df.iloc[1460:]
    
    num_train = train.select_dtypes(include=['int', 'float'])
    
    features = num_train.columns.drop('SalePrice')
    target = 'SalePrice'
    
    lr = LinearRegression()
    lr.fit(train[features], train[target])
    predictions = lr.predict(test[features])
    
    mse = mean_squared_error(test[target], predictions)
    rmse = np.sqrt(mse)
    return rmse

trans_features = transform_features(df)
select_features = features_selection(trans_features)
rmse = train_and_test(select_features)

print(rmse)

# =============================================================================
# 1. FEATURE ENGINEER
# =============================================================================

# remove cols with no. of missing value > 5% of that column    
missing_values = df.isnull().sum()
remove_all = missing_values[missing_values > (len(df)*0.05)]
df = df.drop(remove_all.index, axis =1)

# remove any text col with no. of missing value >1
df_txt = df.select_dtypes(include=['object'])
missing_txt = df_txt.isnull().sum()
remove_txt = missing_txt[missing_txt>0]
df = df.drop(remove_txt.index, axis = 1)

# fill missing value
missing_count = df.isnull().sum().sort_values(ascending=False)
value_missing = missing_count[missing_count>0]
values_to_fill = df[value_missing.index].mode().to_dict(orient='record')[0]
df = df.fillna(values_to_fill)

# create new features
years_owned = df['Yr Sold'] - df['Year Built']
outliers1 = years_owned[years_owned<0]
years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
outliers2 = years_since_remod[years_since_remod<0]
print(outliers1)
print(outliers2)

# remove outliers
df = df.drop([1702, 2180, 2181], axis=0)
# add new features
df['Years Before Sale'] = years_owned
df['Years Since Remod'] = years_since_remod
# remove old features
df = df.drop(['Yr Sold', 'Year Built', 'Year Remod/Add'], axis = 1)

# remove useless cols
print(df.columns)
df = df.drop(['Order', 'PID'], axis = 1)
# remove cols have data leakage problem
df = df.drop(['Mo Sold', 'Sale Type', 'Sale Condition'], axis =1)

# =============================================================================

# Update the Transform_features Function 

def transform_features(df):
    # remove cols with no. of missing value > 5% of that column    
    missing_values = df.isnull().sum()
    remove_all = missing_values[missing_values > (len(df)*0.05)]
    df = df.drop(remove_all.index, axis =1)

    # remove any text col with no. of missing value >1 (not necessary)
#    df_txt = df.select_dtypes(include=['object'])
#    missing_txt = df_txt.isnull().sum()
#    remove_txt = missing_txt[missing_txt>0]
#    df = df.drop(remove_txt.index, axis = 1)

    # fill missing value
    missing_count = df.isnull().sum().sort_values(ascending=False)
    value_missing = missing_count[missing_count>0]
    values_to_fill = df[value_missing.index].mode().to_dict(orient='record')[0]
    df = df.fillna(values_to_fill)

    # create new features
    years_owned = df['Yr Sold'] - df['Year Built']
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    
    # remove outliers where years_owned and years_since_remod <0
    df = df.drop([1702, 2180, 2181], axis=0)
    # add new features
    df['Years Before Sale'] = years_owned
    df['Years Since Remod'] = years_since_remod
    # remove old features
    df = df.drop(['Yr Sold', 'Year Built', 'Year Remod/Add'], axis = 1)

    # remove useless cols                
    df = df.drop(['Order', 'PID'], axis = 1)
    # remove cols have data leakage problem
    df = df.drop(['Mo Sold', 'Sale Type', 'Sale Condition'], axis =1)

    return df

def features_selection (df):
    features = df[['Gr Liv Area', 'SalePrice']]
    return features

def train_and_test (df):
    train = df.iloc[:1460]
    test = df.iloc[1460:]
    
    num_train = train.select_dtypes(include=['int', 'float'])
    
    features = num_train.columns.drop('SalePrice')
    target = 'SalePrice'
    
    lr = LinearRegression()
    lr.fit(train[features], train[target])
    predictions = lr.predict(test[features])
    
    mse = mean_squared_error(test[target], predictions)
    rmse = np.sqrt(mse)
    return rmse

df = pd.read_csv('AmesHousing.tsv', sep = '\t')
trans_df = transform_features(df)
select_features = features_selection(trans_df)
rmse = train_and_test(select_features)

print(rmse)

# =============================================================================
# 2. FEATURE SELECTION 
# =============================================================================
# numerical features
num_df = trans_df.select_dtypes(include=['int', 'float'])
corr_coeff = num_df.corr()['SalePrice'].abs().sort_values(ascending=False)
remove_features1 = corr_coeff[corr_coeff<0.4]
trans_df = trans_df.drop(remove_features1.index, axis = 1)

# categorical features

# nominal features

norminal_cols = [   "MS SubClass", "MS Zoning", "Street", "Alley", 
                    "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", 
                    "House Style", "Roof Style", "Roof Matl", 
                    "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", 
                    "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]

# filter out non-norminal cols
trans_cat_cols = []
for col in norminal_cols:
    if col in trans_df.columns:
        trans_cat_cols.append(col)

col_unique = trans_df[trans_cat_cols].apply(lambda c: len(c.value_counts()))

# filiter out norminal features with more than 10 categories
remove_features2 = col_unique[col_unique>10]
trans_df = trans_df.drop(remove_features2.index, axis =1)

# convert norminal to categorical 
text_col = trans_df.select_dtypes(include=['object'])
for col in text_col:
    trans_df[col] = trans_df[col].astype('category')

trans_df = pd.concat([trans_df, pd.get_dummies(trans_df.select_dtypes(include=['category']))], axis=1)


# =============================================================================
# Update Features_selection function

def transform_features(df):
    # remove cols with no. of missing value > 5% of that column    
    missing_values = df.isnull().sum()
    remove_all = missing_values[missing_values > (len(df)*0.05)]
    df = df.drop(remove_all.index, axis =1)

    # remove any text col with no. of missing value >1 (not necessary)
#    df_txt = df.select_dtypes(include=['object'])
#    missing_txt = df_txt.isnull().sum()
#    remove_txt = missing_txt[missing_txt>0]
#    df = df.drop(remove_txt.index, axis = 1)

    # fill missing value
    missing_count = df.isnull().sum().sort_values(ascending=False)
    value_missing = missing_count[missing_count>0]
    values_to_fill = df[value_missing.index].mode().to_dict(orient='record')[0]
    df = df.fillna(values_to_fill)

    # create new features
    years_owned = df['Yr Sold'] - df['Year Built']
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    
    # remove outliers where years_owned and years_since_remod <0
    df = df.drop([1702, 2180, 2181], axis=0)
    # add new features
    df['Years Before Sale'] = years_owned
    df['Years Since Remod'] = years_since_remod
    # remove old features
    df = df.drop(['Yr Sold', 'Year Built', 'Year Remod/Add'], axis = 1)

    # remove useless cols                
    df = df.drop(['Order', 'PID'], axis = 1)
    # remove cols have data leakage problem
    df = df.drop(['Mo Sold', 'Sale Type', 'Sale Condition'], axis =1)

    return df


def features_selection (df, coeff_cutoff = 0.4, unique_cutoff = 10):
    # numerical features
    num_df = df.select_dtypes(include=['int', 'float'])
    corr_coeff = num_df.corr()['SalePrice'].abs().sort_values(ascending=False)
    remove_features1 = corr_coeff[corr_coeff<coeff_cutoff]
    df = df.drop(remove_features1.index, axis = 1)

    # nominal features

    norminal_cols = ["MS SubClass", "MS Zoning", "Street", "Alley", 
                    "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", 
                    "House Style", "Roof Style", "Roof Matl", 
                    "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", 
                    "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]

    # filter out non-norminal cols
    trans_cat_cols = []
    for col in norminal_cols:
        if col in df.columns:
            trans_cat_cols.append(col)

    col_unique = df[trans_cat_cols].apply(lambda c: len(c.value_counts()))

    # filiter out norminal features with more than 10 categories
    remove_features2 = col_unique[col_unique>unique_cutoff]
    df = df.drop(remove_features2.index, axis =1)

    # convert norminal to categorical 
    text_col = df.select_dtypes(include=['object'])
    for col in text_col:
        df[col] = df[col].astype('category')
    df = pd.concat([df, pd.get_dummies(df.select_dtypes(include=['category']))], axis=1)
    
    return df

def train_and_test (df):
    train = df.iloc[:1460]
    test = df.iloc[1460:]
    
    num_train = train.select_dtypes(include=['int', 'float'])
    
    features = num_train.columns.drop('SalePrice')
    target = 'SalePrice'
    
    lr = LinearRegression()
    lr.fit(train[features], train[target])
    predictions = lr.predict(test[features])
    
    mse = mean_squared_error(test[target], predictions)
    rmse = np.sqrt(mse)
    return rmse

df = pd.read_csv('AmesHousing.tsv', sep = '\t')
trans_df = transform_features(df)
filtered_df = features_selection(trans_df)
rmse = train_and_test(filtered_df)

print(rmse)

# =============================================================================
# 3. TRAIN AND TEST THE MODEL
# =============================================================================
# KFold LinearRegression

numeric_df = filtered_df.select_dtypes(include=['int', 'float'])
features = numeric_df.columns.drop('SalePrice')

# when the dataset is divided by more than 2 groups,  
# each of them consist of train/test sets

lr = LinearRegression()
kf = KFold(n_splits = 4, shuffle = True)
rmse_values = []
for train_index, test_index, in kf.split(filtered_df):
    train = filtered_df.iloc[train_index]
    test = filtered_df.iloc[test_index]
    lr.fit(train[features], train['SalePrice'])
    predictions = lr.predict(test[features])
    mse = mean_squared_error(test['SalePrice'], predictions)
    rmse = np.sqrt(mse)
    rmse_values.append(rmse)
    
avg_rmse = np.mean(rmse_values)
print(rmse_values)
print(avg_rmse)

# =============================================================================
# Update train_and_test function

def transform_features(df):
    missing_all = df.isnull().sum()
    drop_missing1 = missing_all[missing_all > len(df)*0.05]
    df = df.drop(drop_missing1.index, axis=1)
    
    missing_count2 = df.isnull().sum()
    fixable_cols = missing_count2[missing_count2>0]
    values_to_fill = df[fixable_cols.index].mode().to_dict(orient='record')[0]
    df = df.fillna(values_to_fill)
    
    years_owned = df['Yr Sold'] - df['Year Built']
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    df = df.drop([1702, 2180, 2181], axis=0)
    df['Years Before Sale'] = years_owned
    df['Years Since Remod'] = years_since_remod
    
    df = df.drop(['PID', 'Order', 'Mo Sold', 'Sale Condition', 'Sale Type', 
                  'Year Built', 'Year Remod/Add'], axis = 1)
    return df


def features_selection(df, coeff_cutoff = 0.4, unique_cutoff = 10):
    num_df = df.select_dtypes(include=['int', 'float'])
    corr_coeff = num_df.corr()['SalePrice'].abs().sort_values()
    df = df.drop(corr_coeff[corr_coeff < coeff_cutoff].index, axis = 1)
    
    norminal_cols = ["MS SubClass", "MS Zoning", "Street", "Alley", 
                    "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", 
                    "House Style", "Roof Style", "Roof Matl", 
                    "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", 
                    "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]

    transform_cat_cols = []
    for col in norminal_cols:
        if col in df.columns:
            transform_cat_cols.append(col)
            
    
    uniq_counts = df[transform_cat_cols].apply(lambda c: len(c.value_counts()))       
    df = df.drop(uniq_counts[uniq_counts > unique_cutoff].index, axis =1)
    
    text_cols = df.select_dtypes(include=['object'])
    for col in text_cols:
        df[col] = df[col].astype('category')
    df = pd.concat([df, pd.get_dummies(df.select_dtypes(include=['category']))], axis = 1)
    
    return df


def train_and_test(df, k=0):
    num_df = df.select_dtypes(include=['int', 'float'])
    features = num_df.columns.drop('SalePrice')
    target = 'SalePrice'
    lr = LinearRegression()
    
    if k == 0:
        train = df.iloc[:1460]
        test = df.iloc[1460:]
        lr.fit(train[features], train[target])
        predictions = lr.predict(test[features])
        mse = mean_squared_error(test[target], predictions)
        rmse = np.sqrt(mse)
        return rmse
    
    if k==1:
        shuffled_df = df.sample(frac=1, )
        train = shuffled_df.iloc[:1460]
        test = shuffled_df.iloc[1460:]
        
        lr.fit(train[features], train[target])
        prediction1 = lr.predict(test[features])
        mse1 = mean_squared_error(test[target], prediction1)
        rmse1 = np.sqrt(mse1)
        
        lr.fit(test[features], test[target])
        prediction2 = lr.predict(train[features])
        mse2 = mean_squared_error(train[target], prediction2)
        rmse2 = np.sqrt(mse2)
        
        avg_rmse = np.mean([rmse1, rmse2])
        print(rmse1, rmse2)
        return avg_rmse
    
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values =[]
        for train_index, test_index in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            
            lr.fit(train[features], train[target])
            predictions = lr.predict(test[features])
            mse = mean_squared_error(test[target], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
            
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        return avg_rmse
    
df = pd.read_csv('AmesHousing.tsv', sep='\t')
trans_df = transform_features(df)
filtered_df = features_selection(trans_df)
avg_rmse = train_and_test(filtered_df, 4)
print(avg_rmse)

        




