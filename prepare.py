import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def nulls_by_row(df):
    '''
    akes in a dataframe and returns a dataframe with 3 columns: 
    the number of columns missing, percent of columns missing, and number of rows with n columns missing
    '''
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing




def miss_dup_values(df):
    '''
    takes in a dataframe of observations and attributes and returns a dataframe where each row is an atttribute name, 
    the first column is the number of rows with missing values for that attribute, 
    and the second column is percent of total rows that have missing values for that attribute and 
    duplicated rows.
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    '''
    takes in  a df , aproportion of columns and rows that we want to keep and 
    drop rows or columns based on the percent of values that are missing:
    '''
    threshold = int(round(prop_required_columns * len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns),0))
    df = df.dropna(axis=0, thresh=threshold)
    
     
    return df


def drop_low_missing_values(df, per = 1 ):
    '''
    takes in a df and the percentage that you want to drop of rows. the defautl value is 1%
     remove the rows that has columuns with missing values less than 1%**
    '''
    
    #drop rows with null values < per %
    lis =((100 * df.isnull().sum() / len(df))> 0) &  ((100 * df.isnull().sum() / len(df))< per)
    col_drop = list(lis[lis == True].index)
    df = df.dropna(axis=0, subset = col_drop)
    
    return df