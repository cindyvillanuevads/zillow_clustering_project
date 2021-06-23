import pandas as pd
import numpy as np
import os
import acquire as a
import prepare as p


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")





def wrangle_zillow (  prop_required_columns=0.75, prop_required_row=0.75):
    '''
    Acquire zillow db from sql, single unit properties with transactions 2017,
    remove outliers, missing values,  split into train validate and test.
    impute heatingorsystemtypeid with most_frequent value
    return tran, validate test
    '''
    
    #acquire data
    df= a.get_zillow()
    
    #getting the latest transactions 
    df1 = df.sort_values(by ='transactiondate', ascending=True).drop_duplicates( subset = 'parcelid' ,keep= 'last')
    
    #this list has all types of single unit properties
    single= ['Single Family Residential',' Mobile Home' , 'Townhouse ', 'Manufactured, Modular, Prefabricated Homes'  ]
    #create a mask
    single_mask = df1['propertylandusedesc'].isin(single)
    #using that mask and also add  a condition
    df_single = df1[single_mask & ((df1['unitcnt'] == 1) | (df1['unitcnt'].isnull()))]

    #missing values
    df_single = p.handle_missing_values(df_single, prop_required_columns, prop_required_row)
    
    #missing low values
    df_single = p.drop_low_missing_values(df_single, per= 3)


    #get counties
    df_single = p.get_counties(df_single)

    #create features
    df_single = p.create_features (df_single)
    print('before outliers', df_single.shape)

    #remove outliers 
    col_list = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
             'regionidzip', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxrate', 'lotsize_acres', 'age']
    df_single = p.remove_outliers(df_single, col_list, k=1.5)


    #drop duplicated rows
    df_single= df_single.drop(columns = 'propertylandusedesc')
    print('df shape -->', df_single.shape)
    #split data
    train, validate, test = p.split_data(df_single)


    return train, validate, test