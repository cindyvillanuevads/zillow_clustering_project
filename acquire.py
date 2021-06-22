import pandas as pd
import numpy as np
import os


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


sql_query ='''
SELECT prop.parcelid,  prop.basementsqft, bathroomcnt, bedroomcnt, decktypeid, calculatedfinishedsquarefeet,
fips, fireplacecnt, garagecarcnt, hashottuborspa, latitude, longitude, lotsizesquarefeet, poolcnt,
yearbuilt, numberofstories, prop.airconditioningtypeid, airconditioningdesc, prop.architecturalstyletypeid,
architecturalstyledesc,
prop.storytypeid, storydesc, prop.propertylandusetypeid, propertylandusedesc, 
prop.typeconstructiontypeid, regionidcity, regionidcounty, regionidzip, typeconstructiondesc, unitcnt,
structuretaxvaluedollarcnt, landtaxvaluedollarcnt, taxvaluedollarcnt, taxamount, logerror, transactiondate
from properties_2017 as prop
right join predictions_2017 as pred USING (parcelid)
LEFT JOIN airconditioningtype USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
LEFT JOIN propertylandusetype USING (propertylandusetypeid)
LEFT JOIN storytype USING(storytypeid)
LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
WHERE transactiondate like '2017%'  
AND latitude != 'NULL' AND longitude != 'NULL';

'''

# *************************************  connection url **********************************************

# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, username, password
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'

# ************************************ generic acquire function ***************************************************************

def get_data_from_sql(db_name, query):
    """
    This function takes in a string for the name of the database that I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    db_name : df name in a string type
    query: aalready created query that was named as query 
    Example:
    query = '''
    SELECT * 
    FROM table_name;
    '''
    df = get_data_from_sql('zillow', query)
    """
    df = pd.read_sql(query, get_connection(db_name))
    return df




def get_zillow():
    '''
    This function reads in telco_churn data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_data_from_sql('zillow', sql_query)
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df





# ************************************ unique values ***************************************************************

def report_unique_val (df):
    '''
    takes in a df and gives you a report of number of unique values and count values <15 (categorical)
    count values <15 (numerical)
    '''
    num_cols = df.select_dtypes(exclude = 'O').columns.to_list()
    cat_cols = df.select_dtypes(include = 'O').columns.to_list()
    for col in df.columns:
            print(f'**{col}**')
            le = df[col].nunique()
            print ('Unique Values : ', df[col].nunique())
            print(' ')
            if col in cat_cols and le < 15:
                print(df[col].value_counts())
            if col in num_cols and  le < 23:
                 print(df[col].value_counts().sort_index(ascending=True)) 
            elif col in num_cols and le <150:
                print(df[col].value_counts(bins=10, sort=False).sort_index(ascending=True))
            elif col in num_cols and le <1001:
                print(df[col].value_counts(bins=100, sort=False).sort_index(ascending=True))

            print('=====================================================')


# ************************************ sumarize ***************************************************************
def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # shape
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================')
    print('Dataframe shape: ')
    print(df.shape)
    print('=====================================================')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================')
    print('Dataframe Description: ')
    print(df.describe().T)
    print('=====================================================')
    report_unique_val (df)


