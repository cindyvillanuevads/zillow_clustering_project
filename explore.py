import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans

def plot_variable_pairs(df, target):
    '''
    Takes in a dataframe and a target and returns  plots of all the pairwise relationships 
    along with the regression line for each pair.
    '''
    
    # get the list of the columns  that are not object type
    columns = list(df.drop(columns = 'logerror_bins').select_dtypes(exclude= 'O').columns)
    #remove target from columns
    columns.remove(target)
    
    #plot
    for col in columns:
        sns.regplot(x= col, y= target, data=df,line_kws={'color': 'green'})
        plt.title(col)
        plt.show()
    return





def heatmap (df):
    '''
    Takes in a df and return a heatmap
    '''
    x = len(list(df.select_dtypes(exclude='O').columns))
    df_corr= df.corr()
    sns.heatmap(df_corr, cmap='Purples', annot=True, linewidth=0.5, mask= np.triu(df_corr))
    plt.ylim(0, x)
    return


def create_cluster(df,validate, test, X, k, name):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = StandardScaler(copy=True).fit(df[X])
    X_scaled = pd.DataFrame(scaler.transform(df[X]), columns=df[X].columns.values).set_index([df[X].index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df[name] = kmeans.predict(X_scaled)
    df[name] = 'cluster_' + df[name].astype(str)
    
    v_scaled = pd.DataFrame(scaler.transform(validate[X]), columns=validate[X].columns.values).set_index([validate[X].index.values])
    validate[name] = kmeans.predict(v_scaled)
    validate[name] = 'cluster_' + validate[name].astype(str)
    
    t_scaled = pd.DataFrame(scaler.transform(test[X]), columns=test[X].columns.values).set_index([test[X].index.values])
    test[name] = kmeans.predict(t_scaled)
    test[name] = 'cluster_' + test[name].astype(str)
    
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids

def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler, name):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = name)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
    plt.legend(bbox_to_anchor=(1.2,.8))



def scatter_plot_ks (X_df, X_scaled_df, x, y, start, finish):
    fig, axs = plt.subplots(2, 2, figsize=(13, 13), sharex=True, sharey=True)

    for ax, k in zip(axs.ravel(), range(start , finish)):
        clusters = KMeans(k).fit(X_scaled_df).predict(X_scaled_df)
        ax.scatter(X_df[x], X_df[y], c=clusters)
        ax.set(title='k = {}'.format(k), xlabel=x, ylabel=y)



def elbow_chart ( X, end_range):
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns= X.columns).set_index([X.index.values])

    
    # let is explore what values of k might be appropriate
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X_scaled).inertia_ for k in range(2, end_range)}).plot(marker='x')
        plt.xticks(range(2, end_range))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')


def stat_ttest (df,list_cluster, n_cluster):
    alpha=0.05
    for n in range (n_cluster):
        t,p = stats.ttest_1samp(list_cluster[n], df.logerror.mean())
        if (p < alpha):
            print(f'For cluster_{n}, We reject the null hypothesis')
           
        else:
            print(f'For cluster_{n}, We fail to reject the null hypothesis')
        print(f't = {t},    p= {p}')
        
