# accept list, dataframe and numpy array
import pandas as pd
import numpy as np
from scipy import stats

# One-way ANOVA
def anova1w(data):
    # check type of data structure
    if isinstance(data,pd.DataFrame):
        data = data.values.T
    elif isinstance(data,list):
        data = np.array(data).T
    elif isinstance(data,np.ndarray):
        data = data.T
    else:
        return("Invalid data format")
    rows = data.shape[0]
    columns = data.shape[1]
    totobs = rows* columns
    # sum of squares
    SST = np.sum(data**2)-np.sum(data)**2 / totobs
    SSC = np.sum(np.sum(data,axis=0)**2)/rows-np.sum(data)**2/totobs
    SSE = SST-SSC
    # degree of freedom
    ssc_df = columns-1
    sse_df = columns*(rows-1)
    # mean square
    ms_ssc = SSC/ssc_df
    ms_sse = SSE/sse_df
    # f statistic
    f = ms_ssc/ms_sse
    p = 1-stats.f.cdf(f,ssc_df,sse_df)
    # prepare anova table
    result ={'Source':['Column means','Residual'],
             'Sum_Sq':[SSC,SSE],'Dof':[ssc_df,sse_df],
             'Mean_Sq':[ms_ssc,ms_sse],'F':[f,' '],'P_value':[p,' ']}
    anova_tab = pd.DataFrame(result)
    return(anova_tab)

# Two-way ANOVA
def anova2w(data):
    # check type of data structure
    if isinstance(data,pd.DataFrame):
        data = data.values.T
    elif isinstance(data,list):
        data = np.array(data).T
    elif isinstance(data,np.ndarray):
        data = data.T
    rows = data.shape[0]
    columns = data.shape[1]
    totobs = rows* columns
    # Sum of squares
    SST = np.sum(data**2)-np.sum(data)**2 / totobs
    SSR = np.sum(np.sum(data,axis=1)**2)/columns-np.sum(data)**2/totobs
    SSC = np.sum(np.sum(data,axis=0)**2)/rows-np.sum(data)**2/totobs
    SSE = SST-SSR-SSC
    # degree of freedom
    row_df = rows-1
    col_df = columns-1
    sse_df = row_df*col_df
    # mean square
    ms_ssr = SSR/row_df
    ms_ssc = SSC/col_df
    ms_sse = SSE/sse_df
    # f statistic
    f1 = ms_ssr/ms_sse
    f2 = ms_ssc/ms_sse
    # p-values
    p1 = 1-stats.f.cdf(f1,row_df,sse_df)
    p2 = 1-stats.f.cdf(f2,col_df,sse_df)
    # prepare anova table
    result ={'Source':['Row means','Column means','Residual'],
             'Sum_Sq':['SSR','SSC','SSE'],'Dof':[row_df,col_df,sse_df],
             'Mean_Sq':[ms_ssr,ms_ssc,ms_sse],'F':[f1,f2,' '],'P_value':[p1,p2,' ']}
    anova_tab = pd.DataFrame(result)
    return(anova_tab)
