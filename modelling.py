"""
MODELLING.py

AUTHOR: CHETAN CHAUHAN
VERSION: 01

This module consists of all the functions which are used for performing basic data exploration, variable importance 
and for variable transformation to be finally used for modelling

"""

import os
import math  #For Checking Float
import random
import warnings
import pandas as pd
import numpy as np
import collections 

print "Executed Some Steps for Pandas Options"

pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',1000)
pd.set_option('max_colwidth', -1)

warnings.filterwarnings('ignore')

print "Running on Python 2"

warnings.filterwarnings('ignore')


def edd_full(df):
    """Runs Full EDD for all Variables in DF"""
    
    edd_full = pd.DataFrame()
    print "Total Data Size : ",len(df)," Rows & ",len(df.columns)," Columns"
    
    for var in df.columns:
        print "Running for :",var
        edd_full = pd.concat([edd_full,edd(df,var)],axis=0)
        
    edd_full.reset_index(drop=True,inplace=True) ##RETURNS ORDERED VALUES
    return(edd_full)

def edd(df,var):
    """Function to generate a comprehensive EDD for any given variable"""
    
    """Basic Data Indicators"""
    var_data_type   = str(df[var].dtypes)
    min_var_length  = df[var].apply(lambda x: 0 if x!=x else len(str(x))).min()
    max_var_length  = df[var].apply(lambda x: 0 if x!=x else len(str(x))).max()
    data_length     = len(df)
    var_count       = df[var].count()
    missing_count   = data_length-var_count
    distinct_values = df[var].dropna().nunique()  #Should now be able to handle missing values
    missing_rate    = np.round(float(missing_count)/float(data_length),2)
    
    """Modal Value %age"""
    modal_value = df[var].value_counts().index[0]
    modal_freq  = df[var].value_counts().head(1).tolist()[0]
    modal_pct   = np.round(float(modal_freq)/float(data_length),2)
    
    """Freq Estimator"""
    """Calculating Basic Statistics for each variable (basis its variable type)"""
    if var_data_type in ('object'):
        """Check"""
        min_value    = '--'
        max_value    = '--'
        mean_value   = '--'
        median_p50   = '--'
        """Object Module"""
        
        vc = df[var].value_counts()
        
        if distinct_values == 1:
            top2_p05=top3_p25=bot3_p75=bot2_p90='--'
            bot1_p99=top1_p01=str(vc.index[0])+'|'+str(vc.tolist()[0])+'|'+str(np.round(float(vc.tolist()[0])/float(var_count)*100,2))+'%' 
            
        elif distinct_values == 2:            
            top3_p25=bot3_p75='--'
            bot2_p90=top1_p01=str(vc.index[0])+'|'+str(vc.tolist()[0])+'|'+str(np.round(float(vc.tolist()[0]) /float(var_count)*100,2))+'%'
            bot1_p99=top2_p05=str(vc.index[1])+'|'+str(vc.tolist()[1])+'|'+str(np.round(float(vc.tolist()[1]) /float(var_count)*100,2))+'%' 
        else:
            top1_p01     = str(vc.index[0])+'|'+str(vc.tolist()[0]) +'|'+str(np.round(float(vc.tolist()[0]) /float(var_count)*100,2))+'%'
            top2_p05     = str(vc.index[1])+'|'+str(vc.tolist()[1]) +'|'+str(np.round(float(vc.tolist()[1])
/float(var_count)*100,2))+'%'
            top3_p25     = str(vc.index[2])+'|'+str(vc.tolist()[2]) +'|'+str(np.round(float(vc.tolist()[2]) /float(var_count)*100,2))+'%'
            bot3_p75     = str(vc.index[-3])+'|'+str(vc.tolist()[-3])+'|'+str(np.round(float(vc.tolist()[-3]) /float(var_count)*100,2))+'%'  
            bot2_p90     = str(vc.index[-2])+'|'+str(vc.tolist()[-2])+'|'+str(np.round(float(vc.tolist()[-2]) /float(var_count)*100,2))+'%'
            bot1_p99     = str(vc.index[-1])+'|'+str(vc.tolist()[-1])+'|'+str(np.round(float(vc.tolist()[-1]) /float(var_count)*100,2))+'%'
    else:
        min_value       = df[var].min()
        max_value       = df[var].max()
        mean_value      = df[var].mean()
        median_p50      = df[var].median()
        top1_p01        = df[var].quantile(0.01)
        top2_p05        = df[var].quantile(0.05)
        top3_p25        = df[var].quantile(0.25)
        bot3_p75        = df[var].quantile(0.75)
        bot2_p90        = df[var].quantile(0.90)
        bot1_p99        = df[var].quantile(0.99)
    
    
    frame = collections.OrderedDict()

    """Combining Everything Together within Ordered Dictionary"""
    
    frame['variable']     = var
    frame['data_type']    = var_data_type 
    frame['min_len']      = min_var_length
    frame['max_len']      = max_var_length
    frame['total']        = data_length
    frame['count']        = var_count
    frame['missing']      = missing_count
    frame['missing_rate'] = missing_rate
    frame['distinct']     = distinct_values
    frame['mode']         = modal_value
    frame['modal_freq']   = modal_freq
    frame['modal_pct']    = modal_pct
    frame['min']          = min_value
    frame['max']          = max_value
    frame['mean']         = max_value
    frame['median']       = median_p50
    frame['top1_p01']     = top1_p01
    frame['top2_p05']     = top2_p05
    frame['top3_p25']     = top3_p25
    frame['bot3_p75']     = bot3_p75
    frame['bot2_p90']     = bot2_p90
    frame['bot1_p99']     = bot1_p99
    
    single_frame = pd.DataFrame(frame,index=[0]) ##USING ORDERED DICTIONARY ALLOWS ORDERED COLUMNS IN EDD
    
    return(single_frame)


##FUNCTION TO HANDLE NULL VALUE
def null_handler(df,var,dv):
    if len(df[df[var].isnull()])==0:
        return(pd.DataFrame())
    else:
        null_dict = {}
        null_dict['events']     = df[df[var].isnull()][dv].sum()
        null_dict['nobs']       = df[df[var].isnull()][dv].count()
        null_dict['non_events'] = null_dict['nobs'] - null_dict['events']
        null_dict['variable']   = var
        null_dict['category']   = 'NULL'
        null_df = pd.DataFrame(null_dict,index=np.arange(1))
        null_df = null_df[['variable','category','nobs','events','non_events']]
        return(null_df)

##GROUPBY IMPLEMENTATION FUNCTION
def group_by_var(df,var,dv):
    """
    Returns Dataframe with Events grouped by input Variable
       df  : Input Dataframe
       var : Variable to be Grouped
       dv  : Dependant Variable
    """
    ##TO CREATE NEW CATEGORY NAMES FOR BINNED VARIABLES
    if 'qcut' in var.split('_'):
        mod_var = var.split('qcut_')[1]          ##to handle variable names with underscore
        test    = df[~df[mod_var].isnull()].groupby([var]).agg({dv:['sum','count'],
                                                           mod_var:['min','max']}).reset_index()
        test.columns=test.columns.droplevel()
        test.rename(columns={ '':'category',
                             'sum':'events',
                             'count':'nobs'},inplace=True)
        test['variable'] = mod_var
        test['non_events'] = test['nobs']-test['events']
        test['final_category'] = test.apply(lambda x: str(x['category'])+'_['+
                                                      str(np.round(x['min'],2))+'-'+
                                                      str(np.round(x['max'],2))+']', axis=1)
        test.drop(['category'],axis=1,inplace=True)
        test.rename(columns={'final_category':'category'},inplace=True)
        test.sort_values(by=['category'],ascending=True,inplace=True)
        test  = test[['variable','category','nobs','events','non_events']]
        check = pd.concat([null_handler(df,mod_var,dv),test],axis=0)
        check.reset_index(drop=True,inplace=True)
        return(check)
    
    ##CATEGORY NAMES FOR EVERYTHING ELSE
    else:
        test = df[~df[var].isnull()].groupby([var]).agg({dv:['sum','count']}).reset_index()
        test.columns = ['category','events','nobs']
        test['variable'] = var
        test['non_events'] = test['nobs']-test['events']
        test = test[['variable','category','nobs','events','non_events']]
        check = pd.concat([null_handler(df,var,dv),test],axis=0)
        check.reset_index(drop=True,inplace=True)
        return(check)
    
##PRODUCES BIVARIATE FOR VARIABLES  
def bivariate(main,var,dv):
    """
    Returns Bivariate Dataframe with Events grouped by input Variable
       df  : Input Dataframe
       var : Variable to be Grouped
       dv  : Dependant Variable
    """
    df = main[[var,dv]].copy()  ##SMALLER COPIES MORE OPTIMAL
    
    ##CHECKING FOR CORRECT DATA TYPES
    if str(df[var].dtype) not in ['object','int16','int32','int64','float16','float32','float64']:
        return(pd.DataFrame()) #RETURNS EMPTY DATAFRAME FOR FULL RUN
    else:
        ##CATEGORICAL TREATMENT
        if (str(df[var].dtype)=='object') or (df[var].nunique()<=15) :
            return(group_by_var(df,var,dv))
        
        ## RANDOM RANKING CONTINUOUS TREATMENT
        else:
            try:
                df['qcut_'+var]=pd.qcut(df[var],10,labels=['BIN'+str(i) for i in range(10)])
                return(group_by_var(df,'qcut_'+var,dv))
            except:
                ##RANDOM RANKING
                check = []
                for x in np.arange(len(df)):
                    check.append(random.random())
                null_col = pd.DataFrame(check,columns=['rand'])
                df = pd.concat([df,null_col],axis=1)
                df.sort_values(by='rand',ascending=True,inplace=True)
                df.reset_index(drop=True,inplace=True)
                df['qcut_'+var] = pd.qcut(df[var].rank(method='first'),10,labels=['BIN'+str(i) for i in range(10)])
                return(group_by_var(df,'qcut_'+var,dv))
            
##GIVES INFORMATION VALUE FOR SINGLE VARIABLE
def information_value(var_df):
    """Returns Information Value of the Variable whose binning has been performed
       var_df  : Input Dataframe of Variable
    """
    
    ##ADJUSTING FOR 0 EVENTS OR NON EVENTS
    var_df['adj_events']     = var_df.apply(lambda x: x['events']+0.5 if (x['events']==0 or x['non_events']==0) else x['events'],axis=1)
    var_df['adj_non_events'] = var_df.apply(lambda x: x['non_events']+0.5 if (x['events']==0 or x['non_events']==0) else x['non_events'],axis=1)
    
    ##USUAL IV CALCULATIONs
    var_df['pct_events']     = var_df['adj_events']    /var_df['adj_events'].sum()
    var_df['pct_non_events'] = var_df['adj_non_events']/var_df['adj_non_events'].sum()
    var_df['log_g_b']        = np.log(var_df['pct_events']/var_df['pct_non_events'])
    var_df['g_b']            = var_df['pct_events']-var_df['pct_non_events']
    var_df['woe']            = var_df['g_b']*var_df['log_g_b']
    return(var_df['woe'].sum())

def full_iv(df,dv):
    """
    Runs full IV on the DataFrame using Dependant Variable Provided
    """
    for var in df.columns:
        if var==dv:
            continue
        check['var'].append(var)
        check['iv'].append(information_value(bivariate(df,var,dv)))
    final = pd.DataFrame(check,index=np.arange(len(main.columns)-1))
    final.sort_values(by=['iv'],ascending=False,inplace=True)
    final.reset_index(inplace=True,drop=True)
    return(final)
