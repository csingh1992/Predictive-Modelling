
"""

This module consists of all the functions which are used for performing basic data exploration, variable importance 
and for variable transformation to be finally used for modelling

"""

import pandas as pd
import numpy as np


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

def group_by_var(df,var,dv):
    """
    Returns Dataframe with Events grouped by input Variable
       df  : Input Dataframe
       var : Variable to be Grouped
       dv  : Dependant Variable
    """
    ##TO CREATE NEW CATEGORY NAMES FOR BINNED VARIABLES
    if 'qcut' in var.split('_'):
        mod_var = var.split('_')[1]
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
    
    
def bivariate(main,var,dv):
    """
    Returns Bivariate Dataframe with Events grouped by input Variable
       df  : Input Dataframe
       var : Variable to be Grouped
       dv  : Dependant Variable
    """
    df = main.copy()
    ##CHECKING FOR CORRECT DATA TYPES
    if str(df[var].dtype) not in ['object' ,'int32','int64','int16','float32','float16','float64']:
        return(0)
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
                df.sort_values(by='rand',ascending=True).reset_index(drop=True,inplace=True)
                df['qcut_'+var] = pd.qcut(df[var].rank(method='first'),10,labels=['BIN'+str(i) for i in range(10)])
                return(group_by_var(df,'qcut_'+var,dv))

def information_value(var_df):
    """Returns Information Value of the Variable whose binning has been performed
       var_df  : Input Dataframe of Variable
    """
    var_df['pct_events']     = var_df['events']/var_df['events'].sum()
    var_df['pct_non_events'] = var_df['non_events']/var_df['non_events'].sum()
    var_df['log_g_b']        = np.log(var_df['pct_events']/var_df['pct_non_events'])
    var_df['g_b']            = var_df['pct_events']-var_df['pct_non_events']
    var_df['woe']            = var_df['g_b']*var_df['log_g_b']
    return(var_df['woe'].sum())