"""
MODELLING.py

AUTHOR: CHETAN CHAUHAN
VERSION: 01

This module consists of all the functions which are used for performing basic data exploration, variable importance 
and for variable transformation to be finally used for modelling

Check for the classification item


"""
##**************************************************************************************************************##
##  Add Detailed Instructions on how to use the module 
##  MODULES TO BE ADDED
##  00: EDD Variable Selection
##  01: Model Evaluation Metrics and Graphical Results 
##  ----------------------------------------------------------------------->>>>>02: Bivariate Informative Plotting
##  03: Gini Index Variable Importance Metric----->>>> (3)
##  04: Varclus, PCA, Correlation, VIF (Read Relevant Theory First)
##  05: Monotonic WOE Interpreter for Logistic Regression
##  06: Linear Regression Module (Model Evaluation Metrics)----->>>>> (1)
##  07: Single Variable EDD Plotting (Using Seaborn)---->>>>> (2)
##  08: Variable Transformation Impact Assessment (Theory + Output Testing)
##  09: Adding Times (Some sort of Wrapper and Decorator Functions for Enhanced Utility)
##*************************************************************************************************************##

from __future__ import print_function

import os
import math  
import random
import time
import warnings
import pandas as pd
import numpy as np
import collections

from ipywidgets import IntProgress
from IPython.display import display,HTML

import matplotlib.pyplot as plt
from matplotlib import gridspec #RATIO DIVISION

pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',1000)
pd.set_option('max_colwidth', -1)
pd.options.display.float_format = '{:,.2f}'.format  ##IMPROVING DISPLAY OPTIONS

warnings.filterwarnings('ignore')

print("Running on Python 2")
print("Pandas Set Options Modified")

#get_ipython().run_line_magic("config InlineBackend.figure_format='retina'",'') ##TO ENABLE RETINA DSIPLAY

##*******************************************UTILITY: PROGRESS BAR************************************##
def progress_bar(**kwargs):
    """
    Function is designed with two KWARGS:
    
    First Run --> Initialize Floater
    NULL=<num_vars> : This would be the initialization of the Floater Object
    
    Second Run --> Recursively Pass Floater to generate value
    ORIGIN=k            : This would pass an already created floater   
    
    Function Guide: 
    check=progress_bar(NULL=(len(run_list)-len(skip_list)))
    check=progress_bar(ORIGIN=check)
    """
    key = kwargs.items()
    
    if key[0][0]=='NULL':
        f = IntProgress(min=0,max=key[0][1]) 
        f.value=0
        #f.description='Completed '+str(f.value)+'%'
        display(f)
        return(f)
    else:
        f=key[0][1] #FLOATER OBJECT
        f.value+=1  #INCREMENT VALUE BY ONE
        f.description=str(int(float(f.value)/float(f.max)*100))+'% Done'
        return(f)

def full_data_run(**kwargs):
    """
    a) run_list : List of columns on which function needs to be mapped
    b) skip_list: List of columns on which function needs to be skipped
    c) dframe   : Name of DataFrame
    d) func     : Name of the function to be scaled
    e) dv       : Dependant Variable
    
    Usage Example: 
    full_data_run(run_list  = main.columns,
                  skip_list = ['BAD'],
                  dframe    = main,
                  func      = bivariate,
                  dv        = 'BAD')
    """
    
    ##PROPER WAY OF USING KWARGS
    run_list  = kwargs['run_list']
    skip_list = kwargs['skip_list']
    dframe    = kwargs['dframe']
    func      = kwargs['func']
    dv        = kwargs['dv']
    
    ##EDD COMPLETE STATUS
    if func==edd:
        print_html="""
        <p style="text-align:right"> Rows : ##1## || Columns : ##2## </p>
        """.replace('##1##',str(len(dframe))).replace('##2##',str(len(dframe.columns)))
        display(HTML(print_html))
    
    ##DECLARING EMPTY DATAFRAME
    full_run = pd.DataFrame()
    
    ##INITIALIZE PROGRESS BAR
    check=progress_bar(NULL=(len(run_list)-len(skip_list)))   
    
    ##LOOP FOR ALL FUNCTIONS ACROSS DATAFRAME
    for var in run_list:
        if var in skip_list:
            continue
        full_run = pd.concat([full_run,func(dframe,var,dv=dv)],axis=0)
        ##KWARGS NEEDS KEY VALUE PAIRS ELSE IT THROWS AN ERROR
        check=progress_bar(ORIGIN=check)
        
    full_run.reset_index(drop=True,inplace=True) ##RETURNS ORDERED VALUES
    return(full_run)    


##*******************************************EXPLORATORY DATA DICTIONARY************************************##
def edd(df,var,**kwargs):
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
        
        vc  = df[var].value_counts()
        vci = vc.index
        vcl = vc.tolist()
        
        if distinct_values == 1:
            top2_p05=top3_p25=bot3_p75=bot2_p90='--'
            bot1_p99=top1_p01=str(vci[0])+'|'+str(vcl[0])+'|'+str(np.round(float(vcl[0])/float(var_count)*100,2))+'%' 
            
        elif distinct_values == 2:            
            top3_p25=bot3_p75='--'
            bot2_p90=top1_p01=str(vci[0])+'|'+str(vcl[0])+'|'+str(np.round(float(vcl[0]) /float(var_count)*100,2))+'%'
            bot1_p99=top2_p05=str(vci[1])+'|'+str(vcl[1])+'|'+str(np.round(float(vcl[1]) /float(var_count)*100,2))+'%' 
        else:
            top1_p01     = str(vci[0] )+'|'+str(vcl[0] )+'|'+str(np.round(float(vcl[0] ) /float(var_count)*100,2))+'%'
            top2_p05     = str(vci[1] )+'|'+str(vcl[1] )+'|'+str(np.round(float(vcl[1] ) /float(var_count)*100,2))+'%'
            top3_p25     = str(vci[2] )+'|'+str(vcl[2] )+'|'+str(np.round(float(vcl[2] ) /float(var_count)*100,2))+'%'
            bot3_p75     = str(vci[-3])+'|'+str(vcl[-3])+'|'+str(np.round(float(vcl[-3]) /float(var_count)*100,2))+'%'  
            bot2_p90     = str(vci[-2])+'|'+str(vcl[-2])+'|'+str(np.round(float(vcl[-2]) /float(var_count)*100,2))+'%'
            bot1_p99     = str(vci[-1])+'|'+str(vcl[-1])+'|'+str(np.round(float(vcl[-1]) /float(var_count)*100,2))+'%'
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


##*******************************************GROUP BY VAR -- NULL ******************************************##
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

##*******************************************GROUP BY VAR*************************************************##    
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
    
##*******************************************BIVARIATE ANALYSIS******************************************##  
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
        if (str(df[var].dtype)=='object' and df[var].nunique()<30) or (df[var].nunique()<=15):
            return(group_by_var(df,var,dv))
        
        elif (str(df[var].dtype)=='object' and df[var].nunique()>=30):
            large_string_vars = {
                'variable'  :var,
                'category'  :'gt_30_categories',
                'nobs'      :df[var].count(),
                'events'    :df[dv].sum(),
                'non_events':df[var].count()-df[dv].sum()
            }
            ret_df = pd.DataFrame(large_string_vars,index=[0]) 
            return(ret_df[['variable','category','nobs','events','non_events']]) 
        
        ## RANDOM RANKING CONTINUOUS TREATMENT FOR NUMERIC VARIABLES
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
            
            
##*******************************************INFORMATION VALUE*********************************************##
def information_value(df,var,dv):
    """Returns Information Value of the Variable whose binning has been performed
       Input Data Frame should include df,var, and dv
    
    """
    
    var_df=bivariate(df,var,dv)
    
    
    if var_df['category'].loc[0]=='gt_30_categories':
        return(pd.DataFrame())
    
    def niv(a,b):
        if a==0 or b==0:
            return(1)
        else:
            return(0)
    
    
    ##ADJUSTING FOR 0 EVENTS OR NON EVENTS
    var_df['adj_events']     = var_df.apply(lambda x: x['events']+0.5 if niv(x['events'],x['non_events'])==1 else x['events'],axis=1)
    var_df['adj_non_events'] = var_df.apply(lambda x: x['non_events']+0.5 if niv(x['events'],x['non_events'])==1 else x['non_events'],axis=1)
    
    ##USUAL IV CALCULATIONs
    var_df['pct_events']     = var_df['adj_events']    /var_df['adj_events'].sum()
    var_df['pct_non_events'] = var_df['adj_non_events']/var_df['adj_non_events'].sum()
    var_df['log_g_b']        = np.log(var_df['pct_events']/var_df['pct_non_events'])
    var_df['g_b']            = var_df['pct_events']-var_df['pct_non_events']
    var_df['woe']            = var_df['g_b']*var_df['log_g_b']
    
    ##SHOULD RETURN SINGLE VALUE DATAFRAME
    check = {
        'var':[],
        'iv':[],
        'mod_iv':[],
        'diff':[]
    }
    check['var'].append(var_df['variable'].loc[0]) ##SINGLE VALUE LOC SELECTION
    check['iv'].append(var_df['woe'].sum())           ##COMPLETE IV
    check['mod_iv'].append(var_df[~var_df['category'].isin(['NULL'])]['woe'].sum()) ##NON NULL VALUE IV
    check['diff']= [x-y for x,y in zip(check['iv'],check['mod_iv'])]
    final = pd.DataFrame(check,index=[0])    
    return(final[['var','iv','mod_iv','diff']])


##*******************************************BIVARIATE PLOTTING*************************************************##
def bivariate_plot(**kwargs):
    """
    Usage Guide: How to use Bivariate Plot
    
    bivariate_plot( dframe =  main,
                    var    =  <Variable Name>
                    dv     =  <Dependant Variable>
                    sort   =  <'event_rate'    :Sorted by Event Rate, 
                                      'obs'    :Sorted by Number
                                      'default':Sort by default>)
    ## USAGE EXAMPLE
    bivariate_plot(dframe=main,
               var='JOB',
               dv='BAD',
               sort='event_rate')
    """
    
    main =kwargs['dframe']
    var  =kwargs['var']
    dv   =kwargs['dv']
    sort =kwargs['sort']   
    
    sort_var = 'dr' if sort=='event_rate' else 'nobs'
    
    check = bivariate(main,var,dv)
    
    ##CREATING ADDITIONAL VARIABLES
    check['dr']      = check['events']/check['nobs']*100
    check['bin_ind'] = check['category'].apply(lambda x: 1 if ('BIN' in str(x))&('[' in str(x)) else 0)
    binned =1 if check['bin_ind'].sum()>0 else 0 

    ##SORTING DATAFRAME
    if binned==1:
        check=check.sort_values(by=['category'],ascending=True).reset_index(drop=True)
    else:
        check=check.sort_values(by=[sort_var],ascending=False).reset_index(drop=True)

    ##CREATING X1 X2 LABELS
    check['x1_labels'] = check['category'].apply(lambda x: x.split('_')[0] if  binned==1  else x )
    check['x2_labels'] = check['category'].apply(lambda x: x.split('_')[1] if (binned==1 and 'NULL' not in x) else '')
    
    width = 0.6

    fig = plt.figure(figsize=(15,5)) #DEFINING FIGURE
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) # 1,2 FORMATION WITH 3:1 WIDTH RATIO
    ax1 = plt.subplot(gs[0]) #FIRST AXIS
    ax2 = ax1.twinx() #COMBO CHART DUAL AXIS
    ax3 = plt.subplot(gs[1]) #SECOND AXIS FOR TABLE


    check.plot(kind='bar' ,x='x1_labels',y='nobs',rot=0,color='lightblue',width=width,ax=ax1)
    check.plot(kind='line',x='x1_labels',y='dr'  ,rot=0,color='red',ax=ax2)

    x_range = list(np.arange(len(check)))
    y_range = check['dr'].tolist()
    for x,y in zip(x_range,y_range):
        label = "{:.1f}%".format(y)
        ax2.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,0), # distance from text to points (x,y)
                     ha='center',
                     verticalalignment='bottom',
                     size=12) # horizontal alignment can be left, right or center

    ax1.set_title("Bivariate For {}".format(check['variable'].loc[0]),fontsize=15)
    ax1.set_xlim([-width, len(check['variable'])-width*2/3])
    ax1.set_ylim([0 ,int(check['nobs'].max()*1.05)])
    ax2.set_ylim([0 ,int(check['dr'].max()*1.15)])
    ax1.xaxis.set_label_text("")
    ax1.tick_params(axis='x',labelsize=10)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    crow = check[['x1_labels','x2_labels','nobs']]    

    ax3.set_title("Group Split for {}".format(check['variable'].loc[0]),fontsize=15)
    table = ax3.table(
             cellText=crow.values, 
             colLabels=crow.columns, 
             loc='center',
             colWidths=[2,2,2]  ##THIS NEEDS TO BE CHANGED WHENEVER YOU ADD MORE COLUMNS FOR DISPLAY
            )
    ax3.set_xticks([])
    ax3.set_yticks([])
    table.auto_set_column_width((-1, 0, 1, 2, 3))
    table.set_fontsize(12)
    table.scale(1.7, 1.7)
    plt.show()