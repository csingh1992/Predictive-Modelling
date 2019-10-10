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
##  03: Gini Index Variable Importance Metric----->>>> (3)
##  04: Varclus, PCA, Correlation, VIF (Read Relevant Theory First)
##  05: Monotonic WOE Interpreter for Logistic Regression
##  06: Linear Regression Module (Model Evaluation Metrics)----->>>>> (1)
##  07: Single Variable EDD Plotting (Using Seaborn)---->>>>> (2)
##  08: Variable Transformation Impact Assessment (Theory + Output Testing)
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
import base64
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_recall_curve

from ipywidgets import IntProgress
from IPython.display import display,HTML

import matplotlib.pyplot as plt

pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',1000)
pd.set_option('max_colwidth', -1)
pd.options.display.float_format = '{:,.2f}'.format  ##IMPROVING DISPLAY OPTIONS

warnings.filterwarnings('ignore')

plt.ioff() ##STOPPING FROM INTERACTIVE PRINTING

print("Running on Python 2")
print("Pandas Set Options Modified")

#get_ipython().run_line_magic("config InlineBackend.figure_format='retina'",'') ##TO ENABLE RETINA DSIPLAY
#xes.axhline(self, y=0, xmin=0, xmax=1, **kwargs)[source]


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
    [(key,value)] = kwargs.items() ##CHANGE AS SUGGESTED BY VIBHU IMPROVES DICT ERROR OF KWARGS
    
    if key=='NULL':
        f = IntProgress(min=0,max=value) 
        f.value=0
        #f.description='Completed '+str(f.value)+'%'
        display(f)
        return(f)
    else:
        f=value #FLOATER OBJECT
        f.value+=1  #INCREMENT VALUE BY ONE
        f.description=str(int(float(f.value)/float(f.max)*100))+'% Done'
        return(f)

def full_data_run(**kwargs):
    """
    FUNCTION: This would return a Full Run of the Function on Run List - Skip List
    
    USAGE GUIDE: 
    full_data_run(
        *run_list  = <LIST OF VARIABLES ON WHICH FUNCTION NEEDS TO BE SCALED>
        *skip_list = <LIST OF VARIABLES ON WHICH FUNCTION NEEDS TO BE SKIPPED>
        *dframe    = <DATAFRAME>,
        *func      = <FUNCTION NAME WHICH HAS TO BE RUN>
        **kwargs
    
    EXAMPLE: 
    
    full_data_run(
              run_list = df.columns,
              skip_list= ['Survived','Name'],
              func     = bivariate_plot,
              dframe   = df,
              dv       = 'Survived',
              sort     = 'default')
    """
    
    ##PROPER WAY OF USING KWARGS
    run_list  = kwargs['run_list']
    skip_list = kwargs['skip_list']
    dframe    = kwargs['dframe']
    func      = kwargs['func']
    
    
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
        full_run = pd.concat([full_run,func(var=var,**kwargs)],axis=0)
        ##KWARGS NEEDS KEY VALUE PAIRS ELSE IT THROWS AN ERROR
        check=progress_bar(ORIGIN=check)
        
    full_run.reset_index(drop=True,inplace=True) ##RETURNS ORDERED VALUES
    return(full_run)    


##*******************************************EXPLORATORY DATA DICTIONARY************************************##
def edd(**kwargs):
    """
    FUNCTION: This would return a Single Row DataFrame with EDD of the Variable
    
    USAGE GUIDE: 
    edd(dframe= <DATAFRAME>,
        var   = <VARIABLE WHOSE EDD HAS TO BE DONE>)
    """
    df  = kwargs['dframe']
    var = kwargs['var']
    
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
def bivariate(**kwargs):
    """
    FUNCTION: This would return a Dataframe of Bivariate Analysis of a Variable with Dependant Variable
    
    USAGE GUIDE: 
    bivariate(dframe= <DATAFRAME>,
              var   = <VARIABLE WHOSE BIVARIATE HAS TO BE DONE>,
              dv    = <DEPENDANT VARIABLE>)
    """
    
    
    main = kwargs['dframe']
    var  = kwargs['var']
    dv   = kwargs['dv']
    
    
    df = main[[var,dv]].copy()  ##SMALLER COPIES MORE OPTIMAL
    
    ##CHECKING FOR CORRECT DATA TYPES
    if str(df[var].dtype) not in ['object','int16','int32','int64','float16','float32','float64']:
        return(pd.DataFrame()) #RETURNS EMPTY DATAFRAME FOR FULL RUN
    else:
        ##CATEGORICAL TREATMENT
        if (str(df[var].dtype)=='object' and df[var].nunique()<20) or (df[var].nunique()<=15):
            return(group_by_var(df,var,dv))
        
        elif (str(df[var].dtype)=='object' and df[var].nunique()>=20):
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
def information_value(**kwargs):
    """
    FUNCTION: This would return a Information Value of a variable with Dependant Variable
    
    USAGE GUIDE: 
    information_value(dframe= <DATAFRAME>,
              var   = <VARIABLE WHOSE IV HAS TO BE CREATED>,
              dv    = <DEPENDANT VARIABLE>)
    """
    df = kwargs['dframe']
    var= kwargs['var']
    dv = kwargs['dv']

    check=bivariate(dframe=df,
                     var=var,
                     dv=dv)

    without_null = check[~check['category'].isin(['NULL'])]

    if check['category'].loc[0]=='gt_30_categories':
        return(pd.DataFrame())

    def niv(a,b):
        if a==0 or b==0:
            return(1)
        else:
            return(0)

    def inner_woe(var_df):
        
        ##ADJUSTING FOR 0 EVENTS OR NON EVENTS
        var_df['adj_events']     = var_df.apply(lambda x: x['events']+0.5 if niv(x['events'],x['non_events'])==1 else x['events'],axis=1)
        var_df['adj_non_events'] = var_df.apply(lambda x: x['non_events']+0.5 if niv(x['events'],x['non_events'])==1 else x['non_events'],axis=1)

        ##USUAL IV CALCULATIONs
        var_df['pct_events']     = var_df['adj_events']    /var_df['adj_events'].sum()
        var_df['pct_non_events'] = var_df['adj_non_events']/var_df['adj_non_events'].sum()
        var_df['log_g_b']        = np.log(var_df['pct_events']/var_df['pct_non_events'])
        var_df['g_b']            = var_df['pct_events']-var_df['pct_non_events']
        var_df['woe']            = var_df['g_b']*var_df['log_g_b']

        return(var_df['woe'].sum())

    ##SHOULD RETURN SINGLE VALUE DATAFRAME
    check1 = {
        'var':[],
        'iv':[],
        'mod_iv':[],
        'diff':[]
    }
    check1['var'].append(check['variable'].loc[0])    ##SINGLE VALUE LOC SELECTION
    check1['iv'].append(inner_woe(check))           ##COMPLETE IV
    check1['mod_iv'].append(inner_woe(without_null)) ##NON NULL VALUE IV
    check1['diff']= [x-y for x,y in zip(check1['iv'],check1['mod_iv'])]
    final = pd.DataFrame(check1,index=[0])    
    return(final[['var','iv','mod_iv','diff']])


##*******************************************BIVARIATE PLOTTING*************************************************##
def bivariate_plot(**kwargs):
    """
    FUNCTION: This would return a Bivariate Plot with Event Rate and Observation Count
    
    USAGE GUIDE: 
    bivariate_plot(dframe= <DATAFRAME>,
                   var   = <VARIABLE WHOSE BIVARIATE PLOT HAS TO BE MADE>,
                   dv    = <DEPENDANT VARIABLE>,
                   sort  = < "default":Normal Order, "event_rate": (NON-ORDINAL ONLY) Sorted by Event Rate, 
                           "nobs":Sorted by Count (NON-ORDINAL ONLY)>)
    """
    
    main =kwargs['dframe']
    var  =kwargs['var']
    dv   =kwargs['dv']
    sort =kwargs['sort']   
    
    check=bivariate(dframe=main,
                       var=var,
                        dv=dv)
    
    #ADDITIONAL PROCESSING
    if sort=='event_rate':
        sort_var='dr'
        order=False
    elif sort=='default':
        sort_var='category'
        order=True
    else:
        sort_var='nobs'
        order=False
        
    event_rate = float(check['events'].sum())/float(check['nobs'].sum())*100

    check['dr']      = check['events']/check['nobs']*100
    check['bin_ind'] = check['category'].apply(lambda x: 1 if ('BIN' in str(x))&('[' in str(x)) else 0)
    binned =1 if check['bin_ind'].sum()>0 else 0 

    if binned==1:
        check=check.sort_values(by=['category'],ascending=True).reset_index(drop=True)
    else:
        check=check.sort_values(by=[sort_var],ascending=order).reset_index(drop=True)

    check['x1_labels'] = check['category'].apply(lambda x: x.split('_')[0] if  binned==1  else x )
    check['x2_labels'] = check['category'].apply(lambda x: x.split('_')[1] if (binned==1 and 'NULL' not in x) else '')

    ##PLOT DEFINITIONS
    bar_width = 0.5

    fig = plt.figure(figsize=(10,6)) 
    ax1 = fig.add_subplot(111) 
    ax2 = ax1.twinx() 

    x_vector = np.arange(len(check))+1
    y_obs    = check['nobs'].tolist()
    y_dr     = check['dr'].tolist()
    x_labels = check['x1_labels'].tolist()

    ##RAW PLOTS
    ax1.bar(x_vector,y_obs,align='center',color='lightblue',width=bar_width) #NOBS
    ax2.plot(x_vector,y_dr,color='red')                                      #EVENT RATE
    ax2.axhline(y=event_rate,color='black',linestyle='--')                   #AVERAGE EVENT RATE LINE

    x_range = x_vector                                                       #LABELS
    y_range = check['dr'].tolist()
    for x,y in zip(x_range,y_range):
        label = "{:.1f}%".format(y)
        ax2.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(2,2), # distance from text to points (x,y)
                     ha='center',
                     verticalalignment='bottom',
                     size=12) # horizontal alignment can be left, right or center


    ### BEAUTIFICATION OF GRAPH 
    ax1.set_title("Bivariate For {}".format(check['variable'].loc[0]),fontsize=15)
    ax1.xaxis.set_ticks(x_vector)
    ax1.xaxis.set_ticklabels(x_labels)
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    ax1.set_ylim([0 ,int(check['nobs'].max()*1.05)])
    ax2.set_ylim([0 ,int(check['dr'].max()*1.3)])
    #ax1.get_legend().remove()
    #ax2.get_legend().remove()
    ax1.xaxis.set_label_text("")


    ##DISPLAYING WITH DATAFRAME
    tmpfile = BytesIO()
    fig.savefig(tmpfile,format='png',bbox_inches='tight',pad_inches=0.2) ##REMOVES WHITESPACE
    plt.close(fig) ##PREVENT FROM DISPLAYING IMAGE
    encoded = base64.b64encode(tmpfile.getvalue())

    crow = check[['x1_labels','x2_labels','nobs']]
    crow['%obs'] = check['nobs']/check['nobs'].sum()*100
    crow.index=['']*len(crow)
    test_html=("""
    <style>
    .aligncenter {
        text-align: center;
    }
    </style>
    <div class="row">
      <div class="column">
    """+"""
    <p class="aligncenter">
    <img src=\'data:image/png;base64,{}\' align="left" border="0">
    </p>
    """.format(encoded)+
    """
    </div>
    <div class="column"><center><h4> Bins/Groups for {}</h4></center>
    """.format(var)+"<center>"+crow.to_html()+"</center>"+
    """
    </div></div>
    """)
    display(HTML(test_html))
    
def KS_metric(y_actual,y_predicted):
    """
    FUNCTION: This would return a Dataframe with Binned Prediction Column, Events, Non Events etc. 
    
    USAGE GUIDE: 
    KS_metric     (y_actual     = <LIST OF ACTUAL VALUES>,
                   y_predicted  = <LIST OF PREDICTIONS>)
    """
    y = pd.DataFrame({'Actual':y_actual,'pred':y_predicted})
    y['lift_bins'] = pd.qcut(y['pred'],20,labels=[i for i in np.arange(20)])
    temp = y.groupby(['lift_bins']).agg({'Actual':['count','sum'],'pred':['min','max']}).reset_index()
    temp.columns=['lift_bins','p_min','p_max','total','events']
    temp.sort_values(by=['lift_bins'],ascending=False,inplace=True)
    temp.reset_index(drop=True,inplace=True)
    temp['bins'] = temp['lift_bins'].apply(lambda x: str(20-int(x)))
    temp['Non_Events'] = temp['total']-temp['events']
    temp['Cum_Events'] = temp['events'].cumsum()
    temp['Cum_Non_Events'] = temp['Non_Events'].cumsum()
    temp['Perct_Cum_Events']    =(temp['Cum_Events']/temp['Cum_Events'].max())*100
    temp['Perct_Cum_Non_Events']=(temp['Cum_Non_Events']/temp['Cum_Non_Events'].max())*100
    temp.drop(['lift_bins'],axis=1,inplace=True)
    temp['KS_Diff'] = temp['Perct_Cum_Events']-temp['Perct_Cum_Non_Events']
    temp['avg_prob'] = (temp['p_min']+temp['p_max'])/2
    KS = temp['KS_Diff'].max()
    #print(KS)
    return(temp)


##*****************KS METRIC AND EVALUATION METRICS FOR MODELS*****************###

def eval_metrics(y_actual,**kwargs):
    """
    FUNCTION: This would detailed graphical statistics of plots, metrics for evaluating a classification model 
    
    USAGE GUIDE: 
    eval_metrics(y_actual     = <LIST OF ACTUAL VALUES>,
                   <MODEL_NAME> = <LIST OF PREDICTED VALUE>)
    
    SAMPLE EXAMPLE: 
    
    eval_metrics(y_test,GBM=y_pred,XGB=y_pred)
    
    """
    count = len(kwargs)
    colors = ['blue','red','green','yellow']
    datastore=pd.DataFrame(index=range(count),columns=['MODEL NAME','AUC','GINI','KS STAT'])
    #ROC Curve and Precision Recall Curve
    i=0
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    
    for key,value in kwargs.items():
        #print key,value
        datastore['MODEL NAME'][i]=key
        valuetemp=value.copy()
        #print("AUROC For ",key,"=",roc_auc_score(y_actual,value))
        datastore['AUC'][i]  =roc_auc_score(y_actual,value)
        datastore['GINI'][i] =2*datastore['AUC'][i]-1
        fpr,tpr,thresh = roc_curve(y_actual,value)
        ax1.plot(fpr,tpr,color=colors[i],label=key)
        ax1.legend(loc='upper right')
        ax1.set_title("ROC Curve")
        precision, recall, thresholds = precision_recall_curve(y_actual,value)
        ax2.plot(recall,precision,color=colors[i],label=key)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc='upper right')
        ax2.set_title('Precision Recall Curve')
        temp=KS_metric(y_actual,value)
        datastore['KS STAT'][i] = temp['KS_Diff'].max()
        ax3.plot(temp['bins'].tolist(),temp['Perct_Cum_Events'].tolist(),color=colors[i],label=key)
        ax3.legend(loc='upper right')
        ax3.set_title('Lift Curve')
        
        i+=1
    #x = np.linspace(*ax3.get_xlim())
    #ax3.plot(x,x)
    plt.show()
    
    #SEPARATION GRAPH
    i=1
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('Separation Graph')
    for key,value in kwargs.items():
        y = pd.DataFrame({'Actual':y_actual,key:value})
        if count == 1 : ax1 = fig.add_subplot(1,1,1)
        else : ax1 == fig.add_subplot(2,2,i)
        y[y['Actual']==1][key].plot(kind='hist',bins=30,color='g',alpha=0.45,normed=-True)
        y[y['Actual']==0][key].plot(kind='hist',bins=30,color='b',alpha=0.45,normed=-True)
        ax1.set_title(key)
        i+=1
    plt.show()
    datastore.index= ['']*len(datastore)
    display(HTML("<center>"+datastore.to_html()+"</center>"))
    
##*****************ANALYSING THE PRESENCE OF EVENT RATE WITHINT THE NULL BUCKET*****************###

def null_bucket_analysis(**kwargs):
    
    main       = kwargs['dframe']
    var        = kwargs['var']
    dv         = kwargs['dv']
    vc         = main[var].value_counts()
    null_main  = main[main[var].isnull()]
    
    ##TESTING FOR DEFAULT RATES
    if len(null_main)==0:
        return(pd.DataFrame())
    else:
        event_rate  = np.round(float(main[dv].sum())/float(len(main))*100,2)
        null_rate   = np.round(float(null_main[dv].sum())/float(len(null_main))*100,2)
        pop_pct     = np.round(float(len(null_main))/float(len(main))*100,2)
        unq_obs     = main[var].nunique()
        modal_pct   = np.round(float(vc.head(1).tolist()[0])/float(len(main)-len(null_main))*100,2)
        modal_value = vc.index[0]

        frame={
         'variable'       : var,
         'event_rate'     : event_rate,
         'null_event_rate': null_rate,
         'pct_null'       : pop_pct,
         'distinct'       : unq_obs,
         'modal_value'    : modal_value,
         'modal_pct'      : modal_pct
        }
        ret_frame = pd.DataFrame(frame,index=[0])
        ret_frame = ret_frame[['variable','event_rate','null_event_rate',
                               'pct_null','distinct','modal_value','modal_pct']]
        return(ret_frame)


#####*******************************************INFORMATION VALUE PLOT*********************************##
def information_value_plot(**kwargs):
    """
    FUNCTION: This would plot the information value of Top 20 Variables (Or Less) as provided within the input 
    
    USAGE GUIDE: 
    information_value_plot(
        *run_list  = <LIST OF VARIABLES ON WHICH IV PLOT NEEDS TO BE EXECUTED>
        *skip_list = <LIST OF VARIABLES ON WHICH FUNCTION NEEDS TO BE SKIPPED (Should Include Dependant Variable)>
        *dframe    = <DATAFRAME>,
        *dv        = <DEPENDANT VARIABLE>)
    
    SAMPLE EXAMPLE: 
    
    information_value_plot(
                           dframe=main,
                           dv='Survived',
                           run_list=main.columns,
                           skip_list=['Survived']
                           )
    """
    ##PROPER WAY OF USING KWARGS
    run_list  = kwargs['run_list']
    skip_list = kwargs['skip_list']
    dframe    = kwargs['dframe']
    dv        = kwargs['dv']
    
    df = full_data_run(dframe=dframe,
                       run_list=run_list,
                       skip_list=skip_list,
                       func=information_value,
                       dv=dv)

    df.sort_values(by=['iv'],ascending=False,inplace=True)
    df.reset_index(inplace=True,drop=True)
    df=df.head(20)
    fig = plt.figure(figsize=(20,6)) 
    ax1 = fig.add_subplot(111)

    ax1.axhline(y=0.1,color='red',linestyle='--'   )
    ax1.axhline(y=0.2,color='orange',linestyle='--')
    ax1.axhline(y=0.5,color='green',linestyle='--' )

    df.plot(kind='bar',x='var',y='iv',rot=0,ax=ax1,align='center')

    count = len(df) if len(df)<=20 else 20
    
    ax1.set_title("Information Value of Top {} Variables".format(str(count)),fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.xaxis.set_label_text("")
    ax1.axes.get_yaxis().set_ticks([])
    ax1.get_legend().remove()

    x_range = np.arange(len(df)+1)  #LABELS
    y_range = df['iv'].tolist()

    for x,y in zip(x_range,y_range):
            label = "{:.2}".format(y)
            ax1.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(2,2), # distance from text to points (x,y)
                         ha='center',
                         verticalalignment='bottom',
                         size=12) # horizontal alignment can be left, right or center
    plt.show()