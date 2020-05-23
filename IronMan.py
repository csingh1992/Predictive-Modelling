import os
import time
import warnings
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import itertools
from ipywidgets import IntProgress
from IPython.display import display,HTML

##IMPROVING DISPLAY OPTIONS
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('max_colwidth', -1)

warnings.filterwarnings('ignore')

import random
import pandas as pd
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import confusion_matrix
from ipywidgets import interact, interact_manual
sns.set()

from ipywidgets import IntProgress
from IPython.display import display,HTML

numerical_dtypes = ['bool','int16','int32','int64','float16','float32','float64']

print("""
*******************************************************************************
JARVIS V1.0   AUTHOR: CHETAN CHAUHAN
Following program is a series of useful tools for Predictive Modelling
Initiate the class object by passing in the final modelling dataset as shown
Ensure that the dependant variable is binary 0 or 1
*******************************************************************************

model = Jarvis(dataframe,dep_var)
test  = Utils()

""")

class Utils():
    def __init__(self):
        print(""" REPOSITORY
            1. confusion_matrix_report
            2. print_model_feature_importance
            3. cheat_sheet
                A. 'basic_model'
                B. 'grid_search'
                C. 'tree_params' """)

    def cheat_sheet(self,snip):
        if snip=='basic_model':
            print("""
            ********************************
            BASIC MODEL CREATION CHEAT SHEET
            ********************************

            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, XGBoostClassifier
            from sklearn.tree import DecisionTreeClassifier

            model = GradientBoostingClassifier()

            X = DF[feature_list]
            y = DF['BAD'] #DV SERIES FOR STRATIFY

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=713, stratify=y)

            model.fit(X_train,y_train)

            pred_test  = list(model.predict_proba(X_test)[:,1])
            pred_train = list(model.predict_proba(X_train)[:,1])""")
        elif snip=='grid_search':
            print("""
            ******************************
            BASIC GRID SEARCH CHEAT SHEET
            ******************************

            ## GRID SEARCH CV ##

            from sklearn.grid_search import GridSearchCV
            params = {"n_neighbors": np.arange(1,3),
                      "metric"     : ["euclidean", "cityblock"]}
            grid = GridSearchCV(estimator=knn,
                                param_grid=params,
                                cv = 3, #DEFAULT IS 5
                                verbose = 2 ##ANY INTEGER MEANS IT WILL PRINT ALL FITS)
            grid.fit(X_train, y_train)
            print(grid.best_score_)
            print(grid.best_estimator_.n_neighbors)

            ## RANDOM SEARCH CV ##
            from sklearn.model_selection import RandomizedSearchCV
            rf_cv = RandomizedSearchCV( estimator = rf,
                                        param_distributions = param_grid,
                                        n_iter = 100,  #Random 100 Iterations?
                                        cv = 3,
                                        verbose=2,
                                        random_state=42)
            rf_cv.fit(DF[feats],DF['BAD'])
            rf_cv.best_params_
                """)
        elif snip=='tree_params':
            print("""
            ***************************
            IMPORTANT TREE MODEL PARAMS
            ***************************

            A. GBM: (Boosting Params vs Tree Params)
                1. max_depth        : [ 3,4,5 ]
                2. learning_rate    : [0.1, 0.01, 0.5]
                3. n_estimators     : [100, 200,  500]
                4. min_samples_split: []
                5. min_samples_leaf : []
                6. max_features ['auto','log2',None,'sqrt']
                7. n_jobs = -1

            B. Decision Tree:
                1. max_depth :
                2. min_samples_split :
                3. min_samples_leaf :
                4. criterion: 'gini','entropy'

            C. XGBoost has some more, refer to AV Link for it! """)
        else:
            print("Wrong Selection!")

    def confusion_matrix(self,y_true,y_pred,target='Dependant Variable',target_names=['Non-Fraud','Fraud']):
        def con_mat(threshold=.5,normalize='No'):
            x = threshold
            y_hat = [1 if y>=float(x) else 0 for y in y_pred]
            cm = confusion_matrix(y_true,y_hat)
            tn,fp,fn,tp = confusion_matrix(y_true,y_hat).ravel()
            accuracy= round((float(tp+tn)/float(tp+tn+fp+fn)),2)
            precision= round((float(tp)/float(tp+fp)),2)
            recall= round((float(tp)/float(tp+fn)),2)
            misclass=1-accuracy
            cmap=plt.get_cmap('Blues')
            plt.figure(figsize=(8,6))
            plt.imshow(cm,interpolation='nearest',cmap=cmap)
            plt.title('Confusion Matrix', fontsize=13, fontweight= 'bold')
            plt.colorbar()

            if target_names is not None:
                tick_marks= np.arange(len(target_names))
                plt.xticks(tick_marks,target_names)
                plt.yticks(tick_marks,target_names)

            if normalize =='Yes':
              cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
              if normalize =='Yes':
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",fontsize=12, fontweight='bold',
                             color="white" if cm[i, j] > threshold else "black")
              else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",fontsize=12, fontweight='bold',
                             color="green" if cm[i, j] > threshold else "black")
            plt.tight_layout()
            plt.ylabel('True Label')
            plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
            plt.show()
        inter=interact(con_mat, threshold=[i/100 for i in np.arange(1,101)], normalize=['Yes','No'])
        display(inter)

    def print_model_feature_importance(self,model,feature_list,fsize=(10,6),return_data=False):
        feat_imp = pd.Series(model.feature_importances_,feature_list).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances',figsize=fsize)
        plt.ylabel('Feature Importance Plot for Classifier')
        if return_data==True:
            return(feat_imp)

class Jarvis():

    model_bench_df = pd.DataFrame()

    def __init__(self,dframe,bad,non_model=[]):
        for x in list(dframe.columns):
            if x in non_model:
                continue
            elif str(dframe[x].dtypes) not in numerical_dtypes+['object']:
                print("VARIABLE ",x," POSSIBLE NON MODEL!")
        self.dframe = dframe.copy()
        self.bad    = bad
        self.df_len,self.num_c  = dframe.shape
        self.non_model = non_model
        self.all_feats = [x for x in list(dframe.columns) if x not in [bad]+non_model]
        print("""
            *******************
            Functions:
            *******************
           01. edd(skip=False/True)
           02. _single_var_edd(var)
           03. event_rate_plot()
           04. univariate_analysis()
           05. bivariate_plot()
           06. bivariate_analysis()
           07. feature_importance(feature_list=[],plot=True/False)
           08. _single_var_feat_imp(var,cat_check=True/False)
           09. predictive_missing_value(plot=True/False)
           10. model_benchmarking(clf_dict={},feats=[])
           11. model_benchmark_plot()
           12. information_value(feats=[])
           13. _single_var_info_value(var)
           14. model_evaluation_report(y_true,y_pred_dict={})
           15. model_performance_plot(metric='ROC_CURVE'/'PR_CURVE'/'SEPARATION_CURVE',y_true,y_pred_dict={})
             """)

    def _edd_formatter(self,v_index,v_value,v_count):
        """ Simple Formatting for EDD TOP/BOTTOM Objects """
        return(str(v_index)+' | '+str(v_value)+' | '+str(np.round(float(v_value)/float(v_count)*100,2))+'%')

    def _single_var_edd(self,var):
        """EDD for a single Variable, removed redundant items like Mode Count etc"""
        var_data_type   = str(self.dframe[var].dtypes)
        var_count       = self.dframe[var].count()
        missing_count   = self.df_len-var_count
        distinct_values = self.dframe[var].dropna().nunique()
        missing_rate    = np.round(float(missing_count)/float(self.df_len),3)
        """Modal Value %age"""
        modal_value = self.dframe[var].value_counts().index[0]
        modal_freq  = self.dframe[var].value_counts().head(1).tolist()[0]
        modal_pct   = np.round(float(modal_freq)/float(self.df_len),2)
        if var_data_type in ('object'):
            min_value    = '--'
            max_value    = '--'
            mean_value   = '--'
            median_p50   = '--'
            vc  = self.dframe[var].value_counts()
            vci = vc.index
            vcl = vc.tolist()
            if distinct_values == 1:
                top2_p05=top3_p25=bot3_p75=bot2_p90='--'
                bot1_p99=top1_p01= self._edd_formatter(vci[0],vcl[0],var_count)
            elif distinct_values == 2:
                top3_p25=bot3_p75='--'
                bot2_p90=top1_p01= self._edd_formatter(vci[0],vcl[0],var_count)
                bot1_p99=top2_p05= self._edd_formatter(vci[1],vcl[1],var_count)
            else:
                top1_p01     = self._edd_formatter(vci[0] ,vcl[0] ,var_count)
                top2_p05     = self._edd_formatter(vci[1] ,vcl[1] ,var_count)
                top3_p25     = self._edd_formatter(vci[2] ,vcl[2] ,var_count)
                bot3_p75     = self._edd_formatter(vci[-3],vcl[-3],var_count)
                bot2_p90     = self._edd_formatter(vci[-2],vcl[-2],var_count)
                bot1_p99     = self._edd_formatter(vci[-1],vcl[-1],var_count)
        else:
            min_value       = self.dframe[var].min()
            max_value       = self.dframe[var].max()
            mean_value      = self.dframe[var].mean()
            median_p50      = self.dframe[var].median()
            top1_p01        = self.dframe[var].quantile(0.01)
            top2_p05        = self.dframe[var].quantile(0.05)
            top3_p25        = self.dframe[var].quantile(0.25)
            bot3_p75        = self.dframe[var].quantile(0.75)
            bot2_p90        = self.dframe[var].quantile(0.90)
            bot1_p99        = self.dframe[var].quantile(0.99)
        frame = collections.OrderedDict()
        frame['var']          = var
        frame['data_type']    = var_data_type
        frame['total']        = self.df_len
        frame['count']        = var_count
        frame['missing']      = missing_count
        frame['missing_rate'] = missing_rate
        frame['distinct']     = distinct_values
        frame['mode']         = modal_value
        frame['modal_freq']   = modal_freq
        frame['modal_pct']    = modal_pct
        frame['min']          = min_value
        frame['max']          = max_value
        frame['mean']         = mean_value
        frame['median']       = median_p50
        frame['top1_p01']     = top1_p01
        frame['top2_p05']     = top2_p05
        frame['top3_p25']     = top3_p25
        frame['bot3_p75']     = bot3_p75
        frame['bot2_p90']     = bot2_p90
        frame['bot1_p99']     = bot1_p99
        single_frame = pd.DataFrame(frame,index=[0])
        return(single_frame)

    def edd(self,skip=False):
        """Instructions: Skip = True [Skips for Dependant Variable]"""
        if skip==True:
          print("Skipping Dependant & Non Model Variables")
        final_edd = pd.DataFrame()
        iter_list = list(self.dframe.columns)
        iter_list = self.all_feats if skip==True else iter_list
        for var in iter_list:
            final_edd = pd.concat([final_edd,self._single_var_edd(var)],axis=0)
        final_edd.sort_values(by=['missing_rate'],ascending=False,inplace=True)
        final_edd.reset_index(inplace=True,drop=True)
        return(final_edd)

    def event_rate_plot(self):
        """Instructions: Use this function to print a graph for the event rate"""
        print("Total Records :",self.df_len)
        print("Event Rate    :",np.round(float(len(self.dframe[self.dframe[self.bad]==1])/float(self.df_len)),4))
        plt.figure(figsize=(5,5))
        splot = sns.countplot(self.dframe[self.bad])
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.0f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha = 'center', va = 'center',
                           xytext = (0, 10), textcoords = 'offset points')
        splot.get_yaxis().set_visible(False)
        plt.show()

    def _null_handler(self,var):
        if len(self.dframe[self.dframe[var].isnull()])==0:
            return(pd.DataFrame())
        else:
            null_dict = {}
            null_dict['events']     = self.dframe[self.dframe[var].isnull()][self.bad].sum()
            null_dict['nobs']       = self.dframe[self.dframe[var].isnull()][self.bad].count()
            null_dict['non_events'] = null_dict['nobs'] - null_dict['events']
            null_dict['variable']   = var
            null_dict['category']   = 'NULL'
            null_df = pd.DataFrame(null_dict,index=[0])
            null_df = null_df[['variable','category','nobs','events','non_events']]
            return(null_df)

    def _group_events_by_var(self,var):
        if var==self.bad:
            check=pd.DataFrame()
        elif 'QCUT' in var.split('_'):
            real_var = var.split('QCUT_')[1] ##to handle variable names with underscore
            test     = self.dframe[~self.dframe[real_var].isnull()].groupby([var]).agg({self.bad:['sum','count'],
                                                                   real_var:['min','max']}).reset_index()
            test.columns=test.columns.droplevel()
            test.rename(columns={ '':'category',
                                 'sum':'events',
                                 'count':'nobs'},inplace=True)
            test['variable']   = real_var
            test['non_events'] = test['nobs']-test['events']
            test['final_category'] = test.apply(lambda x: str(x['category'])+'_['+
                                                          str(np.round(x['min'],2))+'-'+
                                                          str(np.round(x['max'],2))+']', axis=1)
            test.drop(['category'],axis=1,inplace=True)
            test.rename(columns={'final_category':'category'},inplace=True)
            test.sort_values(by=['category'],ascending=True,inplace=True)
            test  = test[['variable','category','nobs','events','non_events']]
            check = pd.concat([self._null_handler(real_var),test],axis=0)
            check.reset_index(drop=True,inplace=True)
            sum_events = check['events'].sum()
            check['event_rate'] = np.round((check['events']/check['nobs'])*100,2)
            check['event_pct']  = np.round((check['events']/sum_events)*100,2)
        else:
            test_df = self.dframe[[var,self.bad]].copy()
            test = test_df[~test_df[var].isnull()].groupby([var]).agg({self.bad:['sum','count']}).reset_index()
            test.columns = ['category','events','nobs']
            test['variable'] = var
            test['non_events'] = test['nobs']-test['events']
            test = test[['variable','category','nobs','events','non_events']]
            check = pd.concat([self._null_handler(var),test],axis=0)
            check.reset_index(drop=True,inplace=True)
            sum_events = check['events'].sum()
            check['event_rate'] = np.round((check['events']/check['nobs'])*100,2)
            check['event_pct']  = np.round((check['events']/sum_events)*100,2)
        return(check)

    def _single_var_feat_imp(self,var,cat_check=False):
        if var==self.bad:
            print("FEATURE IMPORTANCE : Selected Dependant Variable !",self.bad)
            auc,auc_std = 0.5,0
        elif (cat_check==False) and (str(self.dframe[var].dtypes)=='object'):
            print("FEATURE IMPORTANCE : Selected Object Variable as Numerical!")
            auc,auc_std = 0.5,0
        elif (cat_check==True) and (str(self.dframe[var].dtypes) in numerical_dtypes):
            print("FEATURE IMPORTANCE : Selected Non Object Variable as Object!")
            auc,auc_std = 0.5,0
        elif var in self.non_model:
            print("FEATURE IMPORTANCE: Selected Non Model Variable!")
            auc,auc_std = 0.5,0
        else:
            if cat_check==True:
                self.dframe['InMOD_'+var] = (   self.dframe[var].fillna('NULL')
                                           if   self.dframe[var].count()!=self.df_len
                                           else self.dframe[var])
                if self.dframe['InMOD_'+var].nunique()>10:
                    temp2 = self._group_events_by_var('InMOD_'+var)
                    mapper_dict = pd.Series(temp2['event_rate'].values,index=temp2['category']).to_dict()
                    self.dframe['InMOD_'+var] = self.dframe['InMOD_'+var].map(mapper_dict)
                    auc,auc_std = self._single_var_feat_imp('InMOD_'+var)
                    self.dframe.drop(['InMOD_'+var],axis=1,inplace=True)
                else:
                    dum = pd.get_dummies(self.dframe['InMOD_'+var])
                    self.dframe = pd.concat([self.dframe,dum],axis=1)
                    model = DecisionTreeClassifier(min_samples_leaf=0.05)
                    self.model_bench_df = pd.DataFrame()
                    self.model_benchmarking({'DTFI':model},feats=list(dum.columns),folds=5,print_cond=False)
                    auc,auc_std = (np.round(self.model_bench_df['test'].mean(),5),
                                   np.round(self.model_bench_df['test'].std() ,5))
                    self.model_bench_df = pd.DataFrame()
                    self.dframe.drop(list(dum.columns)+['InMOD_'+var],axis=1,inplace=True)
            else:
                self.dframe['InMOD_'+var] = (     self.dframe[var].fillna(-99999)
                                             if   self.dframe[var].count()!=self.df_len
                                             else self.dframe[var] )
                self.model_bench_df = pd.DataFrame()
                model = DecisionTreeClassifier(min_samples_leaf=0.05)
                self.model_benchmarking({'DTFI':model},feats=['InMOD_'+var],folds=5,print_cond=False)
                auc,auc_std = (np.round(self.model_bench_df['test'].mean(),5),
                               np.round(self.model_bench_df['test'].std() ,5))
                self.model_bench_df = pd.DataFrame()
                self.dframe.drop(['InMOD_'+var],axis=1,inplace=True)
        return([auc,auc_std])

    def feature_importance(self,feature_list=[],plot=False,fsize=(15,6)):
        feature_list = self.all_feats if len(feature_list)==0 else feature_list
        feat_imp_df  = pd.DataFrame()
        for var in feature_list:
            if str(self.dframe[var].dtypes)=='object':
                imp,imp_std = self._single_var_feat_imp(var,cat_check=True)
                excess = imp - 0.5
            else:
                imp,imp_std = self._single_var_feat_imp(var)
                excess = imp - 0.5
            feat_imp_df = pd.concat([feat_imp_df,pd.DataFrame({'var':var,'feat_imp':imp,'imp_std':imp_std,
                                                               'excess':excess},index=[0])],axis=0)
        feat_imp_df.sort_values(by=['excess'],ascending=False,inplace=True)
        feat_imp_df.reset_index(inplace=True,drop=True)
        if plot==True:
            feat_imp_df.plot(kind='bar',x='var',y='excess',figsize=fsize,rot=60)
            plt.show()
        return(feat_imp_df)

    def model_benchmarking(self,clf_dict,feats=[],folds=5,print_cond=True,reset_df=True):
        self.model_bench_df = self.model_bench_df if reset_df==False else pd.DataFrame()
        feats = self.all_feats if len(feats)==0 else feats
        data_folds = StratifiedKFold(n_splits=folds,random_state=None, shuffle=True)
        max_algo_name_len = max([len(x) for x in clf_dict.keys()])
        for algo in clf_dict.keys():
            clf_algo = clf_dict[algo]
            for train_index,test_index in data_folds.split(self.dframe[feats],self.dframe[self.bad]):
                X_train,y_train = (self.dframe[self.dframe.index.isin(train_index)][feats],
                                     self.dframe[self.dframe.index.isin(train_index)][self.bad])
                X_test ,y_test  = (self.dframe[self.dframe.index.isin(test_index)][feats],
                                     self.dframe[self.dframe.index.isin(test_index)][self.bad])
                clf_algo.fit(X_train,y_train)
                pred_test  = list(clf_algo.predict_proba(X_test)[:,1])
                pred_train = list(clf_algo.predict_proba(X_train)[:,1])
                test_auc  = np.round(roc_auc_score(y_test,pred_test),5)
                train_auc = np.round(roc_auc_score(y_train,pred_train),5)
                if print_cond==True:
                    train_len,test_len = len(X_train),len(X_test)
                    train_er ,test_er  = (np.round((float(y_train.sum())/float(train_len))*100,2),
                                          np.round((float(y_test.sum()) /float(test_len))*100 ,2))
                    print_str=(("""MODEL: {0:#0#}| TRAIN LEN: {1:#1#}| TEST LEN: {2:#1#}| """+
                               """TRAIN ER: {3:3.2f}%| TEST ER: {4:3.2f}%| TRAIN AUC:{5:7.5f}| TEST AUC:{6:7.5f}"""
                               ).replace('#0#',str(max_algo_name_len)).replace('#1#',str(len(str(self.df_len)))))
                    print(print_str.format(algo,train_len,test_len,train_er,test_er,train_auc,test_auc))
                self.model_bench_df = pd.concat([self.model_bench_df,
                pd.DataFrame({'model': algo,'test' : test_auc, 'train': train_auc},index=[0])],axis=0)
        self.model_bench_df.reset_index(drop=True,inplace=True)

    def _single_var_missing_gini(self,var):
        temp = pd.DataFrame()
        temp[var]       = pd.Series(np.where(self.dframe[var].isnull(),1,0))
        missing_rate    = np.round(float(temp[var].sum())/float(self.df_len),2)
        temp[self.bad]  = self.dframe[self.bad]
        gpr = temp.groupby([var]).agg({self.bad:['sum','count']})
        gpr.reset_index(inplace=True)
        gpr.columns=[self.bad,'sum','count']
        gpr['event_rate'] = gpr['sum']/gpr['count']
        gpr['non_event_rate'] = 1 - gpr['event_rate']
        gpr['pop_percent'] = gpr['count']/gpr['count'].sum()
        gpr['gini_contri'] = (gpr['event_rate']**2+gpr['non_event_rate']**2)*gpr['pop_percent']
        return((missing_rate,np.round(gpr['gini_contri'].sum(),5)))

    def predictive_missing_value(self,plot=False,fsize=(15,6)):
        missing_predness = pd.DataFrame()
        for var in self.all_feats:
            miss,pred        = self._single_var_missing_gini(var)
            missing_predness = pd.concat([missing_predness,
                               pd.DataFrame({'var':var,'missing_rt':miss,'gini':pred},index=[0])],axis=0)
        missing_predness.sort_values(by=['gini'],ascending=False,inplace=True)
        missing_predness.reset_index(inplace=True,drop=True)
        if plot==True:
            fig, axes = plt.subplots(1,1, figsize=fsize)
            ax2 = axes.twinx()
            missing_predness.plot(kind='bar' ,x='var',y='gini'      ,color='b',rot=60,ax=axes)
            missing_predness.plot(kind='line',x='var',y='missing_rt',color='r',rot=60,ax=ax2 )
            axes.grid(False)
            ax2.grid(False)
            axes.set_ylabel('GINI Of Missing')
            ax2.set_ylabel('Missing Rate')
            axes.get_legend().remove()
            ax2.get_legend().remove()
            plt.show()
        return(missing_predness)

    def model_benchmark_plot(self):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(0.5  ,1.05)
        ax.set_ylim(0.5  ,1.05)
        ax.set_title('TRAIN (AUC) vs TEST (AUC) of MODELs')
        sns.scatterplot(data=self.model_bench_df, x='test', y='train', hue='model',ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        plt.show()

    def model_performance_plot(self,metric,y_true,y_pred_dict={}):
        if metric=='ROC_CURVE':
            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(1,1,1)
            for model in y_pred_dict.keys():
                fpr,tpr,thresh = roc_curve(y_true,y_pred_dict[model])
                ax1.plot(fpr,tpr,label=model)
            ax1.plot([[0,0],[1,1]],linestyle='--',color='black')
            ax1.legend(loc='upper right')
            ax1.set_title("ROC Curve",fontsize=20)
            ax1.set_xlabel('FPR (False Positive Rate)')
            ax1.set_ylabel('TPR (True Positive Rate)')
        elif metric=='PR_CURVE':
            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(1,1,1)
            for model in y_pred_dict.keys():
                precision, recall, thresholds = precision_recall_curve(y_true,y_pred_dict[model])
                ax1.plot(recall,precision,label=model)
            ax1.legend(loc='upper right')
            ax1.set_title('Precision Recall Curve',fontsize=20)
            ax1.set_xlabel('Recall')
            ax1.set_ylabel('Precision')
        elif metric=='SEPARATION_CURVE':
            num_plots = len(y_pred_dict.keys())
            fig, axes = plt.subplots(num_plots,1,figsize=(15,5*num_plots),sharex=True,sharey=True)
            axes      = axes if num_plots==1 else axes.flatten()
            for num,model in enumerate(y_pred_dict.keys()):
                if num_plots==1:
                    y = pd.DataFrame({'Actual':y_true,model:y_pred_dict[model]})
                    y[y['Actual']==1][model].plot(kind='hist',bins=30,color='g',alpha=0.45,density=True,ax=axes)
                    y[y['Actual']==0][model].plot(kind='hist',bins=30,color='b',alpha=0.45,density=True,ax=axes)
                    axes.set_title(model,fontsize=20)
                    axes.set_xlim([0,1])
                else:
                    y = pd.DataFrame({'Actual':y_true,model:y_pred_dict[model]})
                    y[y['Actual']==1][model].plot(kind='hist',bins=30,color='g',alpha=0.45,density=True,ax=axes[num])
                    y[y['Actual']==0][model].plot(kind='hist',bins=30,color='b',alpha=0.45,density=True,ax=axes[num])
                    axes[num].set_title(model,fontsize=20)
                    axes[num].set_xlim([0,1])

    def bivariate_analysis(self,var,cat_check=False):
        if str(self.dframe[var].dtypes) not in numerical_dtypes+['object']:
            print("Non Model Data Type for :",var)
            return(pd.DataFrame())
        else:
            if cat_check==True:
                grp=self._group_events_by_var(var)
            else:
                try:
                    self.dframe['QCUT_'+var]=pd.qcut(self.dframe[var],10,labels=['BIN'+str(i) for i in range(10)])
                    grp = self._group_events_by_var('QCUT_'+var)
                    self.dframe.drop(['QCUT_'+var],axis=1,inplace=True)
                except:
                    self.dframe['QCUT_'+var]=pd.qcut(self.dframe[var].rank(method='first'),10,
                        labels=['BIN'+str(i) for i in range(10)])
                    grp = self._group_events_by_var('QCUT_'+var)
                    self.dframe.drop(['QCUT_'+var],axis=1,inplace=True)
        return(grp)

    def bivariate_plot(self,var,sort_by='default'):
        if var==self.bad:
          print("BIVARIATE PLOT: Selected Dependant Variable")
          return(pd.DataFrame())
        cat_check = True if str(self.dframe[var].dtypes)=='object' else False
        check = self.bivariate_analysis(var,cat_check)
        if sort_by=='event_rate':
            sort_var='event_rate'
            order=False
        elif sort_by=='default':
            sort_var='category'
            order=True
        else:
            sort_var='nobs'
            order=False
        event_rate = float(check['events'].sum())/float(check['nobs'].sum())*100
        check['bin_ind'] = check['category'].apply(lambda x: 1 if ('BIN' in str(x))&('[' in str(x)) else 0)
        binned = 1 if check['bin_ind'].sum()>0 else 0
        if binned==1:
            check=check.sort_values(by=['category'],ascending=True).reset_index(drop=True)
        else:
            check=check.sort_values(by=[sort_var],ascending=order).reset_index(drop=True)
        check['LABEL']  = check['category'].apply(lambda x: x.split('_')[0] if  binned==1  else x )
        check['VALUES'] = check['category'].apply(lambda x: x.split('_')[1] if (binned==1 and 'NULL' not in x) else '')
        ##BEAUTIFICATION OF GRAPH
        bar_width = 0.5
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        x_vector = np.arange(len(check))+1
        y_obs    = check['nobs'].tolist()
        y_dr     = check['event_rate'].tolist()
        x_labels = check['LABEL'].tolist()
        ax1.bar(x_vector,y_obs,align='center',color='lightblue',width=bar_width)
        ax2.plot(x_vector,y_dr,color='red')
        ax2.axhline(y=event_rate,color='black',linestyle='--')
        x_range = x_vector
        y_range = check['event_rate'].tolist()
        for x,y in zip(x_range,y_range):
            label = "{:.1f}%".format(y)
            ax2.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(2,2), # distance from text to points (x,y)
                         ha='center',
                         verticalalignment='bottom',
                         size=12) # horizontal alignment can be left, right or center
        ax1.set_title("Bivariate For {}".format(check['variable'].loc[0]),fontsize=15)
        ax1.xaxis.set_ticks(x_vector)
        ax1.xaxis.set_ticklabels(x_labels)
        ax1.xaxis.set_tick_params(labelsize=12)
        ax1.yaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_tick_params(labelsize=12)
        ax1.set_ylim([0 ,int(check['nobs'].max()*1.05)])
        ax2.set_ylim([0 ,int(check['event_rate'].max()*1.3)])
        ax1.xaxis.set_label_text("")
        ##DISPLAYING WITH DATAFRAME
        tmpfile = BytesIO()
        fig.savefig(tmpfile,format='png',bbox_inches='tight',pad_inches=0.2) ##REMOVES WHITESPACE
        plt.close(fig) ##PREVENT FROM DISPLAYING IMAGE
        encoded = base64.b64encode(tmpfile.getvalue())
        crow = check[['LABEL','VALUES','nobs']]
        crow['%obs'] = check['nobs']/check['nobs'].sum()*100
        crow.index=['']*len(crow)
        test_html=("""
        <style>
        .aligncenter {text-align: center;}
        </style>
        <div class="row">
        <div class="column">
        """+"""
        <p class="aligncenter">
        <img src=\'data:image/png;base64,{}\' align="left" border="0">
        </p>
        """.format(encoded.decode("utf-8"))+
        """
        </div>
        <div class="column"><center><h4> Bins/Groups for {}</h4></center>
        """.format(var)+"<center>"+crow.to_html()+"</center>"+
        """
        </div></div>
        """)
        display(HTML(test_html))

    def univariate_analysis(self,var,num_as_cat=False):

        if (str(self.dframe[var].dtypes) in numerical_dtypes)&(num_as_cat==True):
            self.dframe[var+'___CATEGORICAL'] = self.dframe[var].astype('str')
            self.univariate_analysis(var+'___CATEGORICAL')
            self.dframe.drop([var+'___CATEGORICAL'],axis=1,inplace=True)

        elif (str(self.dframe[var].dtypes)=='object'):
            ##BASIC STATS DF
            temp = self.dframe[~self.dframe[var].isnull()][[var]]
            temp[var+'_UVA'] = temp[var].apply(lambda x: len(str(x).strip()))
            max_string_value , min_string_value = temp[var+'_UVA'].max(), temp[var+'_UVA'].min()
            max_string_value = temp[temp[var+'_UVA']==max_string_value][var].tolist()[0]
            min_string_value = temp[temp[var+'_UVA']==min_string_value][var].tolist()[0]
            del temp
            basic_stats = pd.DataFrame(
                {'Descript':['Distinct Values','Missing Rate','Largest_Entry','Smallest_Entry'],
                 'Stats'   :[str(self.dframe[var].nunique()),
                             str(np.round(100-(self.dframe[var].count()/self.df_len)*100,2))+' %',
                             str(max_string_value),
                             str(min_string_value)]})

            ##DISPLAY CATEGORY DF
            grp = self.dframe[var].fillna('** MISSING **').value_counts()
            display_df = pd.Series.append(grp.head(5),pd.Series(grp.sum()-grp.head(5).sum(),index=['** OTHER RECORDS **']))
            display_df = pd.DataFrame(display_df).reset_index().rename({'index':'Categories',0:'Counts'},axis=1)
            display_df['Count%'] = np.round(display_df['Counts']/display_df['Counts'].sum()*100,2)
            ##TO HIDE INDEX (Create for Both)
            display_df.index=['']*len(display_df)
            basic_stats.index=['']*len(basic_stats)
            ##FIGURE OBJECT
            fig = plt.figure(figsize=(5,4))
            grp[:5].plot(kind='barh')
            tmpfile = BytesIO()
            fig.savefig(tmpfile,format='png',bbox_inches='tight',pad_inches=0.1) ##REMOVES WHITESPACE
            plt.close(fig) ##PREVENT FROM DISPLAYING IMAGE
            encoded = base64.b64encode(tmpfile.getvalue())
            test_html=("""
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            * {
              box-sizing: border-box;
            }

            /* Create three equal columns that floats next to each other */
            .column {
              float: left;
              width: 33.33%;
              padding: 10px;
              height: 300px; /* Should be removed. Only for demonstration */
            }

            /* Clear floats after the columns */
            .row:after {
              content: "";
              display: table;
              clear: both;
            }
            </style>
            </head>
            <body>
            <center><h2>Univariate Analysis for ##0##</h2></center>
            <div class="row">
                <div class="column" >
                <p class="aligncenter"><img src=\'data:image/png;base64,##1##\' align="left" border="0"></p>
                </div>
                <div class="column" >
                <center><h4> Basic Stats</h4></center>
                <center>##2##</center>
                </div>
                <div class="column" >
                <center><h4> Top 5 Categories</h4></center>
                <center>##3##</center>
                </div>
            </div>
            </body>
            </html>""".replace('##0##',var)
                      .replace('##1##',encoded.decode("utf-8"))
                      .replace('##2##',basic_stats.to_html())
                      .replace('##3##',display_df.to_html()))
            display(HTML(test_html))
        elif str(self.dframe[var].dtypes) in numerical_dtypes:
            basic_stats = pd.DataFrame(
                {'Descript':['Distinct Values','Missing Rate','Min Value','Max Value','Median Value'],
                 'Stats'   :[str(self.dframe[var].nunique()),
                             str(np.round(100-(self.dframe[var].count()/self.df_len)*100,2))+' %',
                             str(np.round(self.dframe[var].min(),2)),
                             str(np.round(self.dframe[var].max(),2)),
                             str(np.round(self.dframe[var].median(),2))
                             ]})
            basic_stats.index=['']*len(basic_stats)
            ##FIGURE 1
            fig = plt.figure(figsize=(5,4))
            self.dframe.boxplot(column=[var],vert=False)
            graph1 = BytesIO()
            fig.savefig(graph1,format='png',bbox_inches='tight',pad_inches=0.1) ##REMOVES WHITESPACE
            plt.close(fig)
            ##FIGURE 2
            fig = plt.figure(figsize=(5,4))
            sns.distplot(self.dframe[var],hist=True,bins=10,kde=True)
            graph2 = BytesIO()
            fig.savefig(graph2,format='png',bbox_inches='tight',pad_inches=0.1) ##REMOVES WHITESPACE
            plt.close(fig)

            g1 = base64.b64encode(graph1.getvalue())
            g2 = base64.b64encode(graph2.getvalue())
            test_html=("""
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            * {
              box-sizing: border-box;
            }
            /* Create three equal columns that floats next to each other */
            .column {
              float: left;
              width: 33.33%;
              padding: 10px;
              height: 300px; /* Should be removed. Only for demonstration */
            }

            /* Clear floats after the columns */
            .row:after {
              content: "";
              display: table;
              clear: both;
            }
            </style>
            </head>
            <body>
            <center><h2>Univariate Analysis for ##0##</h2></center>
            <div class="row">
                <div class="column" >
                <p class="aligncenter"><img src=\'data:image/png;base64,##1##\' align="left" border="0"></p>
                </div>
                <div class="column" >
                <p class="aligncenter"><img src=\'data:image/png;base64,##2##\' align="left" border="0"></p>
                </div>
                <div class="column" >
                <center><h4> Basic Stats</h4></center>
                <center>##3##</center>
                </div>
            </div>
            </body>
            </html>""".replace('##0##',var)
                      .replace('##1##',g1.decode("utf-8"))
                      .replace('##2##',g2.decode("utf-8"))
                      .replace('##3##',basic_stats.to_html()))
            display(HTML(test_html))

    def _single_var_info_value(self,var):
        null_value_func = lambda x,y: x+0.5 if (x==0)|(y==0) else x
        def woe_func(check):
            check['adj_events']     = check.apply(lambda x: null_value_func(x['events'],x['non_events']),axis=1)
            check['adj_non_events'] = check.apply(lambda x: null_value_func(x['non_events'],x['events']),axis=1)
            check['pct_events']     = check['adj_events']    /check['adj_events'].sum()
            check['pct_non_events'] = check['adj_non_events']/check['adj_non_events'].sum()
            check['log_g_b']        = np.log(check['pct_events']/check['pct_non_events'])
            check['g_b']            = check['pct_events']-check['pct_non_events']
            check['woe']            = check['g_b']*check['log_g_b']
            return(check['woe'].sum())
        if str(self.dframe[var].dtypes) not in numerical_dtypes+['object']:
            print("Non Model Data Type for :",var)
            return(0)
        elif str(self.dframe[var].dtypes)=='object':
            check = self.bivariate_analysis(var,cat_check=True)
            return(woe_func(check))
        else:
            check = self.bivariate_analysis(var)
            return(woe_func(check))

    def information_value(self,feats=[]):
        frame = pd.DataFrame()
        feats = self.all_feats if len(feats)==0 else feats
        for var in feats:
            if var==self.bad:
                print("Skipping for Dependant Variable")
                temp = pd.DataFrame()
            else:
                temp  = pd.DataFrame({'variable':var,'info_value':self._single_var_info_value(var)},index=[0])
                frame = pd.concat([frame,temp],axis=0)
        frame.sort_values(by=['info_value'],ascending=False,inplace=True)
        frame.reset_index(drop=True,inplace=True)
        return(frame)

    def model_evaluation_report(self,y_true,y_pred_dict):
        frame = pd.DataFrame()
        mod_dframe, mod_bad  = self.dframe,self.bad
        for model in y_pred_dict.keys():
            auc  = roc_auc_score(y_true, y_pred_dict[model])
            gini = 2*auc-1
            ##KS METRIC
            self.dframe,self.bad = pd.DataFrame({'DV':y_true,'PRED':y_pred_dict[model]}),'DV'
            check = self.bivariate_analysis('PRED')
            check['event_%']         = check['events']/check['events'].sum()
            check['non_event_%']     = check['non_events']/check['non_events'].sum()
            check['cum_event_%']     = check['event_%'].cumsum()
            check['cum_non_event_%'] = check['non_event_%'].cumsum()
            check['diff'] = check['cum_non_event_%']-check['cum_event_%']
            #print(check)
            frame = pd.concat([frame,
                    pd.DataFrame({'Model':model,
                                  'KS'   :np.round(check['diff'].max()*100,3),
                                  'AUC'  :auc,
                                  'GINI' :gini},index=[0])],axis=0)
        frame.sort_values(by=['KS'],ascending=False,inplace=True)
        frame.reset_index(inplace=True,drop=True)
        self.dframe,self.bad = mod_dframe, mod_bad
        del mod_dframe
        del mod_bad
        return(frame)
