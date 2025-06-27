import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
import argparse
from random import randrange
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import utils
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import linalg
from scipy.sparse.linalg import eigsh as ssl_eigsh
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.append("/Users/paolo/Documents/methods/NonLinCFA/")
from NonLinCFA import NonLinCFA

def standardize(x_train, x_val, x_test):
    trainScaler = StandardScaler()
    x_train_scaled = trainScaler.fit_transform(x_train)
    x_val_scaled = trainScaler.transform(x_val)
    x_test_scaled = trainScaler.transform(x_test)
    
    return x_train_scaled, x_val_scaled, x_test_scaled

def unfold_dataset(x, y, df, feature_list):
    df_unfolded = pd.DataFrame(df.date.unique(), columns=['date'])
    df_unfolded = df_unfolded.set_index('date', drop=True)
    for feature_name in feature_list:
        for i in x:
            for j in y:
                col = df.loc[(df.x==i) & (df.y==j),[feature_name,'date']].set_index('date',drop=True)
                df_unfolded = pd.concat([df_unfolded,col],axis=1)
                newname = 'mean_'+str(i)+'_'+str(j)
                df_unfolded = df_unfolded.rename({feature_name:newname},axis=1)

    return df_unfolded

def compute_r2(x_train, y_train, x_val, y_val):
    regr = LinearRegression().fit(x_train,y_train)
    y_pred = regr.predict(x_val)
    return r2_score(y_val, y_pred)

def prepare_target(colname,max_train='2013-11-22', max_val='2018-04-10', max_test='2022-06-24', path='/Users/paolo/Documents/OneDrive - Politecnico di Milano/droughts/csv_VHI/Emiliani2.csv'):
    target_df = pd.read_csv(path)
    target_df = target_df.rename({'Unnamed: 0':'date'},axis=1)
    target_df_train = target_df.loc[target_df['date']<=max_train,:].copy()
    target_df_val = target_df.loc[(target_df['date']>max_train) & (target_df['date']<=max_val),:].copy()
    target_df_test = target_df.loc[(target_df['date']>max_val) & (target_df['date']<=max_test),:].copy()
    
    target_df_train['mean_std'], target_df_val['mean_std'], target_df_test['mean_std'] = standardize(target_df_train['mean'].values.reshape(-1,1), target_df_val['mean'].values.reshape(-1,1), target_df_test['mean'].values.reshape(-1,1))
    target_df_trainVal = pd.concat([target_df_train,target_df_val],axis=0).reset_index(drop=True)
    print(f'target samples: {target_df_train}\n target shapes: {target_df_train.shape, target_df_val.shape, target_df_trainVal.shape, target_df_test.shape}')
    
    return target_df_train,target_df_val,target_df_test,target_df_trainVal

def prepare_features(path, colName, multiple=False, max_train='2013-11-22', max_val='2018-04-10', max_test='2022-12-31'):
    if isinstance(path, str):
        if multiple == True:
            import glob
            filenames = glob.glob(path + "/*.csv")
            df = []
            for file in filenames:
                df.append(pd.read_csv(file))
            df = pd.concat(df, ignore_index=True)
        else: df = pd.read_csv(path)
    else: df = path 

    df_train = df.loc[(df['date']>'1990-09-10') & (df['date']<=max_train),:]
    df_val = df.loc[(df['date']>max_train) & (df['date']<=max_val),:]
    df_test = df.loc[(df['date']>max_val) & (df['date']<=max_test),:]
    
    df_train_unfolded = unfold_dataset(df_train.x.unique(), df_train.y.unique(), df_train, [colName])
    df_train_unfolded = df_train_unfolded.loc[:,np.std(df_train_unfolded,axis=0)>0]
    
    df_val_unfolded = unfold_dataset(df_val.x.unique(), df_val.y.unique(), df_val, [colName])
    df_val_unfolded = df_val_unfolded.loc[:,np.std(df_val_unfolded,axis=0)>0]
    
    df_test_unfolded = unfold_dataset(df_test.x.unique(), df_test.y.unique(), df_test, [colName])
    df_test_unfolded = df_test_unfolded.loc[:,np.std(df_test_unfolded,axis=0)>0]
    
    df_train_unfolded_std, df_val_unfolded_std, df_test_unfolded_std = standardize(df_train_unfolded, df_val_unfolded, df_test_unfolded)
    df_train_unfolded_std = pd.DataFrame(data=df_train_unfolded_std, columns=df_train_unfolded.columns)
    df_val_unfolded_std = pd.DataFrame(data=df_val_unfolded_std, columns=df_val_unfolded.columns)
    df_test_unfolded_std = pd.DataFrame(data=df_test_unfolded_std, columns=df_test_unfolded.columns)
    df_trainVal_unfolded_std = pd.concat([df_train_unfolded_std,df_val_unfolded_std],axis=0).reset_index(drop=True)
    
    return df_train_unfolded_std, df_val_unfolded_std, df_test_unfolded_std,df_trainVal_unfolded_std

def aggregate_unfolded_data(path,colnames, target_df_trainVal, eps, multiple=False, max_train='2013-11-22', max_val='2018-04-10', max_test='2022-12-31', neigh=1):

    aggregate_trainVal = pd.DataFrame()
    aggregate_test = pd.DataFrame()

    for col in colnames:
        df_train_unfolded_std,df_val_unfolded_std,df_test_unfolded_std,df_trainVal_unfolded_std = prepare_features(path,col,multiple,max_train,max_val,max_test)
        df_trainVal_unfolded_std_withTar = pd.concat((df_trainVal_unfolded_std,target_df_trainVal['mean_std']), axis=1)
        print(f'Number of features: {df_train_unfolded_std.shape[1]}\n') 
        #df_trainVal_unfolded_std_withTar = df_trainVal_unfolded_std_withTar[np.random.default_rng(seed=42).permutation(df_trainVal_unfolded_std_withTar.columns.values)]
        output = NonLinCFA(df_trainVal_unfolded_std_withTar,'mean_std', eps, -5 , neigh).compute_clusters()

        for i in range(len(output)):
            aggregate_trainVal[col+'_'+str(i)] = df_trainVal_unfolded_std_withTar[output[i]].mean(axis=1)
            aggregate_trainVal = aggregate_trainVal.copy()
            aggregate_test[col+'_'+str(i)] = df_test_unfolded_std[output[i]].mean(axis=1)
            aggregate_test = aggregate_test.copy()
        print(f'Number of aggregated features: {len(output)}\n')
        
    return output,aggregate_trainVal,aggregate_test 

def aggregate_data_withoutUnfolding(df, target_df_trainVal, eps, multiple=False, max_train='2013-11-22', max_val='2018-04-10', max_test='2022-12-31', neigh=1):

    aggregate_trainVal = pd.DataFrame()
    aggregate_test = pd.DataFrame()

    df_train = df.loc[df['date']<=max_train,:]
    df_val = df.loc[(df['date']>max_train) & (df['date']<=max_val),:]
    df_test = df.loc[(df['date']>max_val) & (df['date']<=max_test),:]

    df_train_std, df_val_std, df_test_std = standardize(df_train.iloc[:,1:], df_val.iloc[:,1:], df_test.iloc[:,1:])
    df_train_std = pd.DataFrame(data=df_train_std, columns=df_train.columns[1:])
    df_val_std = pd.DataFrame(data=df_val_std, columns=df_val.columns[1:])
    df_test_std = pd.DataFrame(data=df_test_std, columns=df_test.columns[1:])
    df_trainVal_std = pd.concat([df_train_std,df_val_std],axis=0).reset_index(drop=True)

    df_trainVal_std_withTar = pd.concat((df_trainVal_std,target_df_trainVal['mean_std']), axis=1)
    #noise = np.random.normal(0, 0.01, df_trainVal_std_withTar.shape)
    df_trainVal_std_withTar = df_trainVal_std_withTar#+noise
    print(f'Number of features: {df_train_std.shape[1]}\n')   

    output = NonLinCFA(df_trainVal_std_withTar,'mean_std', eps, -5 , neigh).compute_clusters()

    for i in range(len(output)):
        aggregate_trainVal['snow_'+df.columns[1][-2:]+str(i)] = df_trainVal_std_withTar[output[i]].mean(axis=1)
        aggregate_trainVal = aggregate_trainVal.copy()
        aggregate_test['snow_'+df.columns[1][-2:]+str(i)] = df_test_std[output[i]].mean(axis=1)
        aggregate_test = aggregate_test.copy()
    print(f'Number of aggregated features: {len(output)}\n')
        
    return output,aggregate_trainVal,aggregate_test 
    
def aggregate_unfolded_data_onlyTrain(path,colnames, target_df_train, target_df_val, eps, multiple=False, max_train='2013-11-22', max_val='2018-04-10', max_test='2022-12-31', neigh=1):

    aggregate_train = pd.DataFrame()
    aggregate_val = pd.DataFrame()
    aggregate_test = pd.DataFrame()

    for col in colnames:
        df_train_unfolded_std,df_val_unfolded_std,df_test_unfolded_std,df_trainVal_unfolded_std = prepare_features(path,col,multiple,max_train,max_val,max_test)
        df_train_unfolded_std_withTar = pd.concat((df_train_unfolded_std,target_df_train['mean_std']), axis=1)
        print(f'Number of features: {df_train_unfolded_std.shape[1]}\n')            
        output = NonLinCFA(df_train_unfolded_std_withTar,'mean_std', eps, -5 , neigh).compute_clusters() ### only train for aggregating

        for i in range(len(output)):
            aggregate_train[col+'_'+str(i)] = df_train_unfolded_std_withTar[output[i]].mean(axis=1)
            aggregate_train = aggregate_train.copy()
            aggregate_val[col+'_'+str(i)] = df_val_unfolded_std[output[i]].mean(axis=1)
            aggregate_val = aggregate_val.copy()
            aggregate_test[col+'_'+str(i)] = df_test_unfolded_std[output[i]].mean(axis=1)
            aggregate_test = aggregate_test.copy()
        print(f'Number of aggregated features: {len(output)}\n')
        
    return output,aggregate_train,aggregate_val,aggregate_test 
    

def FS_with_linearWrapper(aggregate_trainVal, target_df_train, target_df_val, max_feat, val_len=200):
    aggregate_train = aggregate_trainVal.iloc[:-val_len]
    aggregate_val = aggregate_trainVal.iloc[-val_len:]  

    scores = []
    selected_cols = []
    best_selected_cols = []
    best_score = -100

    #for i in range(400):
    actual_best = aggregate_train.columns[0]
    regr = LinearRegression()
    actual_score = regr.fit(aggregate_train[actual_best].values.reshape(-1, 1),target_df_train['mean_std']).score(aggregate_val[actual_best].values.reshape(-1, 1),target_df_val['mean_std'])
    for col in aggregate_train.columns:
        regr = LinearRegression()
        score = regr.fit(aggregate_train[col].values.reshape(-1, 1),target_df_train['mean_std']).score(aggregate_val[col].values.reshape(-1, 1),target_df_val['mean_std'])
        if score > actual_score: 
            actual_score = score
            actual_best = col
            
    selected_cols = [actual_best]
    scores = [actual_score]

    all_cols = aggregate_train.columns[aggregate_train.columns!=actual_best]

    for i in range(max_feat):
        actual_score = -10000
        for col in all_cols:
            regr = LinearRegression()
            score = regr.fit(aggregate_train[[col]+selected_cols],target_df_train['mean_std']).score(aggregate_val[[col]+selected_cols],target_df_val['mean_std'])
            if score > actual_score: 
                actual_score = score
                actual_best = col
        selected_cols.append(actual_best)
        scores.append(actual_score)
        all_cols = all_cols[all_cols!=actual_best]
        if actual_score>best_score:
            best_score = actual_score
            best_selected_cols = selected_cols.copy()
        actual_train_score = regr.fit(aggregate_train[selected_cols],target_df_train['mean_std']).score(aggregate_train[selected_cols],target_df_train['mean_std'])
        #print(f'actual training score: {actual_train_score}')
        #print(f'actual validation score: {actual_score}, number of remaining columns: {len(all_cols)}\n')
    print(f'\n\nselected columns: {best_selected_cols}, \n\nvalidation score: {best_score}, \n\nnumber of selected features: {len(best_selected_cols)}')
    return best_selected_cols
        
def compare_methods(aggregate_trainVal, aggregate_test, target_df_trainVal, target_df_test, selected_colnames):

    aggr_regr = LinearRegression()
    aggr_regr = aggr_regr.fit(aggregate_trainVal,target_df_trainVal['mean_std'])
    aggr_regr_score = aggr_regr.score(aggregate_test,target_df_test['mean_std'])
    aggr_regr_train_score = aggr_regr.score(aggregate_trainVal,target_df_trainVal['mean_std'])
    print(f'Full aggregate regression train score: {aggr_regr_train_score}, test score: {aggr_regr_score}')

    fs_aggr_regr = LinearRegression()
    fs_aggr_regr = fs_aggr_regr.fit(aggregate_trainVal[selected_colnames],target_df_trainVal['mean_std'])
    fs_aggr_regr_score = fs_aggr_regr.score(aggregate_test[selected_colnames],target_df_test['mean_std'])
    fs_aggr_regr_train_score = fs_aggr_regr.score(aggregate_trainVal[selected_colnames],target_df_trainVal['mean_std'])
    print(f'Aggregate regression train score with FS: {fs_aggr_regr_train_score}, test score: {fs_aggr_regr_score}')

    return fs_aggr_regr_score