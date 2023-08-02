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
import math

sys.path.append("/Users/paolo/Documents/methods/NonLinCFA/")
from NonLinCFA import NonLinCFA

def standardize(x_train, x_val, x_test):
    trainScaler = StandardScaler()
    x_train_scaled = trainScaler.fit_transform(x_train)
    x_val_scaled = trainScaler.transform(x_val)
    x_test_scaled = trainScaler.transform(x_test)
    
    return x_train_scaled, x_val_scaled, x_test_scaled

def unfold_dataset(x, y, df, feature_list, cols_order = "bottom_left"):
    df_unfolded = pd.DataFrame(df.date.unique(), columns=['date'])
    df_unfolded = df_unfolded.set_index('date', drop=True)
    if cols_order == "bottom_right":
        y = sorted(y)
        x = sorted(x, reverse = True)
    elif cols_order == "top_left":
        y = sorted(y, reverse = True)
        x = sorted(x)
    elif cols_order == "top_right":
        y = sorted(y, reverse = True)
        x = sorted(x, reverse = True)
    else:
        y = sorted(y)
        x = sorted(x)
    
    for feature_name in feature_list:
        for j in y:
            for i in x:
                col = df.loc[(df.x==i) & (df.y==j),[feature_name,'date']].set_index('date',drop=True)
                df_unfolded = pd.concat([df_unfolded,col],axis=1)
                newname = 'mean_'+str(i)+'_'+str(j)
                df_unfolded = df_unfolded.rename({feature_name:newname},axis=1)
    return df_unfolded

def compute_r2(x_train, y_train, x_val, y_val):
    regr = LinearRegression().fit(x_train,y_train)
    y_pred = regr.predict(x_val)
    return r2_score(y_val, y_pred)
    
def prepare_target(colname,max_train='2013-11-22', max_val='2018-04-10', max_test='2022-06-24', path='/Users/paolo/Documents/OneDrive - Politecnico di Milano/droughts/csv_VHI/Emiliani2.csv', 
                   window_size = 1, no_winter = False, only_winter = False):
    target_df = pd.read_csv(path)
    target_df = target_df.rename({'Unnamed: 0':'date'},axis=1)
    
    # min_periods is the number of not NaN needed to compute the mean (if != 0 there can be NaN values in the result)
    target_df['mean'] = target_df['mean'].rolling(window = window_size, min_periods = 0).mean()
    
    target_df_train = target_df.loc[target_df['date']<=max_train,:].copy()
    target_df_val = target_df.loc[(target_df['date']>max_train) & (target_df['date']<=max_val),:].copy()
    target_df_test = target_df.loc[(target_df['date']>max_val) & (target_df['date']<=max_test),:].copy()
    
    if (no_winter):
        target_df_test = target_df_test.loc[(target_df_test['week'] > 9) & (target_df_test['week'] < 45)]
        target_df_train = target_df_train.loc[(target_df_train['week'] > 9) & (target_df_train['week'] < 45)]
        target_df_val = target_df_val.loc[(target_df_val['week'] > 9) & (target_df_val['week'] < 45)]

        target_df_train.reset_index(inplace = True, drop = True)
        target_df_val.index = pd.RangeIndex(start=len(target_df_train),
                                                 stop=len(target_df_train)+len(target_df_val))
        train_val_len = len(target_df_train)+len(target_df_val)
        target_df_test.index = pd.RangeIndex(start=train_val_len, stop=train_val_len+len(target_df_test))

    if (only_winter):
        target_df_test = target_df_test.loc[(target_df_test['week'] <= 9) | (target_df_test['week'] >= 45)]
        target_df_train = target_df_train.loc[(target_df_train['week'] <= 9) | (target_df_train['week'] >= 45)]
        target_df_val = target_df_val.loc[(target_df_val['week'] <= 9) | (target_df_val['week'] >= 45)]

        target_df_train.reset_index(inplace = True, drop = True)
        target_df_val.index = pd.RangeIndex(start=len(target_df_train),
                                                 stop=len(target_df_train)+len(target_df_val))
        train_val_len = len(target_df_train)+len(target_df_val)
        target_df_test.index = pd.RangeIndex(start=train_val_len, stop=train_val_len+len(target_df_test))

    target_df_train['mean_std'], target_df_val['mean_std'], target_df_test['mean_std'] = standardize(target_df_train['mean'].values.reshape(-1,1), target_df_val['mean'].values.reshape(-1,1), target_df_test['mean'].values.reshape(-1,1))
    target_df_trainVal = pd.concat([target_df_train,target_df_val],axis=0).reset_index(drop=True)

    #print(f'target samples: {target_df_train}\n target shapes: {target_df_train.shape, target_df_val.shape, target_df_trainVal.shape, target_df_test.shape}')
    
    return target_df_train,target_df_val,target_df_test,target_df_trainVal

def prepare_features(path, colName, multiple=False, max_train='2013-11-22', max_val='2018-04-10', max_test='2022-12-31', cols_order = "bottom_left",
                     no_winter = False, only_winter = False):
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

    df_train = df.loc[(df['date']>'2001-01-01') & (df['date']<=max_train),:]
    df_val = df.loc[(df['date']>max_train) & (df['date']<=max_val),:]
    df_test = df.loc[(df['date']>max_val) & (df['date']<=max_test),:]

    if no_winter:
        df_train = df_train.loc[(df_train['week'] > 9) & (df_train['week'] < 45)]
        df_val = df_val.loc[(df_val['week'] > 9) & (df_val['week'] < 45)]
        df_test = df_test.loc[(df_test['week'] > 9) & (df_test['week'] < 45)]
    
    if only_winter:
        df_train = df_train.loc[(df_train['week'] <= 9) | (df_train['week'] >= 45)]
        df_val = df_val.loc[(df_val['week'] <= 9) | (df_val['week'] >= 45)]
        df_test = df_test.loc[(df_test['week'] <= 9) | (df_test['week'] >= 45)]        
    
    df_train_unfolded = unfold_dataset(df_train.x.unique(), df_train.y.unique(), df_train, [colName], cols_order = cols_order)
    df_train_unfolded = df_train_unfolded.loc[:,np.std(df_train_unfolded,axis=0)>0]
    
    df_val_unfolded = unfold_dataset(df_val.x.unique(), df_val.y.unique(), df_val, [colName], cols_order = cols_order)
    df_val_unfolded = df_val_unfolded.loc[:,np.std(df_val_unfolded,axis=0)>0]
    
    df_test_unfolded = unfold_dataset(df_test.x.unique(), df_test.y.unique(), df_test, [colName], cols_order = cols_order)
    df_test_unfolded = df_test_unfolded.loc[:,np.std(df_test_unfolded,axis=0)>0]
    
    df_train_unfolded_std, df_val_unfolded_std, df_test_unfolded_std = standardize(df_train_unfolded, df_val_unfolded, df_test_unfolded)
    df_train_unfolded_std = pd.DataFrame(data=df_train_unfolded_std, columns=df_train_unfolded.columns)
    df_val_unfolded_std = pd.DataFrame(data=df_val_unfolded_std, columns=df_val_unfolded.columns)
    df_test_unfolded_std = pd.DataFrame(data=df_test_unfolded_std, columns=df_test_unfolded.columns)
    df_trainVal_unfolded_std = pd.concat([df_train_unfolded_std,df_val_unfolded_std],axis=0).reset_index(drop=True)
    
    return df_train_unfolded_std, df_val_unfolded_std, df_test_unfolded_std,df_trainVal_unfolded_std

def remove_tiny_aggreg_exclude_lowcorr(output, aggregate_trainVal, aggregate_test, col, eps, df_trainVal_unfolded_std, df_test_unfolded_std, df_trainVal_unfolded_std_withTar, min_num_coord = 2):
    print("Working on ", col)
    print("REMOVING SMALL AGGREGATIONS OF 1 OR 2 POINTS...............")
    min_num_coord = 2
    for num_elems in range(1, min_num_coord+1):
        if num_elems == 1:
            print("Let's start removing aggregations of one single point")
        else:
            print("Let's remove also aggregations of 2 points")
        # iterate for all the aggregations
        i = 0
        while i < len(output):
            # looking for small ones
            if (len(output[i]) == num_elems):
                print("One small aggregation found! ")
                 # take all the other coordinates
                cols = [x for x in df_trainVal_unfolded_std.columns if x not in output[i][0]]
                # find the neighbors of the unary aggregation, created a sample NonLinCFA
                neigh_names = NonLinCFA(df_trainVal_unfolded_std,'mean_std', eps, -5 , neigh=1).find_neighbors(output[i], cols)
                print("Found ", str(len(neigh_names)), "neighbours!")
                max_corr = 0.0
                aggregating = False
                # iterate for all the other aggregations
                for j in [x for x in range(len(output)) if x != i]:
                    # consider all aggregations containing at least a neighbor
                    if any(x in output[j] for x in neigh_names):
                        # take correlation between small point and the adjacent aggregation
                        corr = aggregate_trainVal[col+'_'+str(i)].corr(aggregate_trainVal[col+'_'+str(j)])
                        print("Found an adjacent aggregation! The correlation with the small one is of ", str(corr))
                        # take it only if it is the higher correlation found
                        if corr < 0.85:                                        
                            print("WARNING: correlation between 2 aggregation is below 0.85")
                        else:
                            if corr > max_corr:
                                print("It is the highest corr found till now, we will merge eventually aggreg ", str(i), " with ", str(j))
                                max_corr = corr
                                new_aggreg_id = j
                                aggregating = True
                
                if (aggregating):        
                    print("Merging aggreg ", str(i), " with ", str(new_aggreg_id))            
                    # remove the small aggreg from the output set and add it to the new adjacent aggregation    
                    print("Dropped column ", col, str(i))
                    aggregate_trainVal.drop(col+ '_' + str(i), inplace = True, axis = 1)
                    aggregate_test.drop(col+ '_' + str(i), inplace = True, axis = 1)    
                    new_columns = []
                    curr_feature_columns = [x for x in aggregate_trainVal.columns if x.rpartition("_")[0] == col]
                     # for all the columns, rename the ones of the feature col after the deleted one, to have coherent ids
                    for id, old_col in enumerate(aggregate_trainVal.columns):
                        name, _, digit = old_col.rpartition("_")
                        if name == col: 
                            if (int(digit) > i):
                                new_col = name + "_" + str(int(digit)-1)
                                print("Name changed")
                                new_columns.append(new_col)
                                print(new_col)
                            else:
                                new_columns.append(old_col)
                                print(old_col)                                
                        else:
                            new_columns.append(old_col)
                            print(old_col)            
                    aggregate_trainVal.columns = new_columns
                    aggregate_test.columns = new_columns        
                    old_aggreg = output.pop(i)
                    if (i < new_aggreg_id):
                        new_aggreg_id = new_aggreg_id - 1
                                   
                    output[new_aggreg_id].extend(old_aggreg)
                    print("Updating ", col,"_",str(new_aggreg_id))
                    print("The new length of output is ", len(output), " the last aggreg considered was the ", str(i+1))
                    # update features dataframes
                    aggregate_trainVal[col+'_'+str(new_aggreg_id)] = df_trainVal_unfolded_std_withTar[output[new_aggreg_id]].mean(axis=1)
                    aggregate_trainVal = aggregate_trainVal.copy()
                    aggregate_test[col+'_'+str(new_aggreg_id)] = df_test_unfolded_std[output[new_aggreg_id]].mean(axis=1)        
                    aggregate_test = aggregate_test.copy()
                else:
                    i += 1
                    print("No aggregation done.")
                print("The output at this point is: ")
                for p in range(len(output)):
                    print(output[p])
            else:
                i +=1
                print("No aggregation done.")
    return output, aggregate_trainVal, aggregate_test

def remove_tiny_aggreg(output, aggregate_trainVal, aggregate_test, col, eps, df_trainVal_unfolded_std, df_test_unfolded_std, df_trainVal_unfolded_std_withTar, min_num_coord = 2):
    print("Working on ", col)
    print("REMOVING SMALL AGGREGATIONS OF 1 OR 2 POINTS...............")
    min_num_coord = 2
    for num_elems in range(1, min_num_coord+1):
        if num_elems == 1:
            print("Let's start removing aggregations of one single point")
        else:
            print("Let's remove also aggregations of 2 points")
        # iterate for all the aggregations
        i = 0
        while i < len(output):
            print("New aggregation read: ")
            print(output[i])
            # looking for small ones
            if (len(output[i]) == num_elems):
                print("One small aggregation found! ", output[i][0])
                 # take all the other coordinates
                cols = [x for x in df_trainVal_unfolded_std.columns if x not in output[i][0]]
                # find the neighbors of the unary aggregation, created a sample NonLinCFA
                neigh_names = NonLinCFA(df_trainVal_unfolded_std,'mean_std', eps, -5 , neigh=1).find_neighbors(output[i], cols, debug = True)
                print("Found ", str(len(neigh_names)), "neighbours!")
                max_corr = 0.0
                # iterate for all the other aggregations
                for j in [x for x in range(len(output)) if x != i]:
                    # consider all aggregations containing at least a neighbor
                    if any(x in output[j] for x in neigh_names):
                        # take correlation between small point and the adjacent aggregation
                        corr = aggregate_trainVal[col+'_'+str(i)].corr(aggregate_trainVal[col+'_'+str(j)])
                        print("Found an adjacent aggregation! The correlation with the small one is of ", str(corr))
                        # take it only if it is the higher correlation found
                        if (corr < 0.85):                                        
                            print("WARNING: correlation between 2 aggregation is below 0.85")
                        if corr > max_corr:
                            print("It is the highest corr found till now, we will merge eventually aggreg ", str(i), " with ", str(j))
                            max_corr = corr
                            new_aggreg_id = j
                        
                print("Merging aggreg ", str(i), " with ", str(new_aggreg_id))            
                # remove the small aggreg from the output set and add it to the new adjacent aggregation    
                print("Dropped column ", col, str(i))
                aggregate_trainVal.drop(col+ '_' + str(i), inplace = True, axis = 1)
                aggregate_test.drop(col+ '_' + str(i), inplace = True, axis = 1)    
                new_columns = []
                print("New columns names: ")
                curr_feature_columns = [x for x in aggregate_trainVal.columns if x.rpartition("_")[0] == col]
                print(curr_feature_columns)
                 # for all the columns, rename the ones of the feature col after the deleted one, to have coherent ids
                for id, old_col in enumerate(aggregate_trainVal.columns):
                    name, _, digit = old_col.rpartition("_")
                    if name == col: 
                        if (int(digit) > i):
                            new_col = name + "_" + str(int(digit)-1)
                            print("Name changed")
                            new_columns.append(new_col)
                            print(new_col)
                        else:
                            new_columns.append(old_col)
                            print(old_col)                                
                    else:
                        new_columns.append(old_col)
                        print(old_col)
                            
                aggregate_trainVal.columns = new_columns
                aggregate_test.columns = new_columns
                            
                old_aggreg = output.pop(i)
                if (i < new_aggreg_id):
                    new_aggreg_id = new_aggreg_id - 1
                                   
                output[new_aggreg_id].extend(old_aggreg)
                print("Updating ", col,"_",str(new_aggreg_id))
                print("The new length of output is ", len(output), " the last aggreg considered was the ", str(i+1))
                for p in range(len(output)):
                    print(output[p])
                # update features dataframes
                aggregate_trainVal[col+'_'+str(new_aggreg_id)] = df_trainVal_unfolded_std_withTar[output[new_aggreg_id]].mean(axis=1)
                aggregate_trainVal = aggregate_trainVal.copy()
                aggregate_test[col+'_'+str(new_aggreg_id)] = df_test_unfolded_std[output[new_aggreg_id]].mean(axis=1)        
                aggregate_test = aggregate_test.copy()
            else:
                i += 1
    return output, aggregate_trainVal, aggregate_test

def aggregate_unfolded_data(path, colnames, target_df_trainVal, eps, multiple=False, max_train='2013-11-22', 
                            max_val='2018-04-10', max_test='2022-12-31', neigh=1, scale = 0.1, curr_seed = 42, 
                            shuffle = False, no_tiny_aggregations = False, exclude_lowcorr=False, cols_order = "bottom_left", 
                            adaptive_eps = False, aggreg_small_coord_groups = True, no_winter = False, only_winter = False):

    aggregate_trainVal = pd.DataFrame()
    aggregate_test = pd.DataFrame()
    outputs = []

    for col in colnames:
        df_train_unfolded_std,df_val_unfolded_std,df_test_unfolded_std,df_trainVal_unfolded_std = prepare_features(path,col,multiple,max_train,max_val,max_test, 
                                                                                                                   cols_order = cols_order, no_winter = no_winter, only_winter = only_winter)
        
        if shuffle:
            # shuffle columns, useless with internal ordering 
            df_trainVal_unfolded_std = df_trainVal_unfolded_std[np.random.default_rng(seed=curr_seed).permutation(df_trainVal_unfolded_std.columns.values)] 
            df_test_unfolded_std = df_test_unfolded_std[np.random.default_rng(seed=curr_seed).permutation(df_test_unfolded_std.columns.values)]
            
        df_trainVal_unfolded_std_withTar = pd.concat((df_trainVal_unfolded_std,target_df_trainVal['mean_std']), axis=1)
        
        #starting_point = [df_trainVal_unfolded_std.columns[0].split('_')[1], df_trainVal_unfolded_std.columns[0].split('_')[2]]
        
        print(f'Number of features: {df_train_unfolded_std.shape[1]}\n')
         
        # output = NonLinCFA(df_trainVal_unfolded_std_withTar,'mean_std', eps, -5 , neigh).compute_clusters(shuffle_starting_point_only = shuffle_starting_point_only, random_seed = curr_seed)
        # dyn_eps = eps/(df_train_unfolded_std.shape[1]**2)
        # dyn_eps = eps/(np.sqrt(df_train_unfolded_std.shape[1]))
        
        dyn_eps = eps/df_train_unfolded_std.shape[1]
        print("eps value: ", dyn_eps)
        
        if (len(df_trainVal_unfolded_std_withTar.columns) > 10) or aggreg_small_coord_groups:
            output = NonLinCFA(df_trainVal_unfolded_std_withTar,'mean_std', dyn_eps, -5 , neigh, scale = scale).compute_clusters(adaptive_eps)
        else:
            output = [[col] for col in df_trainVal_unfolded_std.columns]
        #i = 0
        #while ((len(output) < min_aggreg_len) or (len(output) > max_aggreg_len)) and (i < 20):
        #    if len(output) < min_aggreg_len:
        #        print("Only ", str(len(output)), " aggreg found. Making eps smaller.")
        #        dyn_eps = dyn_eps/2
        #        print("New eps value: ", str(dyn_eps))
        #        output = NonLinCFA(df_trainVal_unfolded_std_withTar,'mean_std', dyn_eps, -5 , neigh).compute_clusters(adaptive_eps)
        #    elif len(output) > max_aggreg_len:
        #        print(str(len(output)), " aggreg found. Making eps bigger.")
        #         dyn_eps = dyn_eps*2
        #        output = NonLinCFA(df_trainVal_unfolded_std_withTar,'mean_std', dyn_eps, -5 , neigh).compute_clusters(adaptive_eps)
        #    i+=1


        for i in range(len(output)):
            aggregate_trainVal[col+'_'+str(i)] = df_trainVal_unfolded_std_withTar[output[i]].mean(axis=1)
            aggregate_trainVal = aggregate_trainVal.copy()
            aggregate_test[col+'_'+str(i)] = df_test_unfolded_std[output[i]].mean(axis=1)
            aggregate_test = aggregate_test.copy()
        print(f'Number of aggregated features: {len(output)}\n')
        
        if no_tiny_aggregations & exclude_lowcorr:
            output, aggregate_trainVal, aggregate_test = remove_tiny_aggreg_exclude_lowcorr(output, aggregate_trainVal, aggregate_test, col, eps, df_trainVal_unfolded_std, df_test_unfolded_std, df_trainVal_unfolded_std_withTar, min_num_coord = 2)
        elif no_tiny_aggregations:
            output, aggregate_trainVal, aggregate_test = remove_tiny_aggreg(output, aggregate_trainVal, aggregate_test, col, eps, df_trainVal_unfolded_std, df_test_unfolded_std, df_trainVal_unfolded_std_withTar, min_num_coord = 2)
            
        outputs.append(output)
    return outputs,aggregate_trainVal,aggregate_test 

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