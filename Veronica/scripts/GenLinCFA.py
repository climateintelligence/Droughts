import argparse
import glob
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
from sklearn import preprocessing
from sklearn.utils import check_random_state
from scipy.stats import pearsonr
import multiprocessing
from joblib import Parallel, delayed
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from scipy.stats import pearsonr
import csv 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import pickle

class GenLinCFA():
    """
        Class which takes as input a dataframe (or path), the column name of the target, the allowed error and the number of cross-validation batches
        The method compute_clusters prints and returns the list of aggregations
    """
    def __init__(self, df, target_name, eps1, n_val, neigh,  eps2, scale=0.1):
        if type(df)==str:
            pd.read_csv(df)
        else: self.df = df.copy(deep=True)
        self.target_name = target_name
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_val = n_val
        self.clusters = []
        self.scale = scale
        self.neigh = neigh
        self.coord_corr_dict = {}

    def print_header(self):
        print("Dataset: \n{}".format(self.df))

    def compute_corr(self, column1, column2):
        return pearsonr(self.df[column1],self.df[column2])[0]
    
    def prepare_data(self, x1, x2, l):
        x1 = preprocessing.scale(x1, with_mean=True)
        x2 = preprocessing.scale(x2, with_mean=True)
        x = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=1)
        x_aggr = ((x1*l+x2)/(l+1)).reshape(-1,1) 
        x_aggr = preprocessing.scale(x_aggr, with_mean=True)
        y = self.df[self.target_name].values
        y = preprocessing.scale(y, with_mean=True, with_std=True) 
        return x_aggr, x, y

    def find_neighbors(self, actual_clust, cols): 
        neighs = []
        for datum in actual_clust:
            x = float(datum.split('_')[1])
            y = float(datum.split('_')[2])
            for c in cols:
                cx = float(c.split('_')[1])
                cy = float(c.split('_')[2])
                if ((abs(x-cx)<self.scale) & (abs(y-cy)<self.scale)): neighs.append(c) # for droughts self.scale=0.1
        return neighs

    def compute_scores(self, column1_list, column2):
        x_aggr,x,y = self.prepare_data(self.df[column1_list].mean(axis=1).values,self.df[column2].mean(axis=1).values, len(column1_list))
        cov_x1_y = np.cov(x[:,0],y)
        cov_x2_y = np.cov(x[:,1],y)
        cov_xmean_y = np.cov(x_aggr.reshape(1,-1),y.reshape(1,-1))
        var_x1x2 = np.var(x[:,0]+x[:,1])
        var_xmean = np.var(x_aggr.reshape(1,-1))
        var_y = np.var(y.reshape(1,-1))
        return cov_x1_y[0,1],cov_x2_y[0,1],cov_xmean_y[0,1],var_x1x2,var_xmean,var_y
    
    # aggreg_mean: already computed mean of the points in the new aggregation 
    def refresh_corr_values(self, name_elem1, name_elem2, aggreg_mean, i, aggregations_list):
        
        for key in list(self.coord_corr_dict.keys()):
            x1, y1, x2, y2 = key.split("_")

            if (x1 + "_" + y1) in (name_elem1, name_elem2):
                # if the other element is an aggregation, I have to take the mean of all its coordinates
                if x2.startswith("aggreg"):
                    aggreg_index = int(y2)
                    other_coord = aggregations_list[aggreg_index]
                    other_coord_mean = self.df[other_coord].mean(axis=1)
                else:
                    other_coord = "mean_" + x2 + "_" + y2
                    other_coord_mean = self.df[other_coord]
                    
                correl = np.corrcoef(other_coord_mean, aggreg_mean)[0, 1]
                new_key = "aggreg_" + str(i) + "_" + x2 + "_" + y2
                inverse_key = x2 + "_" + y2 + "_" + "aggreg_" + str(i)
                # !!! the new key can be = old key if we are updating a key with an aggregation, in this case just update
                if (new_key == key):
                    self.coord_corr_dict[key] = correl
                else:
                    # checks also if the same aggreg has been considered at previous if (a common neighbor)
                    if inverse_key not in self.coord_corr_dict and new_key not in self.coord_corr_dict:
                        self.coord_corr_dict[new_key] = correl
                    del self.coord_corr_dict[key]
            
            elif (x2 + "_" + y2) in (name_elem1, name_elem2):
                if x1.startswith("aggreg"):
                    aggreg_index = int(y1)
                    other_coord = aggregations_list[aggreg_index]
                    other_coord_mean = self.df[other_coord].mean(axis=1)
                else:
                    other_coord = "mean_" + x1 + "_" + y1
                    other_coord_mean = self.df[other_coord]
                correl = np.corrcoef(other_coord_mean, aggreg_mean)[0, 1]
                new_key =  x1 + "_" + y1 + "_" + "aggreg_" + str(i)
                inverse_key = "aggreg_" + str(i) + "_" + x1 + "_" + y1
                
                if (new_key == key):
                    self.coord_corr_dict[key] = correl
                else:  
                    # checks also if the same aggreg has been considered at previous if (a common neighbor)
                    if inverse_key not in self.coord_corr_dict and new_key not in self.coord_corr_dict:
                        self.coord_corr_dict[new_key] = correl
                    del self.coord_corr_dict[key]
        return

    # once the dictionary has been refreshed, I have to change name to every aggreg > i to i-1 and to remove aggregations_list[i]
    def remove_aggregation(self, i, aggregations_list):
        
        for key in list(self.coord_corr_dict.keys()):
            modified_key = ""
            # !!! there could be more than one aggreg_i in the same key to be updated
            updated = False
            for aggreg_num in range(i+1, len(aggregations_list)): 
                
                if 'aggreg_' + str(aggreg_num) in key:
                    if (updated):
                        decomposed_key = modified_key.split('_')
                    else:
                        decomposed_key = key.split('_')
                    new_key = '_'.join([str(aggreg_num-1) if subkey == str(aggreg_num) else subkey for subkey in decomposed_key])
                    modified_key = new_key
                    updated = True       
            if (updated):
                #print("Removed key : ", key)        
                self.coord_corr_dict[new_key] = self.coord_corr_dict.pop(key)
                
        aggregations_list = aggregations_list[:i] + aggregations_list[i+1:]
        return aggregations_list

    def compute_clusters(self):

        aggreg_num = 0
        aggregations_list = []
        
        cols = self.df.loc[:, self.df.columns != self.target_name].columns # all the columns of the DF
        
        #print("Preparing the correlations dictionary")
        for point in cols:
            # find neighbors of point among all the other points exluding itself
            if len(self.df.columns) > 10:
                neighbors = self.find_neighbors([point], cols.drop(point))
            else:
                neighbors = cols.drop(point)
            point_coords = point.split('_')[1:] # to get the list of coordinates
            point_coords = "_".join(point_coords) # to get coordinates as x_y
            
            for neighbor in neighbors:
                neighbor_coords = neighbor.split('_')[1:] # to get the list of coordinates
                neighbor_coords = "_".join(neighbor_coords) # to get coordinates as x_y
                key = point_coords + "_" + neighbor_coords 
                inverse_key = neighbor_coords + "_" + point_coords
                # check if the pair is already present, otherwise save it
                if inverse_key not in self.coord_corr_dict:
                    self.coord_corr_dict[key] = np.corrcoef(self.df[point], self.df[neighbor])[0, 1]
        aggregated = True
        
        # if aggregated stays false for the whole for iterations, the aggregations are done
        while aggregated:
            # Create a list of keys sorted by the value in descending order
            sorted_keys = sorted(self.coord_corr_dict.keys(), key=self.coord_corr_dict.get, reverse=True)
            aggregated = False
            
            for max_corr in sorted_keys:
                # check for new aggregations if no one has already been done in this while iteration
                if not aggregated:
                    
                    x1, y1, x2, y2 = max_corr.split("_")
                    aggreg_coord = []
                    #print("Trying to aggregate ", x1, y1, "with", x2, y2)
                    
                    # if one of the 2 elements is an aggregation, add all its coordinates to the list "aggreg_coord"
                    if x1.startswith("aggreg"):
                        first_elem = aggregations_list[int(y1)]
                    else:
                        first_elem = ["mean_" + x1 + "_" + y1]
                        
                    if x2.startswith("aggreg"):
                        second_elem = aggregations_list[int(y2)]
                    else:
                        second_elem = ["mean_" + x2 + "_" + y2] 
                        
                    aggreg_coord.extend(first_elem)
                    aggreg_coord.extend(second_elem)

                    cov_x1_y,cov_x2_y,cov_xmean_y,var_x1x2,var_xmean,var_y = self.compute_scores(first_elem, second_elem)

                    if ((abs(cov_x1_y)+abs(cov_x2_y)-self.eps1*0.5*var_x1x2)<=0):

                        aggreg_mean = np.mean(self.df[aggreg_coord].values, axis=1) # mean along rows of elements in aggreg_coord

                        if x1.startswith("aggreg") and not x2.startswith("aggreg"):
                            
                            agg_index = int(y1)
                            aggregations_list[agg_index].append("mean_" + x2 + "_" + y2)
                            del self.coord_corr_dict[max_corr]
                            self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, aggreg_mean, agg_index, aggregations_list)
                            
                        elif not x1.startswith("aggreg") and x2.startswith("aggreg"):

                            agg_index = int(y2)
                            aggregations_list[agg_index].append("mean_" + x1 + "_" + y1)
                            del self.coord_corr_dict[max_corr]
                            self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, aggreg_mean, agg_index, aggregations_list)

                        elif not x1.startswith("aggreg") and not x2.startswith("aggreg"):
                            
                            #print("Adding the two points to the aggregations_list in position ", aggreg_num)
                            aggregations_list.append(["mean_" + x1 + "_" + y1, "mean_" + x2 + "_" + y2])

                            del self.coord_corr_dict[max_corr]
                            self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, aggreg_mean, aggreg_num, aggregations_list)
                            # update the number of aggregations
                            aggreg_num += 1
                            
                        else: # case both are aggregations
                            if (y1 < y2):
                                agg_index = int(y1)
                                to_be_removed = int(y2)
                            else: 
                                agg_index = int(y2)
                                to_be_removed = int(y1)
                                
                            aggregations_list[agg_index].extend(aggregations_list[to_be_removed])
                            del self.coord_corr_dict[max_corr]
                            self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, aggreg_mean, agg_index, aggregations_list)
                            aggregations_list = self.remove_aggregation(to_be_removed, aggregations_list)
                            aggreg_num -= 1
                        #print("\n")
                        aggregated = True

        # add the remaining isolated points to the returned list of aggregations
        for key in self.coord_corr_dict:
            x1, y1, x2, y2 = key.split("_")
            elem1 = ["mean_" + x1 + "_" + y1]
            elem2 = ["mean_" + x2 + "_" + y2]

            if not x1.startswith("aggreg") and elem1 not in aggregations_list:
                aggregations_list.append(elem1)
            if not x2.startswith("aggreg") and elem2 not in aggregations_list:
                aggregations_list.append(elem2)    
        
        #print("Final dictionary:")
        #for key in self.coord_corr_dict:
        #    value = self.coord_corr_dict[key]
        #    print(f"{key}: {value}")
        return aggregations_list
    