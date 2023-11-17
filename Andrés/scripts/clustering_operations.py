import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

def get_corr_mask(arr1, arr2):
    arr1_mask = ~np.isnan(arr1)
    arr2_mask = ~np.isnan(arr2)

    mask = arr1_mask * arr2_mask
    corr_masked = np.corrcoef(np.array(arr1)[mask], np.array(arr2)[mask])[0, 1]
    return corr_masked

def plot_clusters_shp(shapefile, clusters, figsize=(8,8), only_stats=False):
    shp = shapefile.copy()

    clusters_no_singletons = remove_singletons(clusters)
    singletons = clusters[len(clusters_no_singletons):]
    
    print(f'No. clusters (w/o singletons) = {len(clusters_no_singletons)}')
    print(f'No. singletons = {len(singletons)}')
    if only_stats==False:    
        progress_bar = tqdm(total=shp.shape[0], position=0, leave=True, smoothing=0)

        clusterid = 0
        for cluster in clusters:
            for elem in cluster:
                shp.loc[(shp['SUBID'] == elem), 'CLUSTERID'] = clusterid
                        
                progress_bar.update(1)
            
            clusterid += 1

        unique_cluster_ids = shp['CLUSTERID'].unique()
        num_unique_ids = len(unique_cluster_ids)
        
        #seed_value = 45
        #np.random.seed(seed_value)
        random_colors = np.random.rand(num_unique_ids, 3)  # RGB values

        # Create a dictionary to map CLUSTERID to its corresponding random color
        clusterid_to_color = dict(zip(unique_cluster_ids, random_colors))


        # Map each CLUSTERID to its corresponding random color

        shp['color'] = shp['CLUSTERID'].apply(lambda x: clusterid_to_color[x])

        shp.plot(figsize=figsize, color=shp['color'])

def plot_hist_with_mean(arr, title="", xlabel='avg. correlation between original sub-basins and new sub-basin', ylabel='count'):
    arr = np.array(arr)[~np.isnan(arr)]
    mean_value = np.mean(arr)
    plt.hist(arr, bins=np.array(range(700, 1025, 25))/1000)
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label='Mean')

    plt.text(mean_value-0.01, plt.ylim()[1], f'Mean: {mean_value:.3f}', color='red', fontsize=12, ha='right', va='top')

    plt.title(title, fontsize = 16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()

def plot_cluster_stats(df, cluster_num, clusters, clusters_weighted_mean, min_inter_corr_subids, avg_corr_with_mean, avg_inter_corr, min_inter_corr):
    #print(f'clusters_weighted_mean: {clusters_weighted_mean[cluster_num]}'
    print(f'avg_corr_with_mean: {avg_corr_with_mean[cluster_num]:.3f}')
    print(f'avg_inter_corr: {avg_inter_corr[cluster_num]:.3f}')
    print(f'min_inter_corr: {min_inter_corr[cluster_num]:.3f}')

    plt.figure(figsize=(20,5))
    plt.plot(df[clusters[cluster_num]],color='C0')
    #cluster_no_mins = [elem for elem in clusters[cluster_num] if elem not in min_inter_corr_subids[cluster_num]]
    #plt.plot(df[cluster_no_mins],color='blue')
    #plt.plot(df[min_inter_corr_subids[cluster_num]],color='red', linewidth=2)
    plt.plot(clusters_weighted_mean[cluster_num],color='red', linewidth=3)

def correlations_sub_cluster(df, clusters, clusters_mean):
    avg_correlations = []
    progress_bar = tqdm(total=len(clusters), position=0, leave=True, smoothing=0)

    for i in range(len(clusters)):
        correlations = []
        for elem in clusters[i]:
            corr = get_corr_mask(df[elem], clusters_mean[i])
            correlations.append(corr)

        avg_correlations.append(np.mean(correlations))
        progress_bar.update(1)

    return avg_correlations   

def compute_clusters_mean(df, shp, clusters, weighted=True):
    clusters_mean = []
    progress_bar = tqdm(total=len(clusters), position=0, leave=True, smoothing=0)
    
    for cluster in clusters:
        df_temp = df[cluster].copy()
        if weighted:
            cluster_area = shp.loc[shp["SUBID"].isin(df_temp.columns), "AREA"].sum()
        for elem in cluster:
            if weighted:
                df_temp[elem] = df_temp[elem]*(shp.loc[shp["SUBID"] == elem, "AREA"].iloc[0])

        if weighted:
            clusters_mean.append(df_temp.sum(axis=1)/cluster_area)    
        else:
            clusters_mean.append(df_temp.sum(axis=1))    
        progress_bar.update(1)
    return clusters_mean

# It considers only clusters with at least 2 sub-basins
# Some clusters with two or more sub-basins never have two sub-basin FAPAN values at the same time.
def compute_avg_min_internal_corr(df, clusters):
    clusters_no_singletons = remove_singletons(clusters)
    
    avg_inter_corr = []
    min_inter_corr = []
    min_inter_corr_subids = []

    progress_bar = tqdm(total=len(clusters_no_singletons), position=0, leave=True, smoothing=0)
    for cluster in range(len(clusters_no_singletons)):
        min_correlation = 1
        avg_correlation = 0
        length = len(clusters[cluster])
        count = 0
        for i in range(length-1):
            for j in range(i+1, length):
                correlation = get_corr_mask(df[clusters[cluster][i]], df[clusters[cluster][j]])
                
                if not math.isnan(correlation):
                    count += 1
                    avg_correlation = avg_correlation + correlation
                    if correlation <= min_correlation:
                        min_correlation = correlation
                        subids = [clusters[cluster][i], clusters[cluster][j]]         

        if count > 0:
            avg_correlation = avg_correlation / count  
        
            avg_inter_corr.append(avg_correlation)
            min_inter_corr.append(min_correlation)
            min_inter_corr_subids.append(subids)

        progress_bar.update(1)

    return avg_inter_corr, min_inter_corr, min_inter_corr_subids


def remove_singletons(clusters_list):
    new_clusters_list = clusters_list.copy()

    # Find the index of the first sublist with only one element starting from the end
    first_single_element_index = None
    for i in range(len(new_clusters_list) - 1, -1, -1):
        if len(new_clusters_list[i]) > 1:
            first_single_element_index = i
            break

    # Remove all elements after the first sublist with only one element
    if first_single_element_index is not None:
        new_clusters_list = new_clusters_list[:first_single_element_index + 1]

    return new_clusters_list


# input clusters should be ordered from biggest to smallets
def get_clustering_correlation_analysis(df, clusters):
    clusters_no_singletons = remove_singletons(clusters)
    avg_total_correlation = 0
    for cluster in range(len(clusters_no_singletons)):
        min_correlation = 1
        avg_correlation = 0
        length = len(clusters[cluster])
        
        for i in range(length-1):
            for j in range(i+1, length):
                correlation = np.corrcoef(df[clusters[cluster][i]], df[clusters[cluster][j]])[0][1]
                avg_correlation = avg_correlation + correlation
                if correlation < min_correlation:
                    min_correlation = correlation
                    points = (clusters[cluster][i], clusters[cluster][j])
        avg_correlation = avg_correlation / sum(range(1, length))      

        print(f'Cluster {cluster}')
        print(f'Avg. correlation : {avg_correlation}')            
        print(f'Min. correlation : {min_correlation}, between points : {points}\n')        

        avg_total_correlation = avg_total_correlation + avg_correlation

    avg_total_correlation = avg_total_correlation/len(clusters_no_singletons)
    print(f'Avg. Total correlation : {avg_total_correlation}') 