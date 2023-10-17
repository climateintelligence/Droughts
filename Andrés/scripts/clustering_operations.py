import numpy as np
from tqdm import tqdm




def compute_neighbours(df, max_distance=1):
    neighbours = []
    coordinates = df.columns

    print("Computing neighbours...")
    progress_bar = tqdm(total=len(coordinates), position=0, leave=True, smoothing=0)

    for idx, coord1 in enumerate(coordinates):
        for idx2, coord2 in enumerate(coordinates[idx + 1:], start=idx + 1):
            x1, y1 = map(float, coord1.split('_'))
            x2, y2 = map(float, coord2.split('_'))
            distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            
            if distance < max_distance:
                neighbours.append((coord1, coord2))

        progress_bar.update(1) 

    print("neighbours computed")
    return neighbours

def compute_clusters(df, neighbours, without_neighbours, method, threshold, keep_singletons=True):
    
    neighbours_strength_dict = get_neighbours_strength_dict(df, neighbours, method)     

    cluster_num = 0
    clusters_list = []
    clustering_complete = True

    print("Computing clusters...")
    progress_bar = tqdm(total=df.shape[1], position=0, leave=True, smoothing=0)

    while clustering_complete:
        sorted_keys = sorted(neighbours_strength_dict.keys(), key=neighbours_strength_dict.get, reverse=True)
        clustering_complete = False

        for max_strength in sorted_keys:
            # Check for new clusters if none have been made in this iteration yet
            if clustering_complete :
                break
            
            elif enough_strength(neighbours_strength_dict[max_strength], method, threshold):
                x1, y1, x2, y2 = max_strength.split("_")
                
                # Update cluster and correlation values
                if x1 == "cluster" and not x2 == "cluster":
                    cluster_index = int(y1)
                    clusters_list[cluster_index].append(x2 + "_" + y2)
                    del neighbours_strength_dict[max_strength]
                    refresh_corr_values(df, neighbours_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_index, clusters_list, method)
                    
                elif not x1 == "cluster" and x2 == "cluster":
                    cluster_index = int(y2)
                    clusters_list[cluster_index].append(x1 + "_" + y1)
                    del neighbours_strength_dict[max_strength]
                    refresh_corr_values(df, neighbours_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_index, clusters_list, method)

                elif not x1 == "cluster" and not x2 == "cluster":
                    clusters_list.append([x1 + "_" + y1, x2 + "_" + y2])
                    del neighbours_strength_dict[max_strength]
                    refresh_corr_values(df, neighbours_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_num, clusters_list, method)
                    cluster_num += 1
                    
                else: # case both are clusters
                    cluster_index = min(int(y1), int(y2))
                    to_be_removed = max(int(y1), int(y2))
                    clusters_list[cluster_index].extend(clusters_list[to_be_removed])
                    del neighbours_strength_dict[max_strength]   
                    refresh_corr_values(df, neighbours_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_index, clusters_list, method)
                    clusters_list = remove_cluster(neighbours_strength_dict, to_be_removed, clusters_list)
                    cluster_num -= 1

                progress_bar.update(1)  
                clustering_complete = True

    if keep_singletons == True:
        # Add the remaining isolated points to the list of clusters
        for key in neighbours_strength_dict:
            x1, y1, x2, y2 = key.split("_")
            elem1 = [x1 + "_" + y1]
            elem2 = [x2 + "_" + y2]

            if not x1 == "cluster" and elem1 not in clusters_list:
                clusters_list.append(elem1)

            if not x2 == "cluster" and elem2 not in clusters_list:
                clusters_list.append(elem2)

            progress_bar.update(1)    

        for elem in without_neighbours:
            clusters_list.append([elem])
            progress_bar.update(1)     

    print("Clusters computed.")
    clusters_list = sorted(clusters_list, key=lambda x: len(x), reverse=True)
    return clusters_list

def get_neighbours_strength_dict(df, neighbours, method):

    print("Computing neighbours strengths...")
    progress_bar = tqdm(total=len(neighbours), position=0, leave=True, smoothing=0)

    neighbours_strength_dict = {}

    if method == 'correlation' or method == 'complete_correlation':
        for neighbor_pair in neighbours:
            neighbor1, neighbor2 = neighbor_pair

            strength = np.corrcoef(df[neighbor1], df[neighbor2])[0, 1]
            neighbours_strength_dict['_'.join(neighbor_pair)] = strength
            progress_bar.update(1)  

    elif method == 'distance':
        for neighbor_pair in neighbours:
            neighbor1, neighbor2 = neighbor_pair
            strength = np.linalg.norm(df[neighbor1] - df[neighbor2])
            neighbours_strength_dict['_'.join(neighbor_pair)] = strength   
            progress_bar.update(1)  
    
    print("Neighbours strengths computed.")
    return neighbours_strength_dict    

def get_neighbours_strength(df, cluster_index, cluster_mean, x2, y2, clusters_list, method):
    strength = ''

    if method == 'correlation':
        elem2 = df[clusters_list[int(y2)]].mean(axis=1) if x2 == "cluster" else df[x2 + "_" + y2]
        strength = np.corrcoef(cluster_mean, elem2)[0, 1]

    elif method == 'complete_correlation':
        strength = 1
        if x2 == "cluster":
            cluster1 = df[clusters_list[cluster_index]]
            cluster2 = df[clusters_list[int(y2)]]
            for elem1 in cluster1.columns:
                for elem2 in cluster2.columns:
                    strength_temp = np.corrcoef(df[elem1], df[elem2])[0, 1]
                    if strength_temp < strength:
                        strength = strength_temp

        else:
            cluster1 = df[clusters_list[cluster_index]]
            for elem1 in cluster1.columns:
                strength_temp = np.corrcoef(df[elem1], df[x2 + "_" + y2])[0, 1]
                if strength_temp < strength:
                    strength = strength_temp    

    elif method == 'distance':
        elem2 = df[clusters_list[int(y2)]].mean(axis=1) if x2 == "cluster" else df[x2 + "_" + y2]
        strength = np.linalg.norm(cluster_mean - elem2)

    else:
        raise ValueError("Unsupported method. Supported methods are 'correlation', 'complete_correlation and 'distance'.")

    return strength

def enough_strength(strength, method, threshold):
    if method == 'correlation' or method == 'complete_correlation':
        return strength >= threshold
    elif method == 'distance':
        return strength <= threshold
    else:
        raise ValueError("Unsupported method. Supported methods are 'correlation', 'complete_correlation' and 'distance'.")


def refresh_corr_values(df, neighbours_strength_dict, name_elem1, name_elem2, cluster_index, clusters_list, method):
    cluster_mean = df[clusters_list[cluster_index]].mean(axis=1)
    for key in list(neighbours_strength_dict.keys()):
        x1, y1, x2, y2 = key.split("_")

        if (x1 + "_" + y1) in (name_elem1, name_elem2):
                
            strength = get_neighbours_strength(df, cluster_index, cluster_mean, x2, y2, clusters_list, method)
            new_key = f'cluster_{cluster_index}_{x2}_{y2}'
            inverse_new_key = f'{x2}_{y2}_cluster_{cluster_index}'

            # Check if x1_y1 is already a cluster and doesn't change cluster_index
            if (new_key == key): 
                neighbours_strength_dict[key] = strength
            else:
                # Check if the same cluster has been considered previously (a common neighbor)
                if inverse_new_key not in neighbours_strength_dict and new_key not in neighbours_strength_dict:
                    neighbours_strength_dict[new_key] = strength
                del neighbours_strength_dict[key]
        
        elif (x2 + "_" + y2) in (name_elem1, name_elem2):

            strength = get_neighbours_strength(df, cluster_index, cluster_mean, x1, y1, clusters_list, method)
            new_key = f'{x1}_{y1}_cluster_{cluster_index}'
            inverse_new_key = f'cluster_{cluster_index}_{x1}_{y1}'
            
            if (new_key == key):
                neighbours_strength_dict[key] = strength
            else:  
                if inverse_new_key not in neighbours_strength_dict and new_key not in neighbours_strength_dict:
                    neighbours_strength_dict[new_key] = strength
                del neighbours_strength_dict[key]
    return

# once the dictionary has been refreshed, I have to change name to every cluster > i to i-1 and to remove clusters_list[i]
def remove_cluster(neighbours_strength_dict, to_be_removed, clusters_list):
    for key in list(neighbours_strength_dict.keys()):
        modified_key = ""
        # !!! there could be more than one cluster_i in the same key to be updated
        updated = False
        for cluster_num in range(to_be_removed+1, len(clusters_list)): 
            
            if f'cluster_{cluster_num}' in key:
                if updated:
                    decomposed_key = modified_key.split('_')
                else:
                    decomposed_key = key.split('_')

                new_key = '_'.join([str(cluster_num-1) if subkey == str(cluster_num) else subkey for subkey in decomposed_key])
                modified_key = new_key
                updated = True    

        if updated:
            neighbours_strength_dict[new_key] = neighbours_strength_dict.pop(key)
    
    new_clusters_list = clusters_list[:to_be_removed] + clusters_list[to_be_removed+1:]
    return new_clusters_list

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

"""
def compute_neighbours(df, max_distance=1):
    neighbours = []
    coordinates = df.columns

    print("Computing neighbours...")
    progress_bar = tqdm(total=len(coordinates), position=0, leave=True, smoothing=0)

    for idx, coord1 in enumerate(coordinates):
        for idx2, coord2 in enumerate(coordinates[idx + 1:], start=idx + 1):
            distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            if distance < max_distance:
                neighbours.append((coord1, coord2))

        progress_bar.update(1) 

    print("neighbours computed")
    return neighbours

def compute_clusters(df, neighbours, method, threshold, noise=False):
    neighbours_strength_dict = get_neighbours_strength_dict(df, neighbours, method)     
    cluster_num = 0
    clusters_list = []
    clustering_complete = True

    print("Computing clusters...")
    progress_bar = tqdm(total=df.shape[1], position=0, leave=True, smoothing=0)

    while clustering_complete:
        sorted_keys = sorted(neighbours_strength_dict.keys(), key=neighbours_strength_dict.get, reverse=True)
        clustering_complete = False

        for max_strength in sorted_keys:
            # Check for new clusters if none have been made in this iteration yet
            if clustering_complete :
                break
            
            elif enough_strength(neighbours_strength_dict[max_strength], method, threshold):
                (x1, y1), (x2, y2) = max_strength
                
                # Update cluster and correlation values
                if x1 == "cluster" and not x2 == "cluster":
                    cluster_index = int(y1)
                    clusters_list[cluster_index].append((x2, y2))
                    del neighbours_strength_dict[max_strength]
                    refresh_corr_values(df, neighbours_strength_dict, (x1, y1), (x2, y2), cluster_index, clusters_list, method)
                    
                elif not x1 == "cluster" and x2 == "cluster":
                    cluster_index = int(y2)
                    clusters_list[cluster_index].append((x1, y1))
                    del neighbours_strength_dict[max_strength]
                    refresh_corr_values(df, neighbours_strength_dict, (x1, y1), (x2, y2), cluster_index, clusters_list, method)

                elif not x1 == "cluster" and not x2 == "cluster":
                    clusters_list.append([(x1, y1), (x2, y2)])
                    del neighbours_strength_dict[max_strength]
                    refresh_corr_values(df, neighbours_strength_dict, (x1, y1), (x2, y2), cluster_num, clusters_list, method)
                    cluster_num += 1
                    
                else: # case both are clusters
                    cluster_index = min(int(y1), int(y2))
                    to_be_removed = max(int(y1), int(y2))
                    clusters_list[cluster_index].extend(clusters_list[to_be_removed])
                    del neighbours_strength_dict[max_strength]   
                    refresh_corr_values(df, neighbours_strength_dict, (x1, y1), (x2, y2), cluster_index, clusters_list, method)
                    clusters_list = remove_cluster(neighbours_strength_dict, to_be_removed, clusters_list)
                    cluster_num -= 1

                progress_bar.update(1)  
                clustering_complete = True

    if noise == True:
        # Add the remaining isolated points to the list of clusters
        for key in neighbours_strength_dict:
            elem1, elem2 = key

            if not x1 == "cluster" and elem1 not in clusters_list:
                clusters_list.append(elem1)

            if not x2 == "cluster" and elem2 not in clusters_list:
                clusters_list.append(elem2)

            progress_bar.update(1)    

    print("Clusters computed")
    return clusters_list

def refresh_corr_values(df, neighbours_strength_dict, name_elem1, name_elem2, cluster_index, clusters_list, method):

    for key in list(neighbours_strength_dict.keys()):
        (x1, y1), (x2, y2) = key

        if (x1, y1) in (name_elem1, name_elem2):
                
            strength = get_neighbours_strength(df, cluster_index, (x2, y2), clusters_list, method)
            #strength = np.corrcoef(other_coord_mean, cluster_mean)[0, 1]
            new_key = ("cluster", cluster_index), (x2, y2)
            inverse_new_key = (x2, y2), ("cluster", cluster_index)

            # Check if x1_y1 is already a cluster and doesn't change cluster_index
            if (new_key == key): 
                neighbours_strength_dict[key] = strength
            else:
                # Check if the same cluster has been considered previously (a common neighbor)
                if inverse_new_key not in neighbours_strength_dict and new_key not in neighbours_strength_dict:
                    neighbours_strength_dict[new_key] = strength
                del neighbours_strength_dict[key]
        
        elif (x2, y2) in (name_elem1, name_elem2):

            strength = get_neighbours_strength(df, cluster_index, (x1, y1), clusters_list, method)
            #strength = np.corrcoef(other_coord_mean, cluster_mean)[0, 1]
            new_key = (x1, y1), ("cluster", cluster_index)
            inverse_new_key = ("cluster", cluster_index), (x1, y1)
            
            if (new_key == key):
                neighbours_strength_dict[key] = strength
            else:  
                if inverse_new_key not in neighbours_strength_dict and new_key not in neighbours_strength_dict:
                    neighbours_strength_dict[new_key] = strength
                del neighbours_strength_dict[key]
    return


def get_neighbours_strength_dict(df, neighbours, method):
    neighbours_strength_dict = {}

    if method == 'correlation':
        for neighbor_pair in neighbours:
            neighbor1, neighbor2 = neighbor_pair
            strength = np.corrcoef(df[neighbor1], df[neighbor2])[0, 1]
            neighbours_strength_dict[neighbor_pair] = strength

    elif method == 'distance':
        for neighbor_pair in neighbours:
            neighbor1, neighbor2 = neighbor_pair
            strength = np.linalg.norm(df[neighbor1] - df[neighbor2])
            neighbours_strength_dict[neighbor_pair] = strength    

    return neighbours_strength_dict    

def get_neighbours_strength(df, cluster_index, elem2, clusters_list, method):
    elem1 = df[clusters_list[cluster_index]].mean(axis=1)
    elem2 = df[clusters_list[int(elem2[1])]].mean(axis=1) if elem2[0] == "cluster" else df[elem2]

    if method == 'correlation':
        strength = np.corrcoef(elem1, elem2)[0, 1]
    elif method == 'distance':
        strength = np.linalg.norm(elem1 - elem2)
    else:
        raise ValueError("Unsupported method. Supported methods are 'correlation' and 'distance'.")

    return strength

def enough_strength(strength, method, threshold):
    if method == 'correlation':
        return strength >= threshold
    elif method == 'distance':
        return strength <= threshold
    else:
        raise ValueError("Unsupported method. Supported methods are 'correlation' and 'distance'.")


# once the dictionary has been refreshed, I have to change name to every cluster > i to i-1 and to remove clusters_list[i]
def remove_cluster(neighbours_strength_dict, to_be_removed, clusters_list):
    for key in list(neighbours_strength_dict.keys()):
        # there could be more than one cluster_i in the same key to be updated
        updated = False
        for cluster_num in range(to_be_removed+1, len(clusters_list)): 
            
            if ("cluster", cluster_num) in key:
                if updated:
                    decomposed_key = modified_key
                else:
                    decomposed_key = key

                new_key = tuple(
                    (
                        ("cluster", cluster_num - 1)
                        if subkey == ("cluster", cluster_num)
                        else subkey
                    )
                    for subkey in decomposed_key
                )                
                modified_key = new_key
                updated = True    

        if updated:
            neighbours_strength_dict[new_key] = neighbours_strength_dict.pop(key)
    
    new_clusters_list = clusters_list[:to_be_removed] + clusters_list[to_be_removed+1:]
    return new_clusters_list


"""