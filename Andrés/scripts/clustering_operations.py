import numpy as np
from tqdm import tqdm




def compute_neighbors(df, max_distance=1):
    neighbors_list = []
    coordinates = df.columns

    print("Computing neighbors...")
    progress_bar = tqdm(total=len(coordinates), position=0, leave=True, smoothing=0)

    for idx, coord1 in enumerate(coordinates):
        for idx2, coord2 in enumerate(coordinates[idx + 1:], start=idx + 1):
            x1, y1 = map(float, coord1.split('_'))
            x2, y2 = map(float, coord2.split('_'))
            distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            
            if distance < max_distance:
                neighbors_list.append((coord1, coord2))

        progress_bar.update(1) 

    print("Neighbors computed")
    return neighbors_list

def compute_clusters(df, neighbors_list, method, threshold, noise=False):
    
    neighbors_strength_dict = get_neighbors_strength_dict(df, neighbors_list, method)     

    cluster_num = 0
    clusters_list = []
    clustering_complete = True

    print("Computing clusters...")
    progress_bar = tqdm(total=df.shape[1], position=0, leave=True, smoothing=0)

    while clustering_complete:
        sorted_keys = sorted(neighbors_strength_dict.keys(), key=neighbors_strength_dict.get, reverse=True)
        clustering_complete = False

        for max_strength in sorted_keys:
            # Check for new clusters if none have been made in this iteration yet
            if clustering_complete :
                break
            
            elif enough_strength(neighbors_strength_dict[max_strength], method, threshold):
                x1, y1, x2, y2 = max_strength.split("_")
                
                # Update cluster and correlation values
                if x1 == "cluster" and not x2 == "cluster":
                    cluster_index = int(y1)
                    clusters_list[cluster_index].append(x2 + "_" + y2)
                    del neighbors_strength_dict[max_strength]
                    refresh_corr_values(df, neighbors_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_index, clusters_list, method)
                    
                elif not x1 == "cluster" and x2 == "cluster":
                    cluster_index = int(y2)
                    clusters_list[cluster_index].append(x1 + "_" + y1)
                    del neighbors_strength_dict[max_strength]
                    refresh_corr_values(df, neighbors_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_index, clusters_list, method)

                elif not x1 == "cluster" and not x2 == "cluster":
                    clusters_list.append([x1 + "_" + y1, x2 + "_" + y2])
                    del neighbors_strength_dict[max_strength]
                    refresh_corr_values(df, neighbors_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_num, clusters_list, method)
                    cluster_num += 1
                    
                else: # case both are clusters
                    cluster_index = min(int(y1), int(y2))
                    to_be_removed = max(int(y1), int(y2))
                    clusters_list[cluster_index].extend(clusters_list[to_be_removed])
                    del neighbors_strength_dict[max_strength]   
                    refresh_corr_values(df, neighbors_strength_dict, x1 + "_" + y1, x2 + "_" + y2, cluster_index, clusters_list, method)
                    clusters_list = remove_cluster(neighbors_strength_dict, to_be_removed, clusters_list)
                    cluster_num -= 1

                progress_bar.update(1)  
                clustering_complete = True

    if noise == True:
        # Add the remaining isolated points to the list of clusters
        for key in neighbors_strength_dict:
            x1, y1, x2, y2 = key.split("_")
            elem1 = [x1 + "_" + y1]
            elem2 = [x2 + "_" + y2]

            if not x1 == "cluster" and elem1 not in clusters_list:
                clusters_list.append(elem1)

            if not x2 == "cluster" and elem2 not in clusters_list:
                clusters_list.append(elem2)

            progress_bar.update(1)    

    print("Clusters computed")
    return clusters_list

def get_neighbors_strength_dict(df, neighbors_list, method):
    neighbors_strength_dict = {}

    if method == 'correlation':
        for neighbor_pair in neighbors_list:
            neighbor1, neighbor2 = neighbor_pair
            strength = np.corrcoef(df[neighbor1], df[neighbor2])[0, 1]
            neighbors_strength_dict['_'.join(neighbor_pair)] = strength

    elif method == 'distance':
        for neighbor_pair in neighbors_list:
            neighbor1, neighbor2 = neighbor_pair
            strength = np.linalg.norm(df[neighbor1] - df[neighbor2])
            neighbors_strength_dict['_'.join(neighbor_pair)] = strength    

    return neighbors_strength_dict    

def get_neighbors_strength(df, cluster_index, x2, y2, clusters_list, method):
    strength = ''
    elem1 = df[clusters_list[cluster_index]].mean(axis=1)
    elem2 = df[clusters_list[int(y2)]].mean(axis=1) if x2 == "cluster" else df[x2 + "_" + y2]
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


def refresh_corr_values(df, neighbors_strength_dict, name_elem1, name_elem2, cluster_index, clusters_list, method):

    for key in list(neighbors_strength_dict.keys()):
        x1, y1, x2, y2 = key.split("_")

        if (x1 + "_" + y1) in (name_elem1, name_elem2):
                
            strength = get_neighbors_strength(df, cluster_index, x2, y2, clusters_list, method)
            new_key = f'cluster_{cluster_index}_{x2}_{y2}'
            inverse_new_key = f'{x2}_{y2}_cluster_{cluster_index}'

            # Check if x1_y1 is already a cluster and doesn't change cluster_index
            if (new_key == key): 
                neighbors_strength_dict[key] = strength
            else:
                # Check if the same cluster has been considered previously (a common neighbor)
                if inverse_new_key not in neighbors_strength_dict and new_key not in neighbors_strength_dict:
                    neighbors_strength_dict[new_key] = strength
                del neighbors_strength_dict[key]
        
        elif (x2 + "_" + y2) in (name_elem1, name_elem2):

            strength = get_neighbors_strength(df, cluster_index, x1, y1, clusters_list, method)
            new_key = f'{x1}_{y1}_cluster_{cluster_index}'
            inverse_new_key = f'cluster_{cluster_index}_{x1}_{y1}'
            
            if (new_key == key):
                neighbors_strength_dict[key] = strength
            else:  
                if inverse_new_key not in neighbors_strength_dict and new_key not in neighbors_strength_dict:
                    neighbors_strength_dict[new_key] = strength
                del neighbors_strength_dict[key]
    return

# once the dictionary has been refreshed, I have to change name to every cluster > i to i-1 and to remove clusters_list[i]
def remove_cluster(neighbors_strength_dict, to_be_removed, clusters_list):
    for key in list(neighbors_strength_dict.keys()):
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
            neighbors_strength_dict[new_key] = neighbors_strength_dict.pop(key)
    
    new_clusters_list = clusters_list[:to_be_removed] + clusters_list[to_be_removed+1:]
    return new_clusters_list


"""
def compute_neighbors(df, max_distance=1):
    neighbors_list = []
    coordinates = df.columns

    print("Computing neighbors...")
    progress_bar = tqdm(total=len(coordinates), position=0, leave=True, smoothing=0)

    for idx, coord1 in enumerate(coordinates):
        for idx2, coord2 in enumerate(coordinates[idx + 1:], start=idx + 1):
            distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            if distance < max_distance:
                neighbors_list.append((coord1, coord2))

        progress_bar.update(1) 

    print("Neighbors computed")
    return neighbors_list

def compute_clusters(df, neighbors_list, method, threshold, noise=False):
    neighbors_strength_dict = get_neighbors_strength_dict(df, neighbors_list, method)     
    cluster_num = 0
    clusters_list = []
    clustering_complete = True

    print("Computing clusters...")
    progress_bar = tqdm(total=df.shape[1], position=0, leave=True, smoothing=0)

    while clustering_complete:
        sorted_keys = sorted(neighbors_strength_dict.keys(), key=neighbors_strength_dict.get, reverse=True)
        clustering_complete = False

        for max_strength in sorted_keys:
            # Check for new clusters if none have been made in this iteration yet
            if clustering_complete :
                break
            
            elif enough_strength(neighbors_strength_dict[max_strength], method, threshold):
                (x1, y1), (x2, y2) = max_strength
                
                # Update cluster and correlation values
                if x1 == "cluster" and not x2 == "cluster":
                    cluster_index = int(y1)
                    clusters_list[cluster_index].append((x2, y2))
                    del neighbors_strength_dict[max_strength]
                    refresh_corr_values(df, neighbors_strength_dict, (x1, y1), (x2, y2), cluster_index, clusters_list, method)
                    
                elif not x1 == "cluster" and x2 == "cluster":
                    cluster_index = int(y2)
                    clusters_list[cluster_index].append((x1, y1))
                    del neighbors_strength_dict[max_strength]
                    refresh_corr_values(df, neighbors_strength_dict, (x1, y1), (x2, y2), cluster_index, clusters_list, method)

                elif not x1 == "cluster" and not x2 == "cluster":
                    clusters_list.append([(x1, y1), (x2, y2)])
                    del neighbors_strength_dict[max_strength]
                    refresh_corr_values(df, neighbors_strength_dict, (x1, y1), (x2, y2), cluster_num, clusters_list, method)
                    cluster_num += 1
                    
                else: # case both are clusters
                    cluster_index = min(int(y1), int(y2))
                    to_be_removed = max(int(y1), int(y2))
                    clusters_list[cluster_index].extend(clusters_list[to_be_removed])
                    del neighbors_strength_dict[max_strength]   
                    refresh_corr_values(df, neighbors_strength_dict, (x1, y1), (x2, y2), cluster_index, clusters_list, method)
                    clusters_list = remove_cluster(neighbors_strength_dict, to_be_removed, clusters_list)
                    cluster_num -= 1

                progress_bar.update(1)  
                clustering_complete = True

    if noise == True:
        # Add the remaining isolated points to the list of clusters
        for key in neighbors_strength_dict:
            elem1, elem2 = key

            if not x1 == "cluster" and elem1 not in clusters_list:
                clusters_list.append(elem1)

            if not x2 == "cluster" and elem2 not in clusters_list:
                clusters_list.append(elem2)

            progress_bar.update(1)    

    print("Clusters computed")
    return clusters_list

def refresh_corr_values(df, neighbors_strength_dict, name_elem1, name_elem2, cluster_index, clusters_list, method):

    for key in list(neighbors_strength_dict.keys()):
        (x1, y1), (x2, y2) = key

        if (x1, y1) in (name_elem1, name_elem2):
                
            strength = get_neighbors_strength(df, cluster_index, (x2, y2), clusters_list, method)
            #strength = np.corrcoef(other_coord_mean, cluster_mean)[0, 1]
            new_key = ("cluster", cluster_index), (x2, y2)
            inverse_new_key = (x2, y2), ("cluster", cluster_index)

            # Check if x1_y1 is already a cluster and doesn't change cluster_index
            if (new_key == key): 
                neighbors_strength_dict[key] = strength
            else:
                # Check if the same cluster has been considered previously (a common neighbor)
                if inverse_new_key not in neighbors_strength_dict and new_key not in neighbors_strength_dict:
                    neighbors_strength_dict[new_key] = strength
                del neighbors_strength_dict[key]
        
        elif (x2, y2) in (name_elem1, name_elem2):

            strength = get_neighbors_strength(df, cluster_index, (x1, y1), clusters_list, method)
            #strength = np.corrcoef(other_coord_mean, cluster_mean)[0, 1]
            new_key = (x1, y1), ("cluster", cluster_index)
            inverse_new_key = ("cluster", cluster_index), (x1, y1)
            
            if (new_key == key):
                neighbors_strength_dict[key] = strength
            else:  
                if inverse_new_key not in neighbors_strength_dict and new_key not in neighbors_strength_dict:
                    neighbors_strength_dict[new_key] = strength
                del neighbors_strength_dict[key]
    return


def get_neighbors_strength_dict(df, neighbors_list, method):
    neighbors_strength_dict = {}

    if method == 'correlation':
        for neighbor_pair in neighbors_list:
            neighbor1, neighbor2 = neighbor_pair
            strength = np.corrcoef(df[neighbor1], df[neighbor2])[0, 1]
            neighbors_strength_dict[neighbor_pair] = strength

    elif method == 'distance':
        for neighbor_pair in neighbors_list:
            neighbor1, neighbor2 = neighbor_pair
            strength = np.linalg.norm(df[neighbor1] - df[neighbor2])
            neighbors_strength_dict[neighbor_pair] = strength    

    return neighbors_strength_dict    

def get_neighbors_strength(df, cluster_index, elem2, clusters_list, method):
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
def remove_cluster(neighbors_strength_dict, to_be_removed, clusters_list):
    for key in list(neighbors_strength_dict.keys()):
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
            neighbors_strength_dict[new_key] = neighbors_strength_dict.pop(key)
    
    new_clusters_list = clusters_list[:to_be_removed] + clusters_list[to_be_removed+1:]
    return new_clusters_list


"""