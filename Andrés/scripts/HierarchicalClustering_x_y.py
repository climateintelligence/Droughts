import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, clusterID, childs, correlation):
        self.clusterID = clusterID
        self.childs = childs
        self.father = ""
        self.correlation = correlation
        self.dim = sum(child.dim for child in self.childs) if self.childs != "" else 1

    def get_last_childs(self):
        if self.childs == "":
            return [self]
        else: 
            last_childs = []
            for child in self.childs:
                last_childs.extend(child.get_last_childs())  
            return last_childs      

    def set_father(self, father):
        self.father = father

    def get_last_father(self):
        if self.father == "":
            return self
        else: 
            last_father = self.father.get_last_father()
            return last_father


class HierarchicalClustering:
    def __init__(self, df, neighbours, withuot_neighbours, method, threshold):
        self.df = df
        self.neighbours = neighbours
        self.without_neighbours = withuot_neighbours
        self.method = method
        self.threshold = threshold
        self.last_clusters = []
        self.neighbours_strength_dict = self.get_neighbours_strength_dict()
        self.clusters = [Cluster(sub_basin, "", 1) for sub_basin in df.columns]
        self.cluster_mapping = {}
        

    def get_clustering_count_analysis(self):
        ordered_fathers, ordered_correlations, ordered_count, ordered_singletons = self.get_ordered_clusters()

        # Create a single plot with two different y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the first set of data on the first y-axis
        ax1.plot(ordered_correlations, ordered_count, label='No. Clusters', color='blue')
        ax1.plot(ordered_correlations, np.subtract(np.array(ordered_count), np.array(ordered_singletons)), label='No. Clusters - Singletons', color='green')
        ax1.plot(ordered_correlations, ordered_singletons, label='No. Singletons', color='red')
        ax1.set_xlabel('Correlation threshold')
        ax1.set_ylabel('No. Clusters')
        ax1.tick_params(axis='y')

        # Create a secondary y-axis on the same plot
        ax2 = ax1.twinx()

        # Plot the second set of data on the secondary y-axis
        ax2.plot(ordered_correlations, np.subtract(np.array(ordered_count), np.array(ordered_singletons))/np.array(ordered_singletons), label='(No. Clusters - Singletons) / No. Singletons', color='purple')
        ax2.set_ylabel('(No. Clusters - Singletons) / No. Singletons', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        # Combine the legends for both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(lines, labels, loc='upper left')

        # Set titles for the combined plot
        plt.title('Complete-Linkage, Clusters Count')

        # Show the combined plot
        plt.show()    


    def get_ordered_clusters(self):
        ordered_fathers = []
        ordered_correlations = []
        ordered_count = []
        ordered_singletons = []
        
        last_fathers = self.get_last_fathers()
        min_correlation = 0

        # if min_correlation == 1 it could mean that we have some clusters of different elements with correlation 1 and we want to stop there
        while (last_fathers != [] and min_correlation != 1):
            min_correlation = 1
            min_fathers = []
            
            for father in last_fathers:
                if father.correlation < min_correlation:
                    min_correlation = father.correlation 
                    min_fathers = [father]
                
                elif father.correlation == min_correlation:      
                    min_fathers.append(father)

            ordered_fathers.append(min_fathers)
            ordered_correlations.append(min_correlation)
            ordered_count.append(len(last_fathers))

            ordered_singletons.append(len([cluster for cluster in last_fathers if cluster.childs==""]))

            for min_father in min_fathers:
                last_fathers.extend(min_father.childs)
                last_fathers.remove(min_father)

        return ordered_fathers, ordered_correlations, ordered_count, ordered_singletons
        




    def get_clusters_from_fathers(self, fathers):
        clusters_from_fathers = []
        for father in fathers:
            from_father = []
            for last_child in father.get_last_childs():
                from_father.append(last_child.clusterID)

            clusters_from_fathers.append(from_father)
        return sorted(clusters_from_fathers, key=lambda x: len(x), reverse=True)


    def get_fathers_from_last_fathers(self, threshold):
        fathers = []

        if self.method == 'complete_correlation':
            temp_fathers = self.get_last_fathers()

            while (temp_fathers != []):
                temp2_fathers = []
                for father in temp_fathers:
                    if father.correlation >= threshold:
                        fathers.append(father)
                    else: 
                        temp2_fathers.extend(father.childs)

                temp_fathers = temp2_fathers

        else: 
            #centroid_correlation requires to check from bottom-up, since correlations are not monotonic
            raise ValueError("Unsupported method. Supported methods are 'complete_correlation'.")

        return fathers


    def get_clusters_from_last_fathers(self, threshold):
        fathers = self.get_fathers_from_last_fathers(threshold)
        clusters = self.get_clusters_from_fathers(fathers)
        return clusters


    def get_last_fathers(self):
        last_fathers = []        

        for cluster in self.clusters:
            if cluster.father == "":
                last_fathers.append(cluster)

        return last_fathers    


    def get_cluster_by_ID(self, clusterID):
        for cluster in self.clusters:
            if cluster.clusterID == clusterID:
                return cluster


    def create_cluster(self, childs, correlation):
        cluster = Cluster(len(self.clusters) - len(self.df.columns), childs, correlation)
        self.clusters.append(cluster)


    def compute_clusters(self):

        curr_threshold = 1
        cluster_num_internal = 0

        print("Computing clusters...")
        progress_bar = tqdm(total=int(100*round((1-self.threshold), 2)), position=0, leave=True, smoothing=0)

        max_strength = sorted(self.neighbours_strength_dict.keys(), key=self.neighbours_strength_dict.get, reverse=True)[0]

        while self.enough_strength(self.neighbours_strength_dict[max_strength]):
            x1, y1, x2, y2 = max_strength.split("_")
            
            # Update cluster and correlation values
            if x1 == "cluster" and not x2 == "cluster":
                cluster_index = int(y1)
                self.last_clusters[cluster_index].append(x2 + "_" + y2)

                child1 = self.get_cluster_by_ID(self.cluster_mapping[cluster_index])
                child2 = self.get_cluster_by_ID(x2 + "_" + y2)
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                self.cluster_mapping[cluster_index] = self.clusters[-1].clusterID
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]
                self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, cluster_index)
                
            elif not x1 == "cluster" and x2 == "cluster":
                cluster_index = int(y2)
                self.last_clusters[cluster_index].append(x1 + "_" + y1)

                child1 = self.get_cluster_by_ID(x1 + "_" + y1)
                child2 = self.get_cluster_by_ID(self.cluster_mapping[cluster_index])
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                self.cluster_mapping[cluster_index] = self.clusters[-1].clusterID
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]
                self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, cluster_index)

            elif not x1 == "cluster" and not x2 == "cluster":
                self.cluster_mapping[len(self.last_clusters)] = len(self.clusters) - len(self.df.columns)

                self.last_clusters.append([x1 + "_" + y1, x2 + "_" + y2])

                child1 = self.get_cluster_by_ID(x1 + "_" + y1)
                child2 = self.get_cluster_by_ID(x2 + "_" + y2)
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]
                self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, cluster_num_internal)
                cluster_num_internal += 1
                
            else: # case both are clusters
                cluster_index = min(int(y1), int(y2))
                to_be_removed = max(int(y1), int(y2))
                self.last_clusters[cluster_index].extend(self.last_clusters[to_be_removed])

                child1 = self.get_cluster_by_ID(self.cluster_mapping[int(y1)])
                child2 = self.get_cluster_by_ID(self.cluster_mapping[int(y2)])
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                self.cluster_mapping[cluster_index] = self.clusters[-1].clusterID
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]   
                self.refresh_corr_values(x1 + "_" + y1, x2 + "_" + y2, cluster_index)
                self.remove_cluster(to_be_removed)
                cluster_num_internal -= 1

            max_strength = sorted(self.neighbours_strength_dict.keys(), key=self.neighbours_strength_dict.get, reverse=True)[0]
            if self.neighbours_strength_dict[max_strength] < curr_threshold:
                progress_bar.update(1)
                curr_threshold -= 0.01

        print("Adding singletons...")
        # Add the remaining isolated points to the list of clusters
        for key in self.neighbours_strength_dict:
            x1, y1, x2, y2 = key.split("_")
            elem1 = [x1 + "_" + y1]
            elem2 = [x2 + "_" + y2]

            if not x1 == "cluster" and elem1 not in self.last_clusters:
                self.last_clusters.append(elem1)

            if not x2 == "cluster" and elem2 not in self.last_clusters:
                self.last_clusters.append(elem2) 

        for elem in self.without_neighbours:
            self.last_clusters.append([elem])

        print("Clusters computed.")
        self.last_clusters = sorted(self.last_clusters, key=lambda x: len(x), reverse=True)


    def get_neighbours_strength_dict(self):

        print("Computing neighbours strengths...")
        progress_bar = tqdm(total=len(self.neighbours), position=0, leave=True, smoothing=0)

        neighbours_strength_dict = {}

        if self.method == 'correlation' or self.method == 'complete_correlation':
            for neighbor_pair in self.neighbours:
                neighbor1, neighbor2 = neighbor_pair

                strength = np.corrcoef(self.df[neighbor1], self.df[neighbor2])[0, 1]
                neighbours_strength_dict['_'.join(neighbor_pair)] = strength
                progress_bar.update(1)  

        elif self.method == 'distance':
            for neighbor_pair in self.neighbours:
                neighbor1, neighbor2 = neighbor_pair
                strength = np.linalg.norm(self.df[neighbor1] - self.df[neighbor2])
                neighbours_strength_dict['_'.join(neighbor_pair)] = strength   
                progress_bar.update(1)  
        
        print("Neighbours strengths computed.")
        return neighbours_strength_dict    


    def get_neighbours_strength(self, cluster_index, cluster_mean, x2, y2):
        strength = ''

        if self.method == 'correlation':
            elem2 = self.df[self.last_clusters[int(y2)]].mean(axis=1) if x2 == "cluster" else self.df[x2 + "_" + y2]
            strength = np.corrcoef(cluster_mean, elem2)[0, 1]

        elif self.method == 'complete_correlation':
            strength = 1
            if x2 == "cluster":
                cluster1 = self.df[self.last_clusters[cluster_index]]
                cluster2 = self.df[self.last_clusters[int(y2)]]
                for elem1 in cluster1.columns:
                    for elem2 in cluster2.columns:
                        strength_temp = np.corrcoef(self.df[elem1], self.df[elem2])[0, 1]
                        if strength_temp < strength:
                            strength = strength_temp

            else:
                cluster1 = self.df[self.last_clusters[cluster_index]]
                for elem1 in cluster1.columns:
                    strength_temp = np.corrcoef(self.df[elem1], self.df[x2 + "_" + y2])[0, 1]
                    if strength_temp < strength:
                        strength = strength_temp    

        elif self.method == 'distance':
            elem2 = self.df[self.last_clusters[int(y2)]].mean(axis=1) if x2 == "cluster" else self.df[x2 + "_" + y2]
            strength = np.linalg.norm(cluster_mean - elem2)

        else:
            raise ValueError("Unsupported method. Supported methods are 'correlation', 'complete_correlation and 'distance'.")

        return strength


    def enough_strength(self, strength):
        if self.method == 'correlation' or self.method == 'complete_correlation':
            return strength >= self.threshold
        elif self.method == 'distance':
            return strength <= self.threshold
        else:
            raise ValueError("Unsupported method. Supported methods are 'correlation', 'complete_correlation' and 'distance'.")


    def refresh_corr_values(self, name_elem1, name_elem2, cluster_index):
        cluster_mean = self.df[self.last_clusters[cluster_index]].mean(axis=1)
        for key in list(self.neighbours_strength_dict.keys()):
            x1, y1, x2, y2 = key.split("_")

            if (x1 + "_" + y1) in (name_elem1, name_elem2):
                    
                strength = self.get_neighbours_strength(cluster_index, cluster_mean, x2, y2)
                new_key = f'cluster_{cluster_index}_{x2}_{y2}'
                inverse_new_key = f'{x2}_{y2}_cluster_{cluster_index}'

                # Check if x1_y1 is already a cluster and doesn't change cluster_index
                if (new_key == key): 
                    self.neighbours_strength_dict[key] = strength
                else:
                    # Check if the same cluster has been considered previously (a common neighbor)
                    if inverse_new_key not in self.neighbours_strength_dict and new_key not in self.neighbours_strength_dict:
                        self.neighbours_strength_dict[new_key] = strength
                    del self.neighbours_strength_dict[key]
            
            elif (x2 + "_" + y2) in (name_elem1, name_elem2):

                strength = self.get_neighbours_strength(cluster_index, cluster_mean, x1, y1)
                new_key = f'{x1}_{y1}_cluster_{cluster_index}'
                inverse_new_key = f'cluster_{cluster_index}_{x1}_{y1}'
                
                if (new_key == key):
                    self.neighbours_strength_dict[key] = strength
                else:  
                    if inverse_new_key not in self.neighbours_strength_dict and new_key not in self.neighbours_strength_dict:
                        self.neighbours_strength_dict[new_key] = strength
                    del self.neighbours_strength_dict[key]


    # once the dictionary has been refreshed, I have to change name to every cluster > i to i-1 and to remove self.last_clusters[i]
    def remove_cluster(self, to_be_removed):
        for key in list(self.neighbours_strength_dict.keys()):
            modified_key = ""
            # !!! there could be more than one cluster_i in the same key to be updated
            updated = False
            for cluster_n in range(to_be_removed+1, len(self.last_clusters)): 
                
                if f'cluster_{cluster_n}' in key:
                    if updated:
                        decomposed_key = modified_key.split('_')
                    else:
                        decomposed_key = key.split('_')

                    new_key = '_'.join([str(cluster_n-1) if subkey == str(cluster_n) else subkey for subkey in decomposed_key])
                    modified_key = new_key
                    updated = True     

            if updated:
                self.neighbours_strength_dict[new_key] = self.neighbours_strength_dict.pop(key)

        for cluster_n in range(to_be_removed+1, len(self.cluster_mapping)): 
            self.cluster_mapping[cluster_n - 1] = self.cluster_mapping[cluster_n]
        del self.cluster_mapping[len(self.cluster_mapping)-1]

        self.last_clusters = self.last_clusters[:to_be_removed] + self.last_clusters[to_be_removed+1:]


