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
    def __init__(self, df, neighbours, withuot_neighbours, method, threshold, missing_values=False):
        self.df = df
        self.neighbours = neighbours
        self.without_neighbours = withuot_neighbours
        self.method = method
        self.threshold = threshold
        self.missing_values = missing_values        
        self.last_clusters = {}
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
        cluster = Cluster(f'cluster_{len(self.clusters) - len(self.df.columns)}', childs, correlation)
        self.clusters.append(cluster)


    def compute_clusters(self):
        current_corr = 1

        cluster_num_internal = 0

        print("Computing clusters...")
        progress_bar = tqdm(total=self.df.shape[1], position=0, leave=True, smoothing=0.05)
        thresh_100 = int(self.threshold*100)
        #progress_bar = tqdm(total = thresh_100, position=0, leave=True, smoothing=1)

        max_strength = sorted(self.neighbours_strength_dict.keys(), key=self.neighbours_strength_dict.get, reverse=True)[0]

        while len(self.neighbours_strength_dict) != 0 and self.enough_strength(self.neighbours_strength_dict[max_strength]):
            (elem1, elem2) = max_strength
   
            # Update cluster and correlation values
            if elem1.startswith("cluster") and not elem2.startswith("cluster"):
                cluster_index = int(elem1.split("_")[1])
                self.last_clusters[cluster_index].append(elem2)

                child1 = self.get_cluster_by_ID(self.cluster_mapping[cluster_index])
                child2 = self.get_cluster_by_ID(elem2)
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                self.cluster_mapping[cluster_index] = self.clusters[-1].clusterID
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]
                self.refresh_corr_values(elem1, elem2, cluster_index)
                
            elif not elem1.startswith("cluster") and elem2.startswith("cluster"):
                cluster_index = int(elem2.split("_")[1])
                self.last_clusters[cluster_index].append(elem1)

                child1 = self.get_cluster_by_ID(elem1)
                child2 = self.get_cluster_by_ID(self.cluster_mapping[cluster_index])
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                self.cluster_mapping[cluster_index] = self.clusters[-1].clusterID
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]
                self.refresh_corr_values(elem1, elem2, cluster_index)

            elif not elem1.startswith("cluster") and not elem2.startswith("cluster"):

                self.cluster_mapping[len(self.last_clusters)] = f'cluster_{len(self.clusters) - len(self.df.columns)}'
                self.last_clusters[cluster_num_internal] = [elem1, elem2]

                child1 = self.get_cluster_by_ID(elem1)
                child2 = self.get_cluster_by_ID(elem2)
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]
                self.refresh_corr_values(elem1, elem2, cluster_num_internal)
                cluster_num_internal += 1
                
            else: # case both are clusters
                cluster1_ID = int(elem1.split("_")[1])
                cluster2_ID = int(elem2.split("_")[1])
                cluster_index = min(cluster1_ID, cluster2_ID)
                to_be_removed = max(cluster1_ID, cluster2_ID)
                self.last_clusters[cluster_index].extend(self.last_clusters[to_be_removed])

                child1 = self.get_cluster_by_ID(self.cluster_mapping[cluster1_ID])
                child2 = self.get_cluster_by_ID(self.cluster_mapping[cluster2_ID])
                self.create_cluster([child1, child2], self.neighbours_strength_dict[max_strength])
                self.cluster_mapping[cluster_index] = self.clusters[-1].clusterID
                child1.set_father(self.clusters[-1])
                child2.set_father(self.clusters[-1])

                del self.neighbours_strength_dict[max_strength]   
                self.refresh_corr_values(elem1, elem2, cluster_index)
                #self.remove_cluster(to_be_removed)
                #cluster_num_internal -= 1

            max_strength = sorted(self.neighbours_strength_dict.keys(), key=self.neighbours_strength_dict.get, reverse=True)[0]
            if (self.neighbours_strength_dict[max_strength] < current_corr - 0.01):
                current_corr = current_corr - 0.01
                formatted = "%.2f" % current_corr
                print(formatted)

            progress_bar.update(1)


        print("Adding singletons...")
        # Add the remaining isolated points to the list of clusters
        added_elem = []
        for key in self.neighbours_strength_dict:
            elem1, elem2 = key

            added_elem.append(elem1)   

            if not elem1.startswith("cluster") and not elem1 in added_elem:
                self.last_clusters[cluster_num_internal] = [elem1]
                added_elem.append(elem1)   

                cluster_num_internal += 1

                progress_bar.update(1)

            if not elem2.startswith("cluster") and not elem2 in added_elem:
                self.last_clusters[cluster_num_internal] = [elem2]
                added_elem.append(elem2)   

                cluster_num_internal += 1

                progress_bar.update(1)

        for elem in self.without_neighbours:
            self.last_clusters[cluster_num_internal] = [elem]
            cluster_num_internal += 1

            progress_bar.update(1)

        print("Clusters computed.")
        #self.last_clusters = sorted(self.last_clusters, key=lambda x: len(x), reverse=True)


    def get_neighbours_strength_dict(self):

        print("Computing neighbours strengths...")
        progress_bar = tqdm(total=len(self.neighbours), position=0, leave=True, smoothing=0)

        neighbours_strength_dict = {}

        if self.method == 'correlation' or self.method == 'complete_correlation':
            if self.missing_values:
                for neighbour_pair in self.neighbours:
                    neighbour1, neighbour2 = neighbour_pair
                    neighbour1_mask = ~np.isnan(self.df[neighbour1])
                    neighbour2_mask = ~np.isnan(self.df[neighbour2])

                    mask = neighbour1_mask * neighbour2_mask
                    strength = np.corrcoef(self.df[neighbour1][mask], self.df[neighbour2][mask])[0, 1]
                    neighbours_strength_dict[neighbour_pair] = strength
                    progress_bar.update(1)  
            else:
                for neighbour_pair in self.neighbours:
                    neighbour1, neighbour2 = neighbour_pair

                    strength = np.corrcoef(self.df[neighbour1], self.df[neighbour2])[0, 1]
                    neighbours_strength_dict[neighbour_pair] = strength
                    progress_bar.update(1)                        

        elif self.method == 'distance':
            for neighbour_pair in self.neighbours:
                neighbour1, neighbour2 = neighbour_pair
                strength = np.linalg.norm(self.df[neighbour1] - self.df[neighbour2])
                neighbours_strength_dict[neighbour_pair] = strength   
                progress_bar.update(1)  
        
        print("Neighbours strengths computed.")
        return neighbours_strength_dict    


    def get_neighbours_strength(self, cluster_mean, elem, elem_new_neighbour, key):
        strength = ''

        # "correlation" has to be adapted to work with vectors of different lengths
        if self.method == 'correlation':
            value2 = self.df[self.last_clusters[int(elem.split("_")[1])]].mean(axis=1) if elem.startswith("cluster") else self.df[elem]
            strength = np.corrcoef(cluster_mean, value2)[0, 1]

        elif self.method == 'complete_correlation':
            strength = self.neighbours_strength_dict[key]
            
            if elem_new_neighbour.startswith("cluster"):
                columns_neighbour = self.last_clusters[int(elem_new_neighbour.split("_")[1])]
            else:
                columns_neighbour = [elem_new_neighbour]

            if elem.startswith("cluster"):
                columns_elem = self.last_clusters[int(elem.split("_")[1])]
            else:
                columns_elem = [elem]

            if self.missing_values:
                for e1 in columns_elem:
                    e1_mask = ~np.isnan(self.df[e1])
                    for e2 in columns_neighbour:
                        e2_mask = ~np.isnan(self.df[e2])
                        mask = e1_mask * e2_mask
                        strength_temp = np.corrcoef(self.df[e1][mask], self.df[e2][mask])[0, 1]
                        if strength_temp < strength:
                            strength = strength_temp
            else:
                for e1 in columns_elem:
                    for e2 in columns_neighbour:
                        strength_temp = np.corrcoef(self.df[e1], self.df[e2])[0, 1]
                        if strength_temp < strength:
                            strength = strength_temp                                


        elif self.method == 'distance':
            value2 = self.df[self.last_clusters[int(elem.split("_")[1])]].mean(axis=1) if elem.startswith("cluster") else self.df[elem]
            strength = np.linalg.norm(cluster_mean - value2)

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


    def refresh_corr_values(self, elem1_cluster, elem2_cluster, cluster_index):
        cluster_mean = 0

        if self.method == 'correlation' or self.method == 'distance':
            cluster_mean = self.df[self.last_clusters[cluster_index]].mean(axis=1)

        for key in list(self.neighbours_strength_dict.keys()):
            elem1, elem2 = key

            if elem1 in (elem1_cluster, elem2_cluster):
                
                if elem1 == elem1_cluster:
                    elem_new_neighbour = elem2_cluster
                    #elem_old_neighbour = elem1_cluster
                else: 
                    elem_new_neighbour = elem1_cluster
                    #elem_old_neighbour = elem2_cluster

                strength = self.get_neighbours_strength(cluster_mean, elem2, elem_new_neighbour, key)
                new_key = (f'cluster_{cluster_index}', elem2)
                inverse_new_key = (elem2, f'cluster_{cluster_index}')

                # Check if elem1 is already a cluster and doesn't change cluster_index
                if (new_key == key): 
                    self.neighbours_strength_dict[key] = strength
                else:
                    # Check if the same cluster has been considered previously (a common neighbour)
                    if inverse_new_key not in self.neighbours_strength_dict and new_key not in self.neighbours_strength_dict:
                        self.neighbours_strength_dict[new_key] = strength
                    del self.neighbours_strength_dict[key]
            
            elif elem2 in (elem1_cluster, elem2_cluster):

                if elem2 == elem1_cluster:
                    elem_new_neighbour = elem2_cluster
                    #elem_old_neighbour = elem1_cluster
                else: 
                    elem_new_neighbour = elem1_cluster
                    #elem_old_neighbour = elem2_cluster

                strength = self.get_neighbours_strength(cluster_mean, elem1, elem_new_neighbour, key)
                new_key = (elem1, f'cluster_{cluster_index}')
                inverse_new_key = (f'cluster_{cluster_index}', elem1)

                if (new_key == key):
                    self.neighbours_strength_dict[key] = strength
                else:  
                    if inverse_new_key not in self.neighbours_strength_dict and new_key not in self.neighbours_strength_dict:
                        self.neighbours_strength_dict[new_key] = strength
                    del self.neighbours_strength_dict[key]




