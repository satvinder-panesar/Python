
# coding: utf-8

# # Satvinder Singh Panesar
# ## 5024 8888
# ### Project 2

# ### Imports And Helper Functions

# In[2]:

import numpy as np, pandas as pd, random, matplotlib.patches as mpatches, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from random import choices
from math import pow, sqrt
from datetime import datetime

colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#9A6324', '#000000', '#800000', '#e6beff', '#aaffc3', '#a9a9a9', '#000075', '#469990']

def get_reduced_dimensions(eig_vector_x, eig_vector_y, features, no_of_attributes):
    reduced_dimensions = []
    for feature in features:
        x = 0
        y = 0
        for i in range(0,no_of_attributes-1):
            x=x+eig_vector_x[i]*feature[i]
            y=y+eig_vector_y[i]*feature[i]
        reduced_dimensions.append([x,y])
    return pd.DataFrame(data=reduced_dimensions)

def centroids_equal(list1, list2):
    for index, ele in enumerate(list1):
        if ele != list2[index]:
            return False
    return True

def get_ground_truth_matrix(no_of_objects, id_ground_truth_map):
    columns = list(np.arange(1, no_of_objects+1))
    columns = [int(x) for x in columns]
    rows = list(np.arange(1, no_of_objects+1))
    rows = [int(x) for x in rows]

    ground_truth_matrix = pd.DataFrame(columns = columns, index = rows)

    for row in range(1, no_of_objects + 1):
            for column in range(1, row+1):
                if row == column:
                    ground_truth_matrix[column][row] = 1
                else:
                    if id_ground_truth_map[column] == id_ground_truth_map[row]:
                        ground_truth_matrix[column][row] = 1
                    else:
                        ground_truth_matrix[column][row] = 0
                        
    return ground_truth_matrix

def get_prediction_matrix(no_of_objects, id_prediction_map):
    columns = list(np.arange(1, no_of_objects+1))
    columns = [int(x) for x in columns]
    rows = list(np.arange(1, no_of_objects+1))
    rows = [int(x) for x in rows]

    prediction_matrix = pd.DataFrame(columns = columns, index = rows)

    for row in range(1, no_of_objects + 1):
            for column in range(1, row+1):
                if row == column:
                    prediction_matrix[column][row] = 1
                else:
                    if id_prediction_map[column] == id_prediction_map[row]:
                        prediction_matrix[column][row] = 1
                    else:
                        prediction_matrix[column][row] = 0
    return prediction_matrix

def getNeighborPts(id, epsilon):
    neighbors = []
    for ele in ids:
        given_point = id_feature_map[id]
        ele_in_list = id_feature_map[ele]
        diff = np.subtract(given_point, ele_in_list)
        diff_square = [pow(x,2) for x in diff]
        total = np.sum(diff_square)
        if total <= epsilon:
            neighbors.append(ele)
    return neighbors

def expandCluster(id, neighbors, epsilon, min_pts):
    cluster = []
    cluster.append(id)
    id_flags_map[id] = [True, True]
    for neighbor in neighbors:
        flags = id_flags_map[neighbor]
        if flags[0] == False:
            id_flags_map[neighbor] = [True, flags[1]]
            new_neighbors = getNeighborPts(neighbor, epsilon)
            if len(new_neighbors) >= min_pts:
                for new_neighbor in new_neighbors:
                    if new_neighbor not in neighbors:
                        neighbors.append(new_neighbor)
        if flags[1] == False:
            cluster.append(neighbor)
            id_flags_map[neighbor] = [True, True]
    return cluster

def evaluate_clusters(no_of_objects, ground_truth_matrix, prediction_matrix):
    m11 = 0
    m00 = 0
    m10 = 0
    m01 = 0
    for row in range(1, no_of_objects + 1):
        for column in range(1, row+1):
            if ground_truth_matrix[column][row] == 1 and prediction_matrix[column][row] == 1:
                m11 = m11 + 1
            elif ground_truth_matrix[column][row] == 0 and prediction_matrix[column][row] == 0:
                m00 = m00 + 1
            elif ground_truth_matrix[column][row] == 1 and prediction_matrix[column][row] == 0:
                m10 = m10 + 1
            elif ground_truth_matrix[column][row] == 0 and prediction_matrix[column][row] == 1:
                m01 = m01 + 1
    jaccard_coefficient = m11/(m11+m10+m01)
    print("==>Jaccard Coefficient: "+str(round(jaccard_coefficient, 2)))
    rand_index = (m11+m00)/(m11+m00+m10+m01)
    print("==>Rand Index: "+str(round(rand_index, 2)))


# ### Visualizing datasets

# In[5]:

filenames = ['cho.txt', 'iyer.txt']
#filenames = ['new_dataset_1.txt', 'new_dataset_2.txt']

for filename in filenames: 
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_objects = len(data)
    no_of_columns = len(data.columns)
    no_of_attributes = no_of_columns - 2
    features = np.array(data.iloc[:,2:no_of_columns])
    features = [list(x) for x in features]
    ids = np.array(data.iloc[:,0:1])
    ids = [int(x) for x in ids]
    
    ground_truths = np.array(data.iloc[:,1:2])
    ground_truths = [int(x) for x in ground_truths]

    id_ground_truth_map = {}

    for index, id in enumerate(ids):
        id_ground_truth_map[id] = ground_truths[index]

    predictions = []
    for i in range(1, no_of_objects + 1):
        predictions.append(id_ground_truth_map[i])

    # get colors to represent labels
    colors_selected = random.sample(colors, len(set(predictions)))

    # assigning colors to labels
    cluster_no_color_map = {}
    cluster_colors = []
    for cluster_no in predictions:
        if cluster_no not in cluster_no_color_map:
            color = colors_selected.pop(0)
            cluster_no_color_map[cluster_no] = color
            cluster_colors.append(color)
        else:
            cluster_colors.append(cluster_no_color_map[cluster_no])

    # legends for scatter plot    
    patches = []
    for ele in cluster_no_color_map:
        patches.append(mpatches.Patch(color=cluster_no_color_map[ele], label=ele))
        
    pca = PCA(n_components=2)
    pca.fit(features)
    features = np.array(pca.transform(features))
    
    features_x = []
    features_y = []
    
    for ele in features:
        features_x.append(ele[0])
        features_y.append(ele[1])

    # show scatter plot    
    plt.figure(figsize=(8,6))
    plt.scatter(features_x, features_y, c=cluster_colors)
    plt.legend(handles=patches)
    plt.title("Plotting "+filename)
    plt.show()


# ### K-Means Clustering

# In[8]:

filenames = ['cho.txt', 'iyer.txt']
#filenames = ['new_dataset_1.txt', 'new_dataset_2.txt']

file = open("K_Means_Clusters.txt", "w")
file.close()

initial_centroids_given = False

# ids to be selected as centroids, one per file
initial_centroids = [[50,100,150,200,250],[50,100,150,200,250]]

for cycle in range(1, 6):

    for file_index, filename in enumerate(filenames):    
        print("====>"+filename)
        data = pd.read_csv(filename, delimiter="\t", header=None)
        no_of_objects = len(data)
        no_of_columns = len(data.columns)
        no_of_attributes = no_of_columns - 2
        features = np.array(data.iloc[:,2:no_of_columns])
        features = [list(x) for x in features]
        ids = np.array(data.iloc[:,0:1])
        ids = [int(x) for x in ids]

        # id and feature mapping
        id_feature_map = {}

        for index, id in enumerate(ids):
            id_feature_map[id] = features[index]

        ground_truths = np.array(data.iloc[:,1:2])
        ground_truths = [int(x) for x in ground_truths]

        if initial_centroids_given == False:
            no_of_clusters = np.max(ground_truths)
            centroids = choices(features, k = no_of_clusters)
        else:
            no_of_clusters = len(initial_centroids[file_index])
            centroids = []
            for id in initial_centroids[file_index]:
                centroids.append(id_feature_map[id])

        no_of_iterations = 100

        iteration_no = 1

        print(datetime.now())

        while True:

            # cluster_no and ids mapping
            clusters = {}
            for i in range(1, no_of_clusters+1):
                clusters[i] = []

            for id in ids:
                min_dist = -1
                cluster_selected = -1
                for centroid_index, centroid in enumerate(centroids):
                    diff = np.subtract(id_feature_map[id], centroid)
                    diff = [pow(x, 2) for x in diff]
                    curr_dist = np.sum(diff)
                    if min_dist == -1:
                        min_dist = curr_dist
                        cluster_selected = centroid_index + 1
                    elif curr_dist < min_dist:
                        min_dist = curr_dist
                        cluster_selected = centroid_index + 1
                clusters[cluster_selected].append(id)

            prev_centroids = np.copy(centroids)

            new_centroids = []

            # update centroids
            for centroid_index, centroid in enumerate(prev_centroids):
                features_in_cluster = clusters[centroid_index+1]
                if len(features_in_cluster) > 0:
                    features_in_cluster = [id_feature_map[x] for x in features_in_cluster]
                    features_in_cluster = pd.DataFrame(features_in_cluster)
                    no_of_features_in_cluster = len(features_in_cluster)
                    temp = []
                    for i in range(0, no_of_attributes):
                        temp.append(np.sum(features_in_cluster[i])/no_of_features_in_cluster)
                    new_centroids.append(temp)
                else:
                    new_centroids.append(centroid)

            convergence_reached = False

            for index, new_centroid in enumerate(new_centroids):
                convergence_reached = centroids_equal(new_centroid, prev_centroids[index])
                if not convergence_reached:
                    break

            if convergence_reached or iteration_no == no_of_iterations:
                # break infinite loop
                if convergence_reached:
                    print("Convergence reached at Iteration "+str(iteration_no))
                else:
                    print("Iteration "+str(iteration_no)+" completed")                 
                break
            else:
                if iteration_no % 50 == 0:
                    print("Iteration "+str(iteration_no)+" completed")             
                iteration_no = iteration_no + 1
                centroids = np.copy(new_centroids)

        print(datetime.now())
        print("==>Centroids")
        for ele in new_centroids:
            ele = [round(x, 2) for x in ele]
            print(ele)
        file = open("K_Means_Clusters.txt", "a")
        file.write("====>"+filename+" clusters\n")
        file.write("==>"+str(clusters))
        file.write("\n\n")
        file.close()

        id_ground_truth_map = {}

        for index, id in enumerate(ids):
            id_ground_truth_map[id] = ground_truths[index]

        ground_truth_matrix = get_ground_truth_matrix(no_of_objects, id_ground_truth_map)

        id_prediction_map = {}

        for key in clusters:
            for ele in clusters[key]:
                id_prediction_map[ele] = key

        prediction_matrix = get_prediction_matrix(no_of_objects, id_prediction_map)

        evaluate_clusters(no_of_objects, ground_truth_matrix, prediction_matrix)

        predictions = []
        for i in range(1, no_of_objects + 1):
            predictions.append(id_prediction_map[i])

        # get colors to represent labels
        colors_selected = random.sample(colors, len(set(predictions)))

        # assigning colors to labels
        cluster_no_color_map = {}
        cluster_colors = []
        for cluster_no in predictions:
            if cluster_no not in cluster_no_color_map:
                color = colors_selected.pop(0)
                cluster_no_color_map[cluster_no] = color
                cluster_colors.append(color)
            else:
                cluster_colors.append(cluster_no_color_map[cluster_no])

        # legends for scatter plot    
        patches = []
        index = 0
        for ele in cluster_no_color_map:
            index = index + 1
            patches.append(mpatches.Patch(color=cluster_no_color_map[ele], label=index))

        pca = PCA(n_components=2)
        pca.fit(features)
        features = np.array(pca.transform(features))

        features_x = []
        features_y = []

        for ele in features:
            features_x.append(ele[0])
            features_y.append(ele[1])

        # show scatter plot    
        plt.figure(figsize=(8,6))
        plt.scatter(features_x, features_y, c=cluster_colors)
        plt.legend(handles=patches)
        plt.title("Plotting K-Means Clusters for "+filename)
        plt.show()


# ### Agglomerative Clustering (Single Link - Min)

# In[18]:

filenames = ['cho.txt', 'iyer.txt']
#filenames = ['new_dataset_1.txt', 'new_dataset_2.txt']

file = open('Agg_Clustering_Single_Link_Clusters.txt',"w")
file.close()

for filename in filenames:
    print("====>"+filename)    
    file = open('Agg_Clustering_Single_Link_Clusters.txt',"a")
    file.write("====>"+filename+" clusters\n")
    file.close()

    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_objects = len(data)
    no_of_columns = len(data.columns)
    no_of_attributes = no_of_columns - 2
    features = data.iloc[:,2:no_of_columns]
    features.columns = [x for x in range(1, no_of_columns - 1)]
    features.index = np.arange(1, no_of_objects+1)
    ids = np.array(data.iloc[:,0:1])
    ids = [int(x) for x in ids]

    # id and feature mapping
    id_feature_map = {}

    for index, id in enumerate(ids):
        # loc gives row with row_label as argument
        id_feature_map[id] = np.array(features.loc[id])
        
    ground_truths = np.array(data.iloc[:,1:2])
    ground_truths = [int(x) for x in ground_truths]
    
    no_of_clusters = np.max(ground_truths)

    columns = list(np.arange(1, no_of_objects+1))
    columns = [int(x) for x in columns]
    rows = list(np.arange(1, no_of_objects+1))
    rows = [int(x) for x in rows]

    base_distance_matrix = pd.DataFrame(columns = columns, index = rows)

    min_distance = -1
    min_row = -1
    min_column = -1

    for row in range(1, no_of_objects + 1):
        for column in range(1, row+1):
            if row == column:
                # pd accessed by column, row
                base_distance_matrix[column][row] = 0
            else:
                feature1 = id_feature_map[column]
                feature2 = id_feature_map[row]
                diff = np.subtract(feature1, feature2)
                diff_square = [pow(x, 2) for x in diff]
                total = np.sum(diff_square)
                if min_distance == -1:
                    min_distance = total
                    min_row = row
                    min_column = column
                else:
                    if total < min_distance:
                        min_distance = total
                        min_row = row
                        min_column = column
                base_distance_matrix[column][row] = total

    distance_matrix = np.copy(base_distance_matrix)

    clusters = []
    
    temp = []
    for ele in ids:
        temp.append([ele])
    
    # every element is a cluster in itself
    clusters.append(temp)

    print(datetime.now())
    
    cluster_no_ids_map = {}
        
    cluster_no_ids_map[len(temp)] = str(temp)
    
    while True:
        # start with distance matrix and row-column with min value
        cluster = str(min_column)+","+str(min_row)
        # get last element from clusters array
        temp = clusters[len(clusters)-1]
        temp.remove([min_row])
        temp.remove([min_column])
        temp.append([cluster])
        cluster_no_ids_map[len(temp)] = str(temp)
        clusters.append(temp)
        file = open('Agg_Clustering_Single_Link_Clusters.txt',"a")
        file.write("==>"+cluster+"\n")
        file.write(str(temp))
        file.write("\n")
        file.close()
        columns.remove(min_row)
        columns.remove(min_column)
        rows.remove(min_row)
        rows.remove(min_column)
        columns.append(cluster)
        rows.append(cluster)
        distance_matrix = pd.DataFrame(columns = columns, index = rows)
        if len(columns) == 1:
            break
        min_distance = -1
        min_row = -1
        min_column = -1
        for row in rows:
            for column in columns:
                if row == column:
                    distance_matrix[column][row] = 0
                    break
                else:
                    if isinstance(column, int) and isinstance(row, int):
                        dist_val = base_distance_matrix[column][row]
                        distance_matrix[column][row] = dist_val
                        if min_distance == -1:
                            min_distance = dist_val
                            min_row = row
                            min_column = column
                        else:
                            if dist_val < min_distance:
                                min_distance = dist_val
                                min_row = row
                                min_column = column
                    else:
                        # column or row label is a cluster
                        if not isinstance(row, int) and not isinstance(column, int):
                            local_min_distance = -1
                            for ele1 in [int(x) for x in column.split(",")]:
                                for ele2 in [int(y) for y in row.split(",")]:
                                    distance = base_distance_matrix[ele1][ele2] if pd.isnull(base_distance_matrix[ele2][ele1]) else base_distance_matrix[ele2][ele1]
                                    if local_min_distance == -1:
                                        local_min_distance = distance
                                    else:
                                        if distance < local_min_distance:
                                            local_min_distance = distance
                            distance_matrix[column][row] = local_min_distance
                            if min_distance == -1:
                                min_distance = local_min_distance
                                min_row = row
                                min_column = column
                            else:
                                if local_min_distance < min_distance:
                                    min_distance = local_min_distance
                                    min_row = row
                                    min_column = column
                        else:
                            if isinstance(column, int):
                                cluster_value = row
                                single_value = column
                            else:
                                cluster_value = column
                                single_value = row                        
                            local_min_distance = -1
                            for ele in [int(x) for x in cluster_value.split(",")]:
                                distance = base_distance_matrix[single_value][ele] if pd.isnull(base_distance_matrix[ele][single_value]) else base_distance_matrix[ele][single_value]
                                if local_min_distance == -1:
                                    local_min_distance = distance
                                else:
                                    if distance < local_min_distance:
                                        local_min_distance = distance
                            distance_matrix[column][row] = local_min_distance
                            if min_distance == -1:
                                min_distance = local_min_distance
                                min_row = row
                                min_column = column
                            else:
                                if local_min_distance < min_distance:
                                    min_distance = local_min_distance
                                    min_row = row
                                    min_column = column
    print(datetime.now())

    id_ground_truth_map = {}

    for index, id in enumerate(ids):
        id_ground_truth_map[id] = ground_truths[index]

    ground_truth_matrix = get_ground_truth_matrix(no_of_objects, id_ground_truth_map)
    
    clusters_selected = eval(cluster_no_ids_map[no_of_clusters])

    id_prediction_map = {}
    for index, ele in enumerate(clusters_selected):
        if isinstance(ele[0], int):
            id_prediction_map[ele[0]] = index + 1
        else:
            temp = ele[0].split(",")
            temp = [int(x) for x in temp]
            for val in temp:
                id_prediction_map[val] = index + 1

    prediction_matrix = get_prediction_matrix(no_of_objects, id_prediction_map)

    evaluate_clusters(no_of_objects, ground_truth_matrix, prediction_matrix)
    
    predictions = []
    for i in range(1, no_of_objects + 1):
        predictions.append(id_prediction_map[i])

    # get colors to represent labels
    colors_selected = random.sample(colors, len(set(predictions)))

    # assigning colors to labels
    cluster_no_color_map = {}
    cluster_colors = []
    for cluster_no in predictions:
        if cluster_no not in cluster_no_color_map:
            color = colors_selected.pop(0)
            cluster_no_color_map[cluster_no] = color
            cluster_colors.append(color)
        else:
            cluster_colors.append(cluster_no_color_map[cluster_no])

    # legends for scatter plot    
    patches = []
    index = 0
    for ele in cluster_no_color_map:
        index = index + 1
        patches.append(mpatches.Patch(color=cluster_no_color_map[ele], label=index))
        
    features = np.array(data.iloc[:,2:no_of_columns])
    features = [list(x) for x in features]
        
    pca = PCA(n_components=2)
    pca.fit(features)
    features = np.array(pca.transform(features))
    
    features_x = []
    features_y = []
    
    for ele in features:
        features_x.append(ele[0])
        features_y.append(ele[1])

    # show scatter plot    
    plt.figure(figsize=(8,6))
    plt.scatter(features_x, features_y, c=cluster_colors)
    plt.legend(handles=patches)
    plt.title("Plotting Heirarchical Agglomerative Clusters for "+filename)
    plt.show()


# ### Density Based Clustering

# In[17]:

filenames = ['cho.txt', 'iyer.txt']
#filenames = ['new_dataset_1.txt', 'new_dataset_2.txt']

epsilon = 1.2
min_pts = 2

file = open("Density_Based_Clusters.txt", "w")
file.write("Epsilon: "+str(epsilon)+" Min_Pts: "+str(min_pts)+"\n\n")
file.close()
                    
for filename in filenames:
    print("====>"+filename)
    print(datetime.now())
    file = open("Density_Based_Clusters.txt", "a")
    file.write("====>"+filename+"\n")
    file.close()
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_objects = len(data)
    no_of_columns = len(data.columns)
    no_of_attributes = no_of_columns - 2
    features = np.array(data.iloc[:,2:no_of_columns])
    features = [list(x) for x in features]
    ids = np.array(data.iloc[:,0:1])
    ids = [int(x) for x in ids]

    # id and feature mapping
    id_feature_map = {}
    
    # id and visited, added_to_cluster flag
    id_flags_map = {}
    
    ground_truths = np.array(data.iloc[:,1:2])
    ground_truths = [int(x) for x in ground_truths]
    
    clusters = []

    for index, id in enumerate(ids):
        # loc gives row with row_label as argument
        id_feature_map[id] = features[index]
        id_flags_map[id] = [False, False]
        
    noise = []
        
    for id in ids:
        cluster = []
        flags = id_flags_map[id]
        if flags[0] == False:
            id_flags_map[id] = [True, flags[1]]
            neighbors = getNeighborPts(id, epsilon)
            if len(neighbors) < min_pts:
                # add to noise
                noise.append(id)
            else:
                cluster = expandCluster(id, neighbors, epsilon, min_pts)
        if len(cluster) > 0:
            print("==>"+str(cluster))
            clusters.append(cluster)
    
    print(datetime.now())
    
    print("Objects classified as noise: "+str(len(noise)))
            
    if len(clusters) > 0:
        for cluster in clusters:
            file = open("Density_Based_Clusters.txt", "a")
            file.write("==>"+str(cluster))
            file.write("\n")
            file.close()

    id_ground_truth_map = {}

    for index, id in enumerate(ids):
        id_ground_truth_map[id] = ground_truths[index]

    ground_truth_matrix = get_ground_truth_matrix(no_of_objects, id_ground_truth_map)

    id_prediction_map = {}

    cluster_no = 1

    for cluster in clusters:
        for ele in cluster:
            id_prediction_map[ele] = cluster_no
        cluster_no = cluster_no + 1

    for ele in noise:
        id_prediction_map[ele] = -1

    prediction_matrix = get_prediction_matrix(no_of_objects, id_prediction_map)

    evaluate_clusters(no_of_objects, ground_truth_matrix, prediction_matrix)
    
    predictions = []
    for i in range(1, no_of_objects + 1):
        predictions.append(id_prediction_map[i])

    # get colors to represent labels
    colors_selected = random.sample(colors, len(set(predictions)))

    # assigning colors to labels
    cluster_no_color_map = {}
    cluster_colors = []
    for cluster_no in predictions:
        if cluster_no not in cluster_no_color_map:
            color = colors_selected.pop(0)
            cluster_no_color_map[cluster_no] = color
            cluster_colors.append(color)
        else:
            cluster_colors.append(cluster_no_color_map[cluster_no])

    # legends for scatter plot    
    patches = []
    index = 0
    for ele in cluster_no_color_map:
        index = index + 1
        patches.append(mpatches.Patch(color=cluster_no_color_map[ele], label="Noise" if ele == -1 else index))
        
    pca = PCA(n_components=2)
    pca.fit(features)
    features = np.array(pca.transform(features))
    
    features_x = []
    features_y = []
    
    for ele in features:
        features_x.append(ele[0])
        features_y.append(ele[1])

    # show scatter plot    
    plt.figure(figsize=(8,6))
    plt.scatter(features_x, features_y, c=cluster_colors)
    plt.legend(handles=patches)
    plt.title("Plotting Density Based Clusters for "+filename)
    plt.show()


# ### MapReduce K-Means

# In[9]:

filenames = ['cho.txt', 'iyer.txt']

for filename in filenames:    
    print("====>"+filename)
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_objects = len(data)
    no_of_columns = len(data.columns)
    no_of_attributes = no_of_columns - 2
    features = np.array(data.iloc[:,2:no_of_columns])
    features = [list(x) for x in features]
    ids = np.array(data.iloc[:,0:1])
    ids = [int(x) for x in ids]
    
    # id and feature mapping
    id_feature_map = {}
    
    for index, id in enumerate(ids):
        id_feature_map[id] = features[index]
        
    ground_truths = np.array(data.iloc[:,1:2])
    ground_truths = [int(x) for x in ground_truths]
    
    no_of_clusters = np.max(ground_truths)
    
    mapreduce_file = filename.replace(".txt","")+"_mapreduce.txt"
    mapreduce_output = pd.read_csv(mapreduce_file, delimiter="\t", header=None)
    centroids = np.array(mapreduce_output[0])
    for index, ele in enumerate(centroids):
        temp = ele.replace("[","")
        temp = temp.replace("]","")
        temp = temp.split(",")
        temp = [float(x) for x in temp]
        centroids[index] = temp
    
    print(datetime.now())
        
    # cluster_no and ids mapping
    clusters = {}
    for i in range(1, no_of_clusters+1):
        clusters[i] = []

    for id in ids:
        min_dist = -1
        cluster_selected = -1
        for centroid_index, centroid in enumerate(centroids):
            diff = np.subtract(id_feature_map[id], centroid)
            diff = [pow(x, 2) for x in diff]
            curr_dist = np.sum(diff)
            if min_dist == -1:
                min_dist = curr_dist
                cluster_selected = centroid_index + 1
            elif curr_dist < min_dist:
                min_dist = curr_dist
                cluster_selected = centroid_index + 1
        clusters[cluster_selected].append(id)
            
    print(datetime.now())
    print("==>Centroids")
    for ele in centroids:
        ele = [round(x, 2) for x in ele]
        print(ele)

    id_ground_truth_map = {}

    for index, id in enumerate(ids):
        id_ground_truth_map[id] = ground_truths[index]

    ground_truth_matrix = get_ground_truth_matrix(no_of_objects, id_ground_truth_map)

    id_prediction_map = {}

    for key in clusters:
        for ele in clusters[key]:
            id_prediction_map[ele] = key

    prediction_matrix = get_prediction_matrix(no_of_objects, id_prediction_map)

    evaluate_clusters(no_of_objects, ground_truth_matrix, prediction_matrix)
    
    predictions = []
    for i in range(1, no_of_objects + 1):
        predictions.append(id_prediction_map[i])

    # get colors to represent labels
    colors_selected = random.sample(colors, len(set(predictions)))

    # assigning colors to labels
    cluster_no_color_map = {}
    cluster_colors = []
    for cluster_no in predictions:
        if cluster_no not in cluster_no_color_map:
            color = colors_selected.pop(0)
            cluster_no_color_map[cluster_no] = color
            cluster_colors.append(color)
        else:
            cluster_colors.append(cluster_no_color_map[cluster_no])

    # legends for scatter plot    
    patches = []
    index = 0
    for ele in cluster_no_color_map:
        index = index + 1
        patches.append(mpatches.Patch(color=cluster_no_color_map[ele], label=index))
        
    pca = PCA(n_components=2)
    pca.fit(features)
    features = np.array(pca.transform(features))
    
    features_x = []
    features_y = []
    
    for ele in features:
        features_x.append(ele[0])
        features_y.append(ele[1])

    # show scatter plot    
    plt.figure(figsize=(8,6))
    plt.scatter(features_x, features_y, c=cluster_colors)
    plt.legend(handles=patches)
    plt.title("Plotting MapReduce K-Means for "+filename)
    plt.show()


# In[ ]:



