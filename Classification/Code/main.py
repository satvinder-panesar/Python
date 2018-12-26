
# coding: utf-8

# ## Satvinder Singh 
# ### Person No: 5024 8888
# ### Project 3: Classification Algorithms

# In[1]:

import pandas as pd, numpy as np, statistics as stats, math, scipy.stats, random
from sklearn.model_selection import train_test_split
from collections import OrderedDict

def evaluate(actual_labels, predictions):
    a = 0
    b = 0
    c = 0
    d = 0
    for index, prediction in enumerate(predictions):
        if prediction == actual_labels[index] and prediction == 0:
            d = d + 1
        elif prediction == actual_labels[index] and prediction == 1:
            a = a + 1
        elif prediction == 0 and actual_labels[index] == 1:
            b = b + 1
        elif prediction == 1 and actual_labels[index] == 0:
            c = c + 1      
    
    accuracy = round((a+d)/(a+b+c+d), 3)
    miss_classification_rate = round((b+c)/(a+b+c+d), 3)
    
    if a+c > 0:
        precision = round(a/(a+c), 3)
    else:
        precision = 0
    if a+b > 0:
        recall = round(a/(a+b), 3)
    else:
        recall = 0
    if a+b+c > 0:
        f_measure = round((2*a)/(2*a+b+c), 3)
    else:
        f_measure = 0

    return accuracy, precision, recall, f_measure, miss_classification_rate
    
def get_entropy(data):
    if len(data) == 0:
        return 0
    no_of_columns = len(data[0])
    no_of_objects = len(data)
    no_of_ones = len([x for x in data if x[no_of_columns-1] == 1])
    no_of_zeros = no_of_objects - no_of_ones
    prob_of_one = no_of_ones / no_of_objects
    prob_of_zero = no_of_zeros / no_of_objects
    if prob_of_one == 0:
        return round(-(prob_of_zero * math.log(prob_of_zero, 2)), 3)
    elif prob_of_zero == 0:
        return round(-(prob_of_one * math.log(prob_of_one, 2)), 3)
    else:
        return round(-(prob_of_one * math.log(prob_of_one, 2))-(prob_of_zero * math.log(prob_of_zero, 2)), 3)

def build_tree(train_data, data_entropy, columns_with_categorical_data, id, tree_map):
    no_of_columns = len(train_data[0])
    unique_labels = list(set(train_data[:,no_of_columns-1]))
    if len(unique_labels) == 1:
        # print("all labels are same")
        tree_map[id]=unique_labels[0]
        return
    attr_selected = -1
    threshold_of_attr = -1
    max_info_gained = -1
    for i in range(0, no_of_columns - 1):      
        temp = get_threshold(train_data[:,[i, no_of_columns - 1]], data_entropy, i, columns_with_categorical_data).split(":")
        temp = [float(x) for x in temp]
        info_gained = temp[0]
        threshold_value = temp[1]
        if max_info_gained == -1:
            max_info_gained = info_gained
            attr_selected = i
            threshold_of_attr = threshold_value
        elif info_gained > max_info_gained:
            max_info_gained = info_gained
            attr_selected = i
            threshold_of_attr = threshold_value
    # attribute selected is zero based
    # print("Attribute selected: "+str(attr_selected)+"(0-based) Threshold: "+str(threshold_of_attr))
    if attr_selected in columns_with_categorical_data:
        left_set = np.array([x for x in train_data if x[attr_selected] == threshold_of_attr])
        right_set = np.array([x for x in train_data if x[attr_selected] != threshold_of_attr])
        if len(left_set) == 0:
            try:
                tree_map[id] = stats.mode(right_set[:,len(right_set[0])-1])
            except:
                tree_map[id] = random.choice(right_set[:,len(right_set[0])-1])
        elif len(right_set) == 0:
            try:
                tree_map[id] = stats.mode(left_set[:,len(left_set[0])-1])
            except:
                tree_map[id] = random.choice(left_set[:,len(left_set[0])-1])
        else:
            tree_map[id] = [attr_selected, threshold_of_attr]
            build_tree(left_set, data_entropy, columns_with_categorical_data, id+"1", tree_map)
            build_tree(right_set, data_entropy, columns_with_categorical_data, id+"2", tree_map)
    else:
        left_set = np.array([x for x in train_data if x[attr_selected] < threshold_of_attr])
        right_set = np.array([x for x in train_data if x[attr_selected] >= threshold_of_attr])
        if len(left_set) == 0:
            try:
                tree_map[id] = stats.mode(right_set[:,len(right_set[0])-1])
            except:
                tree_map[id] = random.choice(right_set[:,len(right_set[0])-1])
        elif len(right_set) == 0:
            try:
                tree_map[id] = stats.mode(left_set[:,len(left_set[0])-1])
            except:
                tree_map[id] = random.choice(left_set[:,len(left_set[0])-1])
        else:
            tree_map[id] = [attr_selected, threshold_of_attr]
            build_tree(left_set, data_entropy, columns_with_categorical_data, id+"1", tree_map)
            build_tree(right_set, data_entropy, columns_with_categorical_data, id+"2", tree_map)
    return tree_map

def build_tree_2(train_data, no_of_features_to_use, data_entropy, columns_with_categorical_data, id, tree_map, columns_selected_map):
    
    while True:
        
        no_of_columns = len(train_data[0])

        # select columns randomly from train data
        columns_selected = random.sample(range(0, no_of_columns - 1), no_of_features_to_use)

        local_columns_with_categorical_data = [x for x in columns_with_categorical_data if x in columns_selected]
        # append label column
        columns_selected.append(no_of_columns - 1)

        # get selected columns from train data
        local_train_data = train_data[:, columns_selected]

        data_entropy = get_entropy(local_train_data)

        unique_labels = list(set(local_train_data[:,len(local_train_data[0])-1]))

        if len(unique_labels) == 1:
            tree_map[id]=unique_labels[0]
            return

        # if training samples are small or height of tree > number of features
        if len(local_train_data) <= 3 or len(id) > no_of_columns:
            try:
                tree_map[id] = stats.mode(local_train_data[:,len(local_train_data[0])-1])
            except:
                tree_map[id] = random.choice(local_train_data[:,len(local_train_data[0])-1])
            return

        attr_selected = -1
        threshold_of_attr = -1
        max_info_gained = -1

        no_of_columns = len(local_train_data[0])

        for i in range(0, no_of_columns - 1):      
            temp = get_threshold(local_train_data[:,[i, no_of_columns - 1]], data_entropy, i, local_columns_with_categorical_data).split(":")
            temp = [float(x) for x in temp]
            info_gained = temp[0]
            threshold_value = temp[1]
            if max_info_gained == -1:
                max_info_gained = info_gained
                attr_selected = i
                threshold_of_attr = threshold_value
            elif info_gained > max_info_gained:
                max_info_gained = info_gained
                attr_selected = i
                threshold_of_attr = threshold_value
        # attribute selected is zero based
        attr_selected = columns_selected[attr_selected]
        # print("Attribute selected: "+str(attr_selected)+"(0-based) Threshold: "+str(threshold_of_attr))
        
        if attr_selected not in columns_selected_map:
            columns_selected_map[attr_selected] = str(threshold_of_attr)
            break
            
        if attr_selected in columns_selected_map:
            if attr_selected not in columns_with_categorical_data:
                if ":" not in columns_selected_map[attr_selected]:
                    if threshold_of_attr != float(columns_selected_map[attr_selected]):
                        columns_selected_map[attr_selected] = columns_selected_map[attr_selected] + ":" + str(threshold_of_attr)
                        break
                else:
                    temp = columns_selected_map[attr_selected].split(":")
                    temp = [float(x) for x in temp]
                    found = False
                    for ele in temp:
                        if ele == threshold_of_attr:
                            found = True
                            break
                    if found == False:
                        columns_selected_map[attr_selected] = columns_selected_map[attr_selected] + ":" + str(threshold_of_attr)
                        break
                
    if attr_selected in local_columns_with_categorical_data:
        left_set = np.array([x for x in train_data if x[attr_selected] == threshold_of_attr])
        right_set = np.array([x for x in train_data if x[attr_selected] != threshold_of_attr])
        if len(left_set) == 0:
            try:
                tree_map[id] = stats.mode(right_set[:,len(right_set[0])-1])
            except:
                tree_map[id] = random.choice(right_set[:,len(right_set[0])-1])
        elif len(right_set) == 0:
            try:
                tree_map[id] = stats.mode(left_set[:,len(left_set[0])-1])
            except:
                tree_map[id] = random.choice(left_set[:,len(left_set[0])-1])
        else:
            tree_map[id] = [attr_selected, threshold_of_attr]
            build_tree_2(left_set, no_of_features_to_use, get_entropy(left_set), columns_with_categorical_data, id+"1", tree_map, columns_selected_map)
            build_tree_2(right_set, no_of_features_to_use, get_entropy(right_set), columns_with_categorical_data, id+"2", tree_map, columns_selected_map)
    else:
        left_set = np.array([x for x in train_data if x[attr_selected] < threshold_of_attr])
        right_set = np.array([x for x in train_data if x[attr_selected] >= threshold_of_attr])
        if len(left_set) == 0:
            try:
                tree_map[id] = stats.mode(right_set[:,len(right_set[0])-1])
            except:
                tree_map[id] = random.choice(right_set[:,len(right_set[0])-1])
        elif len(right_set) == 0:
            try:
                tree_map[id] = stats.mode(left_set[:,len(left_set[0])-1])
            except:
                tree_map[id] = random.choice(left_set[:,len(left_set[0])-1])
        else:
            tree_map[id] = [attr_selected, threshold_of_attr]
            build_tree_2(left_set, no_of_features_to_use, get_entropy(left_set), columns_with_categorical_data, id+"1", tree_map, columns_selected_map)
            build_tree_2(right_set, no_of_features_to_use, get_entropy(right_set), columns_with_categorical_data, id+"2", tree_map, columns_selected_map)
    return tree_map

def get_threshold(column, data_entropy, column_index, columns_with_categorical_data):
    # column = [column label]
    no_of_objects = len(column)
    max_info_gained = -1
    threshold_value = -1
    unique_elements = list(set(column[:,0]))
    if column_index in columns_with_categorical_data:
        if len(unique_elements) > 1:
            for unique_element in unique_elements:
                div1 = [x for x in column if x[0] == unique_element]
                div2 = [x for x in column if x[0] != unique_element]
                entropy_div1 = get_entropy(div1)
                entropy_div2 = get_entropy(div2)
                info = (len(div1)/no_of_objects*entropy_div1) + (len(div2)/no_of_objects*entropy_div2)
                info_gained = data_entropy - info
                if max_info_gained == -1:
                    max_info_gained = info_gained
                    threshold_value = unique_element
                elif info_gained > max_info_gained:
                    max_info_gained = info_gained
                    threshold_value = unique_element
            return str(max_info_gained)+":"+str(threshold_value)
        else:
            entropy_div1 = get_entropy(column)
            info = (len(column)/no_of_objects*entropy_div1)
            info_gained = data_entropy - info
            return str(info_gained)+":"+str(column[:,0][0])
    else:        
        for unique_element in unique_elements:
            div1 = [x for x in column if x[0] < unique_element]
            div2 = [x for x in column if x[0] >= unique_element]
            entropy_div1 = get_entropy(div1)
            entropy_div2 = get_entropy(div2)
            info = (len(div1)/no_of_objects*entropy_div1) + (len(div2)/no_of_objects*entropy_div2)
            info_gained = data_entropy - info
            if max_info_gained == -1:
                max_info_gained = info_gained
                threshold_value = unique_element
            elif info_gained > max_info_gained:
                max_info_gained = info_gained
                threshold_value = unique_element
        return str(max_info_gained)+":"+str(threshold_value)

def predict_using_decision_tree(data, tree_map, columns_with_categorical_data):
    #getting root node of decision tree
    key = "1."
    root = tree_map[key]
    attribute = root[0]
    threshold = root[1]
    while True:
        if attribute in columns_with_categorical_data:
            if data[attribute] == threshold:
                # getting left node
                key = key + "1"
                left_node = tree_map[key]
                if not isinstance(left_node, list):
                    return(left_node)
                else:
                    attribute = left_node[0]
                    threshold = left_node[1]
            else:
                # getting right node
                key = key + "2"
                right_node = tree_map[key]
                if not isinstance(right_node, list):
                    return(right_node)
                else:
                    attribute = right_node[0]
                    threshold = right_node[1]                    
        else:
            if data[attribute] < threshold:
                # getting left node
                key = key + "1"
                left_node = tree_map[key]
                if not isinstance(left_node, list):
                    return(left_node)
                else:
                    attribute = left_node[0]
                    threshold = left_node[1]
            else:
                # getting right node
                key = key + "2"
                right_node = tree_map[key]
                if not isinstance(right_node, list):
                    return(right_node)
                else:
                    attribute = right_node[0]
                    threshold = right_node[1]
                    
def print_decision_tree(tree_map, columns_with_categorical_data, string_to_number_map):
    tree_copy = tree_map.copy()
    for key in tree_copy:
        if(isinstance(tree_copy[key], list)):
            value = tree_copy[key]
            if value[0] in columns_with_categorical_data:
                temp = [x for (x, y) in string_to_number_map.items() if y == value[1]]
                tree_copy[key] = [value[0], temp]
    print(tree_copy)


# ### Nearest Neighbors

# In[12]:

filenames = ['project3_dataset1.txt', 'project3_dataset2.txt']

for filename in filenames: 
    print("====>"+filename)
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_objects = len(data)
    no_of_columns = len(data.columns)
    features = np.array(data.iloc[:,0:no_of_columns])
    
    string_to_number_map = {}
    
    columns_with_categorical_data = []
    
    # pre processing
    for ele_index, ele in enumerate(features):
        for val_index, val in enumerate(ele):
            try:
                isinstance(float(val), float)
            except:
                if val_index not in columns_with_categorical_data:
                    columns_with_categorical_data.append(val_index)
                if val in string_to_number_map:
                    ele[val_index] = string_to_number_map[val]
                else:
                    length = len(string_to_number_map.keys())
                    ele[val_index] = length
                    string_to_number_map[val] = length
        features[ele_index] = ele
    features = np.array([[float(y) for y in x] for x in features])
    
    # normalizing data
    means = {}
    std_devs = {}
    
    for i in range(0, no_of_columns - 1):
        if i not in columns_with_categorical_data:
            means[i] = np.mean(features[:,i])
            std_devs[i] = np.std(features[:,i])
            
    for feature_index, feature in enumerate(features):
        for attr_index, ele in enumerate(feature):
            if attr_index not in columns_with_categorical_data and attr_index < no_of_columns - 1:
                temp = (ele - means[attr_index])/std_devs[attr_index]
                feature[attr_index] = temp
            features[feature_index] = feature
    
    no_of_folds = 10
    feature_sets = np.array_split(features, no_of_folds)
    
    k = 5
    
    acc_accuracy = 0
    acc_precision = 0
    acc_recall = 0
    acc_f_measure = 0
    
    for i in range(0, no_of_folds):
        
        print("Fold: "+str(i+1))    
        test_data = feature_sets[i]
        temp = [x for x in range(0,10) if x !=i]
        train_data = feature_sets[temp[0]]
        for i in temp[1:len(temp)]:
            train_data = np.concatenate((train_data, feature_sets[i]))
        
        train_labels = train_data[:,no_of_columns-1]
        test_labels = test_data[:,no_of_columns-1]
        # removing labels from train and test data
        train_data = [x[0:no_of_columns-1] for x in train_data]
        test_data = [x[0:no_of_columns-1] for x in test_data]
        
        # predictions
        predictions = []
        for feature in test_data:
            # stores distance-index of nearest neighbors for each test data
            distance_index_map = {}
            for feature_index, another_feature in enumerate(train_data):
                distance = np.subtract(feature, another_feature)
                distance = [pow(x, 2) for x in distance]
                distance = math.sqrt(sum(distance))
                if len(distance_index_map) < k and distance not in distance_index_map:
                    distance_index_map[distance] = feature_index
                    if len(distance_index_map) == k:
                        distance_index_map = OrderedDict(sorted(distance_index_map.items()))
                elif len(distance_index_map) >= k and (distance < list(distance_index_map.keys())[0] or distance < list(distance_index_map.keys())[len(distance_index_map) - 1]):
                    # deleting last value from dict
                    del distance_index_map[list(distance_index_map.keys())[len(distance_index_map)-1]]
                    # adding new value
                    distance_index_map[distance] = feature_index
                    distance_index_map = OrderedDict(sorted(distance_index_map.items()))
            nearest_neighbors = []
            for key in distance_index_map:
                nearest_neighbors.append(distance_index_map[key])
            votes = [train_labels[x] for x in nearest_neighbors]
            try:
                predictions.append(stats.mode(votes))
            except:
                # >=2 values are repeated equal number of times
                # print("tie")
                vote_count_map = {}
                for vote in votes:
                    if vote not in vote_count_map:
                        vote_count_map[vote] = 1
                    else:
                        vote_count_map[vote] = vote_count_map[vote] + 1
                if len(vote_count_map) == 2:
                    # 2 values are repeated, so get nearest value
                    predictions.append(votes[0])
                else:
                    # sorted dict based on values
                    print("Tie: more than 2 values are repeated equal number of times")

        accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(test_labels, predictions)
        print("Accuracy:"+str(accuracy)+"\tPrecision: "+str(precision)+"\tRecall: "+str(recall)+"\tF-measure: "+str(f_measure))
        acc_accuracy += accuracy
        acc_precision += precision
        acc_recall += recall
        acc_f_measure += f_measure
        
    print("==>Average metric values:")
    print("Accuracy:"+str(round(acc_accuracy/no_of_folds, 3)))
    print("Precision: "+str(round(acc_precision/no_of_folds, 3)))
    print("Recall: "+str(round(acc_recall/no_of_folds, 3)))
    print("F-measure: "+str(round(acc_f_measure/no_of_folds, 3)))


# ### Decision Trees

# In[15]:

filenames = ['project3_dataset1.txt', 'project3_dataset2.txt']
    
for filename in filenames: 
    print("====>"+filename)
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_columns = len(data.columns)
    features = np.array(data.iloc[:,0:no_of_columns])
    features = np.array([list(x) for x in features])
    
    string_to_number_map = {}
    columns_with_categorical_data = []
    
    # pre processing
    for ele_index, ele in enumerate(features):
        for val_index, val in enumerate(ele):
            try:
                isinstance(float(val), float)
            except:
                if val_index not in columns_with_categorical_data:
                    columns_with_categorical_data.append(val_index)
                if val in string_to_number_map:
                    ele[val_index] = string_to_number_map[val]
                else:
                    length = len(string_to_number_map.keys())
                    ele[val_index] = length
                    string_to_number_map[val] = length
        features[ele_index] = ele
    features = np.array([[float(y) for y in x] for x in features])
    
    no_of_folds = 10
    feature_sets = np.array_split(features, no_of_folds)
    
    acc_accuracy = 0
    acc_precision = 0
    acc_recall = 0
    acc_f_measure = 0
    
    for i in range(0, no_of_folds):
        
        print("Fold: "+str(i+1))    
        test_data = feature_sets[i]
        temp = [x for x in range(0,10) if x !=i]
        train_data = feature_sets[temp[0]]
        for i in temp[1:len(temp)]:
            train_data = np.concatenate((train_data, feature_sets[i]))
        
        train_labels = train_data[:,no_of_columns-1]
        test_labels = test_data[:,no_of_columns-1]
        # removing labels from test data
        test_data = [x[0:no_of_columns-1] for x in test_data]
        data_entropy = get_entropy(train_data)
        
        # construct decision tree
        tree_map = build_tree(train_data, data_entropy, columns_with_categorical_data, "1.", {})
        
        predictions = []
                
        local_test_data = [x[0:no_of_columns-1] for x in train_data]
        for data in local_test_data:
                predictions.append(predict_using_decision_tree(data, tree_map, columns_with_categorical_data))

        accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(train_labels, predictions)

        if int(accuracy) == 1: 
            # post pruning
            while True:            
                leaf_nodes = [x for x in tree_map if tree_map[x] == 1 or tree_map[x] == 0]
                level = [len(x) for x in leaf_nodes]
                max_level = level.index(max(level))
                node_to_prune = leaf_nodes[max_level]
                parent_node = node_to_prune[0:len(node_to_prune)-1]
                if parent_node == "1.":
                    break
                leaf_node_value = tree_map[node_to_prune]
                del tree_map[node_to_prune]
                tree_map[parent_node] = leaf_node_value

                predictions = []

                for data in local_test_data:
                    predictions.append(predict_using_decision_tree(data, tree_map, columns_with_categorical_data))

                accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(train_labels, predictions)

                if accuracy < 1:
                    break
                    
        # print_decision_tree(tree_map, columns_with_categorical_data, string_to_number_map)
        
        predictions = []
        
        for data in test_data:
            predictions.append(predict_using_decision_tree(data, tree_map, columns_with_categorical_data))

        accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(test_labels, predictions)
        print("Accuracy:"+str(accuracy)+"\tPrecision: "+str(precision)+"\tRecall: "+str(recall)+"\tF-measure: "+str(f_measure))
        acc_accuracy += accuracy
        acc_precision += precision
        acc_recall += recall
        acc_f_measure += f_measure
        
    print("==>Average metric values:")
    print("Accuracy:"+str(round(acc_accuracy/no_of_folds, 3)))
    print("Precision: "+str(round(acc_precision/no_of_folds, 3)))
    print("Recall: "+str(round(acc_recall/no_of_folds, 3)))
    print("F-measure: "+str(round(acc_f_measure/no_of_folds, 3)))


# ### Naive Bayes

# In[12]:

filenames = ['project3_dataset1.txt', 'project3_dataset2.txt']

for filename in filenames: 
    print("====>"+filename)
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_columns = len(data.columns)
    features = np.array(data.iloc[:,0:no_of_columns])
    features = np.array([list(x) for x in features])
    
    string_to_number_map = {}    
    columns_with_categorical_data = []
    
    # pre processing
    for ele_index, ele in enumerate(features):
        for val_index, val in enumerate(ele):
            try:
                isinstance(float(val), float)
            except:
                if val_index not in columns_with_categorical_data:
                    columns_with_categorical_data.append(val_index)
                if val in string_to_number_map:
                    ele[val_index] = string_to_number_map[val]
                else:
                    length = len(string_to_number_map.keys())
                    ele[val_index] = length
                    string_to_number_map[val] = length
        features[ele_index] = ele
    features = np.array([[float(y) for y in x] for x in features])
            
    no_of_folds = 10
    feature_sets = np.array_split(features, no_of_folds)
    
    acc_accuracy = 0
    acc_precision = 0
    acc_recall = 0
    acc_f_measure = 0
    
    for i in range(0, no_of_folds):
        
        print("Fold: "+str(i+1))    
        test_data = feature_sets[i]
        temp = [x for x in range(0,10) if x !=i]
        train_data = feature_sets[temp[0]]
        for i in temp[1:len(temp)]:
            train_data = np.concatenate((train_data, feature_sets[i]))        
        no_of_objects = len(train_data)
        
        train_labels = train_data[:,no_of_columns-1]
        train_labels = [float(x) for x in train_labels]
        test_labels = test_data[:,no_of_columns-1]
        test_labels = [float(x) for x in test_labels]
        # removing labels from test data
        test_data = [x[0:no_of_columns-1] for x in test_data]
        
        # stores probability of 0 and 1
        class_prior_probabilities = []        
        
        for ele in set(train_labels):
            class_prior_probabilities.append(len([x for x in train_labels if x == ele])/len(train_labels))
            
        labels = list(set(train_labels))
        labels = [int(x) for x in labels]
        labels_count = []
        
        for label in labels:
            labels_count.append(len([x for x in train_labels if x == label]))
    
        # stores [mean, std] for 0 and 1 of all columns
        mean_std_dev = []
        for i in range(0, len(train_data[0]) -1):
            temp = []
            for label in labels:
                data = [x[i] for x in train_data if x[-1] == label]
                temp.append([np.mean(data), np.std(data)])
            mean_std_dev.append(temp)

        predictions = []
        
        for data in test_data:
            final_probabilities = []
            intermediate_probabilities = []
            # descriptor prior
            den = no_of_objects
            for ele_index, ele in enumerate(data):
                corrected_using_laplacian = 0
                num = len([x for x in train_data if x[ele_index] == ele])
                if num == 0:
                    # laplacian correction
                    num = 1
                    corrected_using_laplacian = 1
                intermediate_probabilities.append(num/(den + corrected_using_laplacian))
            descriptor_prior_probability = np.product(intermediate_probabilities)
            for label_index, label in enumerate(labels):
                den = labels_count[label]
                intermediate_probabilities = []
                class_prior_probability = class_prior_probabilities[label]
                den = labels_count[label]
                for ele_index, ele in enumerate(data):
                    if ele_index not in columns_with_categorical_data:
                        temp = mean_std_dev[ele_index]
                        temp = temp[label_index]
                        # representing random variables as a normal distibution and using pdf
                        intermediate_probabilities.append(scipy.stats.norm(temp[0], temp[1]).pdf(ele))
                    else:
                        num = len([x for x in train_data if x[ele_index] == ele and x[-1] == label])
                        intermediate_probabilities.append(num/den)
                posterior_probability = np.product(intermediate_probabilities)
                final_probabilities.append((posterior_probability * class_prior_probability)/descriptor_prior_probability)
                index = final_probabilities.index(max(final_probabilities))
            predictions.append(labels[index])
            
        accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(test_labels, predictions)
        print("Accuracy:"+str(accuracy)+"\tPrecision: "+str(precision)+"\tRecall: "+str(recall)+"\tF-measure: "+str(f_measure))
        acc_accuracy += accuracy
        acc_precision += precision
        acc_recall += recall
        acc_f_measure += f_measure
    
    print("==>Average metric values:")
    print("Accuracy:"+str(round(acc_accuracy/no_of_folds, 3)))
    print("Precision: "+str(round(acc_precision/no_of_folds, 3)))
    print("Recall: "+str(round(acc_recall/no_of_folds, 3)))
    print("F-measure: "+str(round(acc_f_measure/no_of_folds, 3)))                


# ### Random Forest

# In[5]:

filenames = ['project3_dataset1.txt', 'project3_dataset2.txt']
    
for filename in filenames: 
    print("====>"+filename)
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_columns = len(data.columns)
    features = np.array(data.iloc[:,0:no_of_columns])
    features = np.array([list(x) for x in features])
    
    string_to_number_map = {}
    columns_with_categorical_data = []
    
    # pre processing
    for ele_index, ele in enumerate(features):
        for val_index, val in enumerate(ele):
            try:
                isinstance(float(val), float)
            except:
                if val_index not in columns_with_categorical_data:
                    columns_with_categorical_data.append(val_index)
                if val in string_to_number_map:
                    ele[val_index] = string_to_number_map[val]
                else:
                    length = len(string_to_number_map.keys())
                    ele[val_index] = length
                    string_to_number_map[val] = length
        features[ele_index] = ele
    features = np.array([[float(y) for y in x] for x in features])
    
    no_of_folds = 10
    feature_sets = np.array_split(features, no_of_folds)
    
    acc_accuracy = 0
    acc_precision = 0
    acc_recall = 0
    acc_f_measure = 0
    
    no_of_decision_trees = 5
    no_of_features_to_use = math.ceil(0.2 * (no_of_columns - 1))
    
    for i in range(0, no_of_folds):
        
        print("Fold: "+str(i+1))    
        test_data = feature_sets[i]
        temp = [x for x in range(0,10) if x !=i]
        train_data = feature_sets[temp[0]]
        for i in temp[1:len(temp)]:
            train_data = np.concatenate((train_data, feature_sets[i]))
        
        test_labels = test_data[:,no_of_columns-1]
        # removing labels from test data
        test_data = [x[0:no_of_columns-1] for x in test_data]
        
        trees = []
            
        for i in range(0, no_of_decision_trees):
            
            min_miss_classification_rate = -1
            best_tree_map = {}
            
            data = []
            
            # sampling with replacement
            indices = np.random.choice(len(train_data), int(len(train_data) / no_of_decision_trees), replace = True)
            for index in indices:
                data.append(train_data[index])
            
            temp = np.array_split(data, 3)
            local_train_data = np.concatenate((temp[0], temp[1]))
            local_test_data = temp[2]
            
            local_test_labels = local_test_data[:,no_of_columns-1]
            # removing labels from local test data
            local_test_data = [x[0:no_of_columns-1] for x in local_test_data]
            
            # repeat n times to get best columns set
            for iteration in range(0, 20):
            
                # construct decision tree
                tree_map = build_tree_2(local_train_data, no_of_features_to_use, get_entropy(local_train_data), columns_with_categorical_data, "1.", {}, {})
                
                predictions = []

                for data in local_test_data:
                    predictions.append(predict_using_decision_tree(data, tree_map, columns_with_categorical_data))

                accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(local_test_labels, predictions)
                
                if min_miss_classification_rate == -1:
                    min_miss_classification_rate = miss_classification_rate
                    best_tree_map = tree_map.copy()
                else:
                    if miss_classification_rate < min_miss_classification_rate:
                        min_miss_classification_rate = miss_classification_rate
                        best_tree_map = tree_map.copy()
                
            trees.append(best_tree_map)
        
        predictions = []
        
        for data in test_data:
            different_predictions = []
            for i in range(0, no_of_decision_trees):
                different_predictions.append(predict_using_decision_tree(data, trees[i], columns_with_categorical_data))
            predictions.append(stats.mode(different_predictions))

        accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(test_labels, predictions)
        print("Accuracy:"+str(accuracy)+"\tPrecision: "+str(precision)+"\tRecall: "+str(recall)+"\tF-measure: "+str(f_measure))
        acc_accuracy += accuracy
        acc_precision += precision
        acc_recall += recall
        acc_f_measure += f_measure
        
    print("==>Average metric values:")
    print("Accuracy:"+str(round(acc_accuracy/no_of_folds, 3)))
    print("Precision: "+str(round(acc_precision/no_of_folds, 3)))
    print("Recall: "+str(round(acc_recall/no_of_folds, 3)))
    print("F-measure: "+str(round(acc_f_measure/no_of_folds, 3)))


# ### Boosting

# In[6]:

filenames = ['project3_dataset1.txt', 'project3_dataset2.txt']
    
for filename in filenames: 
    print("====>"+filename)
    data = pd.read_csv(filename, delimiter="\t", header=None)
    no_of_columns = len(data.columns)
    features = np.array(data.iloc[:,0:no_of_columns])
    features = np.array([list(x) for x in features])
    
    string_to_number_map = {}
    columns_with_categorical_data = []
    
    # pre processing
    for ele_index, ele in enumerate(features):
        for val_index, val in enumerate(ele):
            try:
                isinstance(float(val), float)
            except:
                if val_index not in columns_with_categorical_data:
                    columns_with_categorical_data.append(val_index)
                if val in string_to_number_map:
                    ele[val_index] = string_to_number_map[val]
                else:
                    length = len(string_to_number_map.keys())
                    ele[val_index] = length
                    string_to_number_map[val] = length
        features[ele_index] = ele
    features = np.array([[float(y) for y in x] for x in features])
    
    no_of_folds = 10
    feature_sets = np.array_split(features, no_of_folds)
    
    acc_accuracy = 0
    acc_precision = 0
    acc_recall = 0
    acc_f_measure = 0
    
    no_of_decision_trees = 5
    
    for i in range(0, no_of_folds):
        
        print("Fold: "+str(i+1))    
        test_data = feature_sets[i]
        temp = [x for x in range(0,10) if x !=i]
        train_data = feature_sets[temp[0]]
        for i in temp[1:len(temp)]:
            train_data = np.concatenate((train_data, feature_sets[i]))
        
        test_labels = test_data[:,no_of_columns-1]
        # removing labels from test data
        test_data = [x[0:no_of_columns-1] for x in test_data]
        
        trees = []
        alpha_values = []
        
        weights = []
        weight = 1/len(train_data)
        for i in range(0, len(train_data)):
            weights.append(weight)
            
        for i in range(0, no_of_decision_trees):
            
            while True:

                local_train_data = []
                # sampling with replacement
                indices = list(np.random.choice(len(train_data), int(len(train_data) / no_of_decision_trees), replace = True, p = weights))
                for index in indices:
                    local_train_data.append(train_data[index])
                local_train_data = np.array(local_train_data)
                local_test_data = [x[0:no_of_columns-1] for x in local_train_data]
                local_train_labels = local_train_data[:,no_of_columns-1]
                
                tree_map = build_tree(local_train_data, get_entropy(local_train_data), columns_with_categorical_data, "1.", {})
                
                # only root and two children then repeat because we cant prune anything
                if len(tree_map) == 3:
                    continue
                
                predictions = []
                
                for data in local_test_data:
                        predictions.append(predict_using_decision_tree(data, tree_map, columns_with_categorical_data))
                        
                accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(local_train_labels, predictions)
                
                if int(accuracy) == 1: 
                    # post pruning
                    while True:            
                        leaf_nodes = [x for x in tree_map if tree_map[x] == 1 or tree_map[x] == 0]
                        level = [len(x) for x in leaf_nodes]
                        max_level = level.index(max(level))
                        node_to_prune = leaf_nodes[max_level]
                        parent_node = node_to_prune[0:len(node_to_prune)-1]
                        if parent_node == "1.":
                            break
                        leaf_node_value = tree_map[node_to_prune]
                        del tree_map[node_to_prune]
                        tree_map[parent_node] = leaf_node_value

                        predictions = []

                        for data in local_train_data:
                            predictions.append(predict_using_decision_tree(data, tree_map, columns_with_categorical_data))

                        accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(local_train_labels, predictions)

                        if accuracy < 1:
                            break
                
                error = 0
                
                # calculating error
                for index, weight in enumerate(weights):
                    if index in indices:
                        position = indices.index(index)
                        if predictions[position] != local_train_labels[position]:
                            error = error + weights[index]
                # total of all weights is 1, so no need to divide anything
                
                if error <= 0.5:
                    break
            
            trees.append(tree_map)
            alpha = (1/2) * math.log((1-error)/error)
            alpha_values.append(alpha)
            
            # updating weights
            for index, weight in enumerate(weights):
                if index in indices:
                    position = indices.index(index)
                    if predictions[position] != local_train_labels[position]:
                        weights[index] = weights[index] * math.sqrt((1-error)/error)
                    else:
                        weights[index] = weights[index] * math.sqrt(error/(1-error))
            
            # normalizing weights
            total = np.sum(weights)
            weights = [x/total for x in weights]
                
        predictions = []
        
        for data in test_data:
            prediction_weight_map = {}
            for i in range(0, no_of_decision_trees):
                prediction = predict_using_decision_tree(data, trees[i], columns_with_categorical_data)
                if prediction in prediction_weight_map:
                    prediction_weight_map[prediction] = prediction_weight_map[prediction] + alpha_values[i]
                else:
                    prediction_weight_map[prediction] = alpha_values[i]
            max_weight = -1
            prediction = -1
            for key in prediction_weight_map:
                weight = prediction_weight_map[key]
                if max_weight == -1:
                    max_weight = weight
                    prediction = key
                else:
                    if weight > max_weight:
                        max_weight = weight
                        prediction = key
            predictions.append(prediction)

        accuracy, precision, recall, f_measure, miss_classification_rate = evaluate(test_labels, predictions)
        print("Accuracy:"+str(accuracy)+"\tPrecision: "+str(precision)+"\tRecall: "+str(recall)+"\tF-measure: "+str(f_measure))
        acc_accuracy += accuracy
        acc_precision += precision
        acc_recall += recall
        acc_f_measure += f_measure
        
    print("==>Average metric values:")
    print("Accuracy:"+str(round(acc_accuracy/no_of_folds, 3)))
    print("Precision: "+str(round(acc_precision/no_of_folds, 3)))
    print("Recall: "+str(round(acc_recall/no_of_folds, 3)))
    print("F-measure: "+str(round(acc_f_measure/no_of_folds, 3)))


# In[ ]:



