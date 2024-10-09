import os 
import json
import random
import numpy as np
import pandas as pd

def get_cleaned_data(raw_data_path):
    ''' Drop the idx without image'''
    
    with open(raw_data_path, 'r') as file:
        raw_data = json.load(file)
    cleaned_data = {key: value for key, value in raw_data.items() if value['image'] != None}
    return cleaned_data

def non_iid_split_dirichlet(raw_data, train, topics, ideal_counts, n_clients, alpha=0.4):
    '''iid split if alpha is large; non-iid split if alpha is small'''
    
    # Split raw data idx into n_clients 
    client = []
    clients = {}
    total_samples = len(raw_data)
    idx_train = [value['id'] for value in train]
    
    if ideal_counts < 1:
        err = f'[SIMULATE] Decrease the number of participating clients (`n_clients` < {n_clients})!'
        raise Exception(err)
    
    #=============== Split the idx of the raw data by topics,and drop marginal topics =============#
    idx_raw_data_split = {}
    for value in topics:
        idx_raw_data_split[value] = []

    for key, value in raw_data.items():
        idx_raw_data_split[value['topic']].append(key)
        
    # Sort the topics of raw_data by length of its value
    idx_raw_data_split = dict(sorted(idx_raw_data_split.items(),
                                    key=lambda item: len(item[1])))

    # Delete the value with length less than 100
    # idx_raw_data_split = {key: value for key, value in idx_raw_data_split.items() if len(value) > 100}
    for key in list(idx_raw_data_split.keys()):
        if len(idx_raw_data_split[key]) < 100:
            del idx_raw_data_split[key]
            if key in topics:
                topics.remove(key)
            
    n_classes = len(topics) # number of different classes 
    #================ Split the raw_data into n_clients ===============#
    for k in range(n_clients):
        ### define Dirichlet distribution of which prior distribution is an uniform distribution
        diri_prior = np.random.uniform(size=n_classes)
        
        ### sample a parameter corresponded to that of categorical distribution
        cat_param = np.random.dirichlet(alpha * diri_prior)

        ### try to sample by amount of `ideal_counts``
        sampled = np.random.choice(n_classes, ideal_counts, p=cat_param)

        ### count per-class samples
        unique, counts = np.unique(sampled, return_counts=True)
        
        for idx, selected_class in enumerate(unique):
            selected_class = int(selected_class)
            selected_topic = topics[idx]
            
            if counts[idx] == 0: continue
            ### avoid the samples selected are more than the original samples
            samples_size_of_this_class = len(idx_raw_data_split[topics[idx]])
            if counts[idx] > samples_size_of_this_class:
                counts[idx] = samples_size_of_this_class
            
            # Get the samples and drop samples don't belong to idx_train
            selected_samples = random.sample(idx_raw_data_split[selected_topic], counts[idx])
            # Element from selected_class is str, element from idx_train is str
            idx_train = [int(idx) for idx in idx_train]
            selected_samples = [int(idx) for idx in selected_samples]
            for idx in selected_samples:
                if idx not in idx_train:
                    selected_samples.remove(idx)
            # selected_samples = [idx for idx in selected_class if idx in idx_train]
            
            for idx in selected_samples:  
                for sample in train:
                    if sample['id'] == str(idx):   
                        client.append(sample)
                        break
                
        # get client k
        clients[str(k)] = client
        client = []
    
    return clients

import os
def save_clients_json(saving_folder, clients):
    os.makedirs(saving_folder, exist_ok=True)
    
    for key, value in clients.items():
        client_name = 'client_' + key + '.json'
        saving_path = os.path.join(saving_folder, client_name)
        print(f'The saving paths are:\n{saving_path}')
        with open(saving_path, 'w') as file:
            json.dump(value, file)
            
train_path = '/Users/shuogudaojin/ScienceQA/raw_data/llava_train_QCM-LEA.json'
raw_data_path = '/Users/shuogudaojin/ScienceQA/raw_data/problems.json'
saving_folder = '/Users/shuogudaojin/ScienceQA/new_iid_split_1'
cleaned_data = get_cleaned_data(raw_data_path)

with open(train_path, 'r') as file:
    train = json.load(file)
    
# Drop the idx without image
cleaned_data = get_cleaned_data(raw_data_path)

# Get all of the topics
raw_data_pandas = pd.read_json(raw_data_path)
topics = raw_data_pandas.loc['topic']
topics = pd.unique(topics)
topics = topics.tolist()

#=============== Set the parameter to split clients ==================#
clients = non_iid_split_dirichlet(raw_data=cleaned_data, train=train, topics=topics,
                                  ideal_counts=1000, n_clients=10, alpha=100)
save_clients_json(saving_folder, clients)
    

cleaned_data = get_cleaned_data(raw_data_path)