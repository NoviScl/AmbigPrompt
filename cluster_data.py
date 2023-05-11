import json 
import random 
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import os
from sklearn.cluster import KMeans
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='0'
random.seed(2023)

def split_data(cluster1, cluster2):
    data_1_1 = []
    data_0_0 = []
    data_1_0 = []
    data_0_1 = []

    for d in cluster1:
        if d["answer"] == "1":
            data_1_0.append(d)
        else:
            data_0_0.append(d)
    
    for d in cluster2:
        if d["answer"] == "1":
            data_1_1.append(d)
        else:
            data_0_1.append(d)
    
    random.shuffle(data_1_1)
    random.shuffle(data_0_0)

    dataset = {}
    dataset["demos_1_1"] = data_1_1[:32]
    dataset["demos_0_0"] = data_0_0[:32]
    dataset["testset_1_1"] = data_1_1[32:82]
    dataset["testset_0_0"] = data_0_0[32:82]
    dataset["testset_1_0"] = data_1_0[:120]
    dataset["testset_0_1"] = data_0_1[:120]
    return dataset

# with open("testsets_ambiguous/sa_sentiment_domain.json", "r") as f:
#     data = json.load(f)

with open("testsets_ambiguous/comments_toxicity_uppercase.json", "r") as f:
    data = json.load(f)

data = data["demos_1_1"] + data["demos_0_0"] + data["testset_1_1"] + data["testset_1_0"] + data["testset_0_1"] + data["testset_0_0"]
print (len(data))

# embeddings = np.load("sa_sentiment_domain_embeddings.npy")
embeddings = np.load("comments_toxicity_uppercase_embeddings.npy")
print (embeddings.shape)

kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(embeddings)

clusters = {}
for i in range(20):
    clusters[i] = []

for i, d in enumerate(data):
    clusters[kmeans.labels_[i]].append(d)

for i in range(20):
    print (i, len(clusters[i]))

## sample 10 pairs of clusters
for i in range(20):
    cluster1 = i
    cluster1_data = clusters[cluster1]
    # cluster2 = random.choice(list(clusters.keys()))
    # if cluster1 == cluster2:
    #     continue
    
    cluster_remaining_data = []
    for j in range(20):
        if j != cluster1:
            cluster_remaining_data += clusters[j]
    
    dataset = split_data(cluster1_data, cluster_remaining_data)
    if len(dataset["demos_1_1"]) < 32 or len(dataset["demos_0_0"]) < 32:
        continue
    if len(dataset["testset_1_0"]) < 120 or len(dataset["testset_0_1"]) < 120:
        continue
    print ("cluster{}".format(cluster1))
    print (len(dataset["demos_1_1"]), len(dataset["demos_0_0"]), len(dataset["testset_1_1"]), len(dataset["testset_0_0"]), len(dataset["testset_1_0"]), len(dataset["testset_0_1"]))
    with open("testsets_ambiguous_unknown/civilcomments_cluster{}_vs_others.json".format(cluster1), "w") as f:
        json.dump(dataset, f, indent=4)



# with open("testsets_ambiguous/comments_toxicity_uppercase.json", "r") as f:
#     data = json.load(f)

# data = data["demos_1_1"] + data["demos_0_0"] + data["testset_1_1"] + data["testset_1_0"] + data["testset_0_1"] + data["testset_0_0"]
# print (len(data))

# tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
# model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

# ## load on GPU 
# model = model.cuda()

# texts = [d["question"] for d in data]
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# # Get the embeddings
# all_embeddings = []
# with torch.no_grad():
#     for i in tqdm(range(0, len(inputs["input_ids"]), 100)):
#         print (i)
#         batch = {k: v[i:i+100] for k, v in inputs.items()}
#         batch = {k: v.cuda() for k, v in batch.items()}
#         embeddings = model(**batch, output_hidden_states=True, return_dict=True).pooler_output
#         ## convert to np array 
#         embeddings = embeddings.cpu().detach().numpy()
#         all_embeddings.append(embeddings)
    
# ## save all embeddings to a file
# all_embeddings = np.concatenate(all_embeddings, axis=0)
# np.save("comments_toxicity_uppercase_embeddings.npy", all_embeddings)
