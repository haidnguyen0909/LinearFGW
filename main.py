import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sb

import time
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from dataio import structural_data_list, format_graph_data, k_barycenter_load
from algot import *
from trainsvm import *


DATA_GRAPH_DIR = "./data/"
name = "ENZYMES"
level = 3



# reading graph data-------------------------------------------------------

filename_pkl = os.path.join(DATA_GRAPH_DIR, name, 'processed_data.pkl')
graph_data, num_classes = structural_data_list(filename_pkl)
if len(graph_data[0])==4:
	dim_embedding = graph_data[0][2].shape[1]*level
else:
	dim_embedding = 1

print("dataset name:", name)
print("number of graphs:", len(graph_data))

# processing graph data-----------------------------------------------------
graphs =[]
preprocess = False
if name=="PROTEINS":
	# for PROTEINS, a WL iteration is needed to preprocess data.
	preprocess = True
for i, data in enumerate(graph_data):
	[adj, dist, features, label] = format_graph_data(data, preprocess)
	graphs.append([adj, dist, features, label])
	
np.random.shuffle(graphs)


print("number of processed graphs:",len(graphs))
#print(d_gw, tran.shape)



# learning barycenter (its size is set to the first graph's size)
size = graphs[0][0].shape[0]
print("size of barycenter (No. nodes):", size)

size_centers =[size]
ot_method ="ppa"
gamma =0.1
gwb_layers = 5
ot_layers = 5
n_iters = 3

filename = os.path.join(DATA_GRAPH_DIR, name, 'k_barycenter.pkl')
#alphas =[0.01, 0.05 ,0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
alphas =[0.1, 0.5, 0.9]

distances=[]
labels = []
for i in range(len(graphs)):
	labels.append(graphs[i][-1].item())
labels = np.array(labels)

for alpha in alphas:
	print("alpha=", alpha)
	centers, ps, embeddings = k_barycenter(graphs, size_centers, dim_embedding, ot_method, gamma, gwb_layers, ot_layers, n_iters, filename, alpha)
	#centers, ps, embeddings = k_barycenter_load(filename)
	centers = np.array(centers)
	ps = np.array(ps)
	embeddings = np.array(embeddings)

	centers = [torch.from_numpy(center).type(torch.FloatTensor) for center in centers]
	ps = [torch.from_numpy(p).type(torch.FloatTensor) for p in ps]
	if embeddings[0] is not None:
		embeddings = [torch.from_numpy(embedding).type(torch.FloatTensor) for embedding in embeddings]

	G2Fs = fusedGW_featurize(graphs, centers, ps, embeddings, alpha)
	n = len(graphs)
	distance = np.zeros((n,n))
	for i in range(n):
		for j in range(i, n):
			val = euclidean(G2Fs[i], G2Fs[j], alpha)
			distance[i, j] = val
			distance[j, i] = val
	distances.append(distance)
cv_acc=internalCV_mp(distances, labels, 10)
print("Accuracy =", ":",cv_acc, np.mean(cv_acc))
			








