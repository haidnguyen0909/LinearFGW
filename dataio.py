

import numpy as np
import pickle
import torch
import networkx as nx


def normalize_adj_tensor(adj, sparse=False):
    device = torch.device("cuda" if adj.is_cuda else "cpu")
    if sparse:
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx 

def update_feature(A, feature):
	A_n = normalize_adj_tensor(A)
	return A_n @ feature

def format_graph_data(data, preprocess):
	if len(data) == 4:
		edges = data[0]
		num_nodes = data[1]
		features = data[2]
		label = data[3]

		adj = np.zeros((num_nodes, num_nodes))
		dist = np.ones((num_nodes, 1))
		dist /= np.sum(dist)

		for edge in edges:
			src = edge[0]
			dst = edge[1]
			adj[src, dst] = 1.0
		



		features = torch.from_numpy(features).type(torch.FloatTensor)
		if preprocess:
			features = update_feature(adj, features)
		dist = torch.from_numpy(dist).type(torch.FloatTensor)
		adj = torch.from_numpy(adj).type(torch.FloatTensor)
		label = torch.LongTensor([label])


		features_1 = update_feature(adj, features)
		features_2 = update_feature(adj, features_1)
		features_3 = update_feature(adj, features_2)
		features_4 = update_feature(adj, features_3)
		features_5 = torch.cat([features, features_1, features_2], dim=1)
		#maxvalue = torch.max(torch.abs(features_5))
		#print(features_5)

		#for n in range(features.shape[0]):
			#features_5[n] = features_5[n] * 1/torch.sqrt(torch.sum(features_5[n]*features_5[n]))
			#features_5[n] = features_5[n] * 1/maxvalue
		#exit(1)
		return [adj, dist, features_5, label]
	else:
		edges = data[0]
		num_nodes = data[1]
		label = data[2]
		features = None
		adj = np.zeros((num_nodes, num_nodes))
		dist = np.ones((num_nodes, 1))
		dist /= np.sum(dist)

		for edge in edges:
			src = edge[0]
			dst = edge[1]
			adj[src, dst] = 1.0
		dist = torch.from_numpy(dist).type(torch.FloatTensor)
		adj = torch.from_numpy(adj).type(torch.FloatTensor)
		label = torch.LongTensor([label])
		return [adj, dist, features,label]


def k_barycenter_load(pkl_path):
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f)
		centers = data[0]
		ps = data[1]
		embeddings=data[2]
	centers = np.array(centers)
	ps = np.array(ps)
	embeddings = np.array(embeddings)

	centers = [torch.from_numpy(center).type(torch.FloatTensor) for center in centers]
	ps = [torch.from_numpy(p).type(torch.FloatTensor) for p in ps]
	if embeddings[0] is not None:
		embeddings = [torch.from_numpy(embedding).type(torch.FloatTensor) for embedding in embeddings]

	return centers, ps, embeddings

def structural_data_list(pkl_path):
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f)
	if len(data) == 5:
		graph2edge =data[0]
		graph2size=data[1]
		graph2labels = data[2]
		graph2feature = data[3]
		num_class = data[4]
		graph_data =[]
		for i in range(len(graph2size)):
			graph_data.append([graph2edge[i], graph2size[i], np.array(graph2feature[i]), graph2labels[i]])
	else:
		graph2edge=data[0]
		graph2size=data[1]
		graph2labels=data[2]
		num_class = data[3]
		graph_data =[]
		for i in range(len(graph2size)):
			graph_data.append([graph2edge[i], graph2size[i], graph2labels[i]])
	return graph_data, num_class






