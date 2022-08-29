import torch
import numpy as np
import pickle
import ot
from ot.gromov import gromov_wasserstein, fused_gromov_wasserstein


ALPHA = 0.0

def ot_fgw_lib(cost_s, cost_t, p_s, p_t, ot_method, gamma, num_layer, emb_s, emb_t, alpha):

	w1 = ot.unif(cost_s.shape[0])
	w2 = ot.unif(cost_t.shape[0])

	if emb_s is not None and emb_t is not None:
		tmp1 = 2 * emb_s @ torch.t(emb_t)
		tmp2 = ((emb_s ** 2) @ torch.ones(emb_s.size(1), 1)).repeat(1, cost_t.size(0))
		tmp3 = ((emb_t ** 2) @ torch.ones(emb_t.size(1), 1)).repeat(1, cost_s.size(0))
		cost2 = (tmp2 + torch.t(tmp3) - tmp1)
	
	cost_s_ = cost_s.detach().numpy()
	cost_t_ = cost_t.detach().numpy()
	cost2_ = cost2.detach().numpy()
	#p_s = p_s.detach().numpy()
	#p_t = p_t.detach().numpy()
	#Gg, log = gromov_wasserstein(cost_s, cost_t, w1, w2, loss_fun='square_loss', verbose=True, log=True)
	tran = fused_gromov_wasserstein(cost2_, cost_s_, cost_t_, w1, w2, loss_fun='square_loss', alpha=1-alpha, verbose=False, log=False)
	tran = torch.tensor(tran)

	d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t, alpha) * tran).sum()
	return d_gw, tran




def cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t, alpha):
	f1_st = ((cost_s**2)@p_s).repeat(1, tran.size(1))
	f2_st = (torch.t(p_t)@torch.t(cost_t**2)).repeat(tran.size(0), 1)
	cost_st = f1_st + f2_st
	cost1 = (cost_st - 2 * cost_s @ tran @ torch.t(cost_t))

	if emb_s is not None and emb_t is not None:
		#tmp1 = emb_s @ torch.t(emb_t)
		#tmp2 = torch.sqrt((emb_s ** 2) @ torch.ones(emb_s.size(1), 1))
		#tmp3 = torch.sqrt((emb_t ** 2) @ torch.ones(emb_t.size(1), 1))
		#cost2 = 0.5 * (1 - tmp1 / (tmp2 @ torch.t(tmp3)))
		#cost1 * (1 - alpha) + alpha*cost2
		tmp1 = 2 * emb_s @ torch.t(emb_t)
		tmp2 = ((emb_s ** 2) @ torch.ones(emb_s.size(1), 1)).repeat(1, tran.size(1))
		tmp3 = ((emb_t ** 2) @ torch.ones(emb_t.size(1), 1)).repeat(1, tran.size(0))
		cost2 = (tmp2 + torch.t(tmp3) - tmp1) / (emb_s.size(1)**2)
		#print(cost2, torch.max(cost2), torch.max(emb_s), torch.max(emb_t), torch.min(emb_s), torch.min(emb_t))
		return cost1 * (1 - alpha) + alpha*cost2
	#exit(1)
	return cost1



def ot_badmm(cost_s, cost_t, p_s, p_t, tran, dual, gamma, num_layer):
	all1_s = torch.ones(p_s.size())
	all1_t = torch.ones(p_t.size())
	for m in range(num_layer):
		kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
		b = p_t / (torch.t(kernel_a) @ all1_s)
		aux = (all1_s @ torch.t(b)) * kernel_a
		dual = dual + gamma * (tran - aux)
		kernel_t = torch.exp(-(cost_mat(cost_s, cost_t, p_s, p_t, aux) + dual) / gamma) * aux
		a = p_s / (kernel_t @ all1_t)
		tran = (a @ torch.t(all1_t)) * kernel_t
	return tran, dual


def ot_ppa(cost_s, cost_t, p_s, p_t, tran, dual, gamma, num_layer, sinkhorn_iters):
	for m in range(num_layer):
		kernel = torch.exp(-cost_mat(cost_s, cost_t, p_s, p_t, tran)/gamma) * tran
		b = p_t / (torch.t(kernel) @ dual)
		for i in range(sinkhorn_iters):
			dual = p_s / (kernel @ b)
			b = p_t / (torch.t(kernel) @ dual)
		tran = (dual @ torch.t(b)) * kernel
	return tran, dual

def ot_fgw(cost_s, cost_t, p_s, p_t, ot_method, gamma, num_layer, emb_s, emb_t, alpha):
	#tran = p_s @ torch.t(p_t)
	tran = torch.abs(torch.randn(cost_s.shape[0],cost_t.shape[0]))
	tran = tran/torch.sum(tran)
	if ot_method == 'ppa':
		dual = torch.ones(p_s.size()) / p_s.size(0)
		for m in range(num_layer):
			cost = cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t, alpha)
			# cost /= torch.max(cost)
			kernel = torch.exp(-cost / gamma) * tran
			b = p_t / (torch.t(kernel) @ dual)
			for i in range(5):
			    dual = p_s / (kernel @ b)
			    b = p_t / (torch.t(kernel) @ dual)
			tran = (dual @ torch.t(b)) * kernel
			d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t, alpha) * tran).sum()
			#print(m, kernel, d_gw)
	elif ot_method == 'b-admm':
		all1_s = torch.ones(p_s.size())
		all1_t = torch.ones(p_t.size())
		dual = torch.zeros(p_s.size(0), p_t.size(0))
		for m in range(num_layer):
			kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
			b = p_t / (torch.t(kernel_a) @ all1_s)
			aux = (all1_s @ torch.t(b)) * kernel_a
			dual = dual + gamma * (tran - aux)
			cost = cost_mat(cost_s, cost_t, p_s, p_t, aux, emb_s, emb_t, alpha)
			# cost /= torch.max(cost)
			kernel_t = torch.exp(-(cost + dual) / gamma) * aux
			a = p_s / (kernel_t @ all1_t)
			tran = (a @ torch.t(all1_t)) * kernel_t
	d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t, alpha) * tran).sum()
	return d_gw, tran


def barycenter(graphs, init_barycenter, dim_embedding, ot_method, gamma, gwb_layers, ot_layers, alpha):
	graph_b = init_barycenter[0]
	p_b = init_barycenter[1]
	emb_b = init_barycenter[2]

	tmp1 = p_b @ torch.t(p_b)
	tmp2 = p_b @ torch.ones(1, dim_embedding)
	weight = 1/len(graphs)

	noembedding = False
	if graphs[0][2] is None:
		noembedding = True


	for n in range(gwb_layers):
		graph_b_tmp =0
		emb_b_tmp =0
		trans =[]
		for k in range(len(graphs)):
			graph_k = graphs[k][0]
			emb_k = graphs[k][2]
			p_k = graphs[k][1]
			_, tran_k = ot_fgw(cost_s =graph_k, cost_t =graph_b, p_s=p_k, p_t=p_b, ot_method="ppa", gamma=gamma, num_layer=10, emb_s=emb_k, emb_t= emb_b, alpha=alpha)
			trans.append(tran_k)
			if np.isnan(torch.sum(tran_k).item()):
				print(n, k)
				print("Nan", graph_k, graph_b)
				print("Nan", emb_k, emb_b)
				print("Nan",tran_k)
				exit(1)

			graph_b_tmp += weight * (torch.t(tran_k) @ graph_k @ tran_k)
			if noembedding is False:
				emb_b_tmp += weight * (torch.t(tran_k) @ emb_k)
		graph_b = graph_b_tmp/tmp1
		if noembedding:
			return [graph_b, p_b, None]
		else:
			#print(tmp2.shape, emb_b_tmp.shape)
			emb_b = emb_b_tmp/tmp2
			return [graph_b, p_b, emb_b]



def normalize_km(km):
    n = len(km)
    for i in range(n):
        if km[i,i] == 0:
            km[i,i] = 1.0/100000
    return km / np.array(np.sqrt(np.mat(np.diag(km)).T * np.mat(np.diag(km))))


def euclidean(graph_i, graph_j, alpha):
	W_feats_i = graph_i[0]
	W_feats_j = graph_j[0]
	GW_feats_i = graph_i[1]
	GW_feats_j = graph_j[1]
	K = len(W_feats_i)

	W_d =0.0
	GW_d =0.0
	for k in range(K):
		if W_feats_i[k] is not None:
			W_k_i = W_feats_i[k]
			W_k_j = W_feats_j[k]
			W_d += torch.sum((W_k_i - W_k_j)**2)

		GW_k_i = GW_feats_i[k]
		GW_k_j = GW_feats_j[k]
		GW_d += torch.sum((GW_k_i - GW_k_j)**2)
	return W_d * alpha + GW_d * (1 - alpha)

def linear_kernel(graph_i, graph_j):
	W_feats_i = graph_i[0]
	W_feats_j = graph_j[0]
	#GW_feats_i = graph_i[1]
	#GW_feats_j = graph_j[1]
	K = len(W_feats_i)

	W_K =0.0
	GW_K =0.0
	for k in range(K):
		W_k_i = W_feats_i[k]
		W_k_j = W_feats_j[k]
		W_K += torch.sum(W_k_i * W_k_j)

		#GW_k_i = GW_feats_i[k]
		#GW_k_j = GW_feats_j[k]
		#GW_K += torch.sum(GW_k_i * GW_k_j)
	return W_K

def fusedGW_featurize(graphs, centers, ps, embeddings, alpha):
	#print(len(graphs), len(centers), len(ps), len(embeddings))
	K = len(centers)
	G2Fs =[]
	#for k in range(K):
	#	print(k, centers[k].shape, embeddings[k].shape, ps[k].shape)
	for i, graph in enumerate(graphs):
		#print("featuring graph:", i)
		center_i, p_i, emb_i, label = graph[0], graph[1], graph[2], graph[3]
		
		W_feats =[]
		GW_feats =[]

		for k in range(K):
			center_k = centers[k]
			p_k = ps[k]
			sig_k = center_k.shape[0]
			emb_k =embeddings[k]
			_, tran_ki= ot_fgw(cost_s =center_k, cost_t =center_i, p_s=p_k, p_t=p_i, ot_method="ppa", gamma=0.1, num_layer=10, emb_s=emb_k, emb_t=emb_i, alpha=alpha)
			#_, tran_ki= ot_fgw_lib(cost_s =center_k, cost_t =center_i, p_s=p_k, p_t=p_i, ot_method="ppa", gamma=0.1, num_layer=10, emb_s=emb_k, emb_t=emb_i, alpha=alpha)
			#print(tran_ki)

			if emb_i is not None:
				W_feat = sig_k * tran_ki @ emb_i
			else:
				W_feat = None
			W_feats.append(W_feat)

			GW_feat =sig_k * sig_k* tran_ki @ center_i @ torch.t(tran_ki)
			#print(GW_feat[:10,:10])
			#exit(1)
			GW_feats.append(GW_feat)
			
			
		G2Fs.append([W_feats, GW_feats, label])
	return G2Fs

def vectorize(GFs, alpha):
	vects =[]
	labels =[]
	for i, GF in enumerate(GFs):
		W = GF[0]
		GW = GF[1]
		label = GF[2]
		#W = torch.reshape(W, (-1,))
		#GW = torch.reshape(GW, (-1,))
		labels.append(label)
		
		flat_Ws =[]
		flat_GWs=[]
		for k in range(len(W)):
			if W[k] is not None:
				W_k = W[k]
				W_k = torch.reshape(W_k, (-1,))
			else:
				W_k = None
			flat_Ws.append(W_k)
			GW_k = GW[k]
			GW_k = torch.reshape(GW_k, (-1,))
			flat_GWs.append(GW_k)

		if flat_Ws[0] is not None:
			flat_Ws = torch.cat(flat_Ws)
		flat_GWs = torch.cat(flat_GWs)
		if flat_Ws[0] is not None:
			fused = torch.cat([np.sqrt(alpha) * flat_Ws, np.sqrt(1-alpha)*flat_GWs])
		else:
			fused = flat_GWs
		vects.append(fused)
	return torch.stack(vects,dim=0), np.array(labels)





def linearFusedGWDistance(graphs, centers, ps, embeddings):
	# compute trans
	K = len(centers)
	n = len(graphs)
	T =[]
	for i in range(len(graphs)):
		G_i = graphs[i]
		center_i, p_i, emb_i, label_i = G_i[0], G_i[1], G_i[2], G_i[3]
		T_i = []
		for k in range(K):
			center_k = centers[k]
			p_k = ps[k]
			emb_k =embeddings[k]
			
			_, tran_ki= ot_fgw(cost_s =center_k, cost_t =center_i, p_s=p_k, p_t=p_i, ot_method="ppa", gamma=0.1, num_layer=5, emb_s=emb_k, emb_t=emb_i)
			T_i.append(tran_ki)
		T.append(T_i)

	dist_matrix =np.zeros((n,n))
	for i in range(n):
		for j in range(i+1, n):
			#print(i, j, "dkm")
			val = linearFusedGWpair(graphs[i], graphs[j], centers, ps, embeddings, T[i], T[j])
			dist_matrix[i, j] = val
			dist_matrix[j, i] = dist_matrix[i, j]
			#print(i, j, dist_matrix[i, j], val)
	return dist_matrix




def linearFusedGWpair(G_i, G_j, centers, ps, embeddings, T_i, T_j):
	K = len(centers)
	center_i, p_i, emb_i, label_i = G_i[0], G_i[1], G_i[2], G_i[3]
	center_j, p_j, emb_j, label_j = G_j[0], G_j[1], G_j[2], G_j[3]
	total_dist = 0.0
	
	for k in range(K):
		center_k = centers[k]
		p_k = ps[k]
		emb_k =embeddings[k]
		tran_ki = T_i[k]# F
		tran_kj = T_j[k]# G

		size_k = center_k.shape[0]
		size_i = center_i.shape[0]
		size_j = center_j.shape[0]

		A1 = torch.sum(emb_i * emb_i)/(size_i * size_k)
		B1 = torch.sum(emb_j * emb_j)/(size_j * size_k)

		
		C1 = torch.sum(tran_ki @ (emb_i @ torch.t(emb_j)) @ torch.t(tran_kj))
		wdist = A1 + B1 - 2 *C1


		A2 = torch.sum(tran_ki @ (center_i*center_i) @ torch.t(tran_ki))
		B2 = torch.sum(tran_kj @ (center_j*center_j) @ torch.t(tran_kj))
		C2 = torch.sum((tran_ki @ center_i @ torch.t(tran_ki)) * ((tran_kj @ center_j @ torch.t(tran_kj))))
		gwdist = A2 + B2 - 2 *C2
		total_dist += (ALPHA * wdist + (1-ALPHA) *gwdist)
		#print(k, A1, B1, C1, wdist,"shit")
	return total_dist/K



		

def k_barycenter(graphs, size_centers, dim_embedding, ot_method, gamma, gwb_layers, ot_layers, n_iters, filename, alpha):
	num_centers = len(size_centers)

	# init
	ps =[]
	centers =[]
	embeddings =[]

	for k in range(num_centers):
		center = torch.sigmoid(torch.randn(size_centers[k], size_centers[k]))
		#center= graphs[1][0]
		#embedding = torch.zeros(size_centers[k], dim_embedding)
	
		embedding = graphs[0][2]
		dist = torch.ones(size_centers[k], 1)/size_centers[k]
		ps.append(dist)
		centers.append(center)
		embeddings.append(embedding)


	for it in range(n_iters):
		total_distace =0.0
		center2sample ={}
		for i in range(len(graphs)):
			gi = graphs[i]
			d_gws =[]
			for k in range(num_centers):
				#print(centers[k])
				d_gw, tran = ot_fgw(cost_s =gi[0], cost_t =centers[k], p_s=gi[1], p_t=ps[k], ot_method="ppa", gamma=gamma, num_layer=10, emb_s=gi[2], emb_t=embeddings[k], alpha=alpha)
				
				d_gws.append(d_gw.item())
			min_idx = np.argmin(np.array(d_gws))
			
			if np.isnan(d_gws[min_idx]):
				print("Nan")
				exit(1)
				continue
			total_distace += d_gws[min_idx]
			if min_idx not in center2sample.keys():
				center2sample[min_idx] =[]
				center2sample[min_idx].append(i)
			else:
				center2sample[min_idx].append(i)
			#print(i, min_idx, d_gws[min_idx], graphs[i][-1])
		print(alpha,it, total_distace)
		#print(centers[0], embeddings[0])
		

		# barycenter
		for k in range(num_centers):
			if k not in center2sample.keys():
				continue
			samples_k = [graphs[i] for i in center2sample[k]]
			[graph_b, p_b, emb_b] = barycenter(samples_k, [centers[k], ps[k], embeddings[k]], dim_embedding, ot_method, gamma, gwb_layers, ot_layers, alpha)
			centers[k] = graph_b
			ps[k] = p_b
			embeddings[k] = emb_b


	s_centers =[]
	s_ps=[]
	s_embeddings =[]
	for k in range(num_centers):
		center = centers[k].numpy().tolist()
		p = ps[k].numpy().tolist()
		if embeddings[k] is not None:
			embedding = embeddings[k].numpy().tolist()
		s_centers.append(center)
		s_embeddings.append(embedding)
		s_ps.append(p)

	with open(filename, 'wb') as f:
		pickle.dump([s_centers, s_ps, s_embeddings], f)
	return s_centers, s_ps, s_embeddings

