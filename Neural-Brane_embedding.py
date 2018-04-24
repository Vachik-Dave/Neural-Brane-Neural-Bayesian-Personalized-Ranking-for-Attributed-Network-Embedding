"""

Neural-Brane: End to end implmentation:
inputs: Network file, attribute file [emb_dimensions, hidden_neurons]
output: Embedding file


Author: Vachik Dave.
"""


import sys
import numpy as np
import random
import my_NueralBPR_gpu
import datetime
import argparse


def separate_adj_wei(adj_lists):
	'''
	For each node separate adjacency list and weights also generate corresponding weight based probability
	'''
	new_adj = {};
	for key in adj_lists:
		s = adj_lists[key];
		total = 0.0;
		neighbors = [];
		weights = [];
		for each in s:
			neighbors.append(each[0]);
			weights.append(each[1]);
			total += each[1];

		probs = [ each/total for each in weights];
		new_adj[key] = [neighbors,probs];

	return new_adj;

def load_data(filename):
	'''
	Read network file
	'''
	N = 0;
	M = 0;
	adj_lists = {}
	f = open(filename);
	index = 0;
	for line in f:
		l = line.strip().split(" ")
		if index == 0:
			N = int(l[0]);
			M = int(l[1]);
			index += 1;
			continue;
		u = int(l[0]);
		v = int(l[1]);
		w = 1.0;
		try:
			w = float(l[2]);
		except IndexError:
			pass;
		try:
			t = adj_lists[u];
			t.add((v,w));
			adj_lists[u] = t;
		except KeyError:
			t = set([]);
			t.add((v,w));
			adj_lists[u] = t;

		try:
			t = adj_lists[v];
			t.add((u,w));
			adj_lists[v] = t;
		except KeyError:
			t = set([]);
			t.add((u,w));
			adj_lists[v] = t;

	adj_lists = separate_adj_wei(adj_lists);

	return N,M,adj_lists;

def get_degree_dist(adj_lists):
	'''
	gnerate degree based probability for all nodes
	'''
	node_ids = [];
	prob = [];
	total = 0.0;
	for key in adj_lists:
		node_ids.append(key);
		d = float(len(adj_lists[key][0]));
		val = pow(d,0.75);
		prob.append(val);
		total += val;

	for i in range(len(prob)):
		prob[i] = prob[i]/total;

	return node_ids,prob;


def load_feat_data(filename):
	'''
	Read node attribute file
	'''
	node_feat = {}
	f = open(filename);
	index = 0;
	max_feat_id = 0;
	prev_u = -1;
	for line in f:
		l = line.strip().split(" ")
		u = 0;
		try:
			u = int(l[0]);
		except ValueError:
			print "prev index = "+str(index);
			print "prev node_id = "+str(prev_u);
			print line;
			sys.exit();
		feat_list = [];
		for i in range(1,len(l)):
			f = map(int,l[i].strip().split(":"));
  			feat_list.append(f[0]);
			if max_feat_id < f[0]:
				max_feat_id = f[0];

		node_feat[u] = feat_list;
#		if u == 5408:
#			print str(u) + " " +str(feat_list);
		prev_u = u;
		index += 1;
	print "max_feat_id: "+str(max_feat_id);
	return max_feat_id, node_feat;

def save_embedding(filename,embedding):
	f = open(filename,'w');
	N,dim = np.array(embedding).shape;
	f.write(str(N)+" "+str(dim));
	index = 0;
	for row in embedding:
		f.write("\n"+str(index));
		index += 1;
		for ele in row:
			f.write(" "+str(ele));
	f.close();

def convert_nodeAtt(node_feat,emb,N):
	'''
	Get attribute based node embedding for each node from Attribute embedding matrix (P)
	Similar to embedding layer task
	'''
	index = 0;
	# assuming in nodeAtt file node_ids are in order
	new_emb = [];
	emb_dim = len(emb[0]);
	for i in range(N):
		feat_list = [];
		try:
			feat_list = node_feat[i];
		except KeyError:
			a = 0;

		att_emb_row = np.zeros(emb_dim);
		for i in range(len(feat_list)):
			idx = feat_list[i];
			for j in range(len(emb[idx])):
				if att_emb_row[j] < emb[idx][j]:
					att_emb_row[j] = emb[idx][j];
		new_emb.append(att_emb_row);
		#if len(line) > 1:
		#	new_emb.append( np.divide(att_emb_row,float(len(line)-1) ) );
		#else:
		#	new_emb.append(att_emb_row);
		index += 1;

	return new_emb;

def parse_args():
	'''
	Parses the input arguments.
	'''
	parser = argparse.ArgumentParser(description="Run Neural-Brane.")
	parser.add_argument('-input_graph', nargs='?', help='Input network path')
	parser.add_argument('-node_att', nargs='?', help='Node attribute file path')
	parser.add_argument('-dimensions', type=int, default=150, help='Number of dimensions. (default = 150).')
	parser.add_argument('-num_neurons', type=int, default=150, help='Number of neurons in hidden layer. (default = 150).')
	parser.add_argument('-num_epoch', type=int, default=50, help='Number of echops. (default = 50).')
	parser.add_argument('-learning_rate', type=float, default=0.05, help='Learning rate (default = 0.05).')
	parser.add_argument('-regularation_rate', type=float, default=0.0001, help='Regularation rate (default = 0.0001).')
def main():

	start_t = datetime.datetime.now()
	network_filename = sys.argv[1];
	feat_filename = sys.argv[2];

	N,M,adj_lists = load_data(network_filename);
	print "# of nodes: "+str(N);
	node_ids,prob = get_degree_dist(adj_lists);

	max_feat_id, node_feat = load_feat_data(feat_filename);

	hidden_dim = int(sys.argv[3]);
	num_neurons = int(sys.argv[4]);

#	num_epoch = 25;
	num_epoch = 30;
#	num_epoch = int(sys.argv[5]);
#	num_iter = 5000;
#	batch_size = 100;
	num_iter = 1000;
	batch_size = 100;
	learning_rate = 0.5;
#	learning_rate = float(sys.argv[6]);
	regulation_rate = 0.00005;

	print "learning rate:" +str(learning_rate);
	print "num_epoch:" + str(num_epoch);

	
	start_t = datetime.datetime.now();

	# only used for linked prediction to avoid training on test instances
	test_pos = {};
	node_emb, att_emb = my_NueralBPR_gpu.main(hidden_dim,num_epoch,num_iter,batch_size,N,adj_lists,test_pos, learning_rate, regulation_rate, num_neurons, max_feat_id, node_feat,node_ids,prob)

	print "node_emb shape: " + str(np.array(node_emb).shape)

	print "att_emb shape: " + str(np.array(att_emb).shape);

	node_att_emb = convert_nodeAtt(node_feat,att_emb,N);

	print "node_att_emb shape: " + str(np.array(node_att_emb).shape);

	node_emb_combined = np.hstack((node_emb[:-1],node_att_emb));
	
	print "node_emb_combined shape: " + str(np.array(node_emb_combined).shape);

	end_t = datetime.datetime.now();

	print "Embedding timing: "+str(end_t - start_t);

	emb_filename1 = network_filename + "_combined_Emb_" +str(learning_rate)+"_"+str(hidden_dim)+"_"+str(num_neurons) +".txt";
	save_embedding(emb_filename1,node_emb_combined);

main();
