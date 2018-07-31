"""
Neural-Brane: Tensorflow GPU implementation.
Author: Vachik Dave

"""


import numpy as np
import tensorflow as tf
import os
import random

def generate_train_batch(adj_lists, test_pos, N,node_ids,prob, batch_size=512):
	'''
	uniform sampling (node, neighbor node, non-neighbor node)
	'''
	t = []
	# select list of non-neighbors based on degree
	j_list = np.random.choice(node_ids, batch_size,p=prob); #randint(1, N-1)
	for b in xrange(batch_size):
		# random first node
		u = random.sample(adj_lists.keys(), 1)[0]
		# weight based sample neighbor node
		i = np.random.choice(adj_lists[u][0],p = adj_lists[u][1]);
		ind = 0;
		try:
			while i == test_pos[u]:
				#i = random.sample(adj_lists[u], 1)[0]
				i = np.random.choice(adj_lists[u][0],p = adj_lists[u][1]);
				ind += 1;
				if ind == 100:
					u = random.sample(adj_lists.keys(), 1)[0]
					i = np.random.choice(adj_lists[u][0],p = adj_lists[u][1]);
					ind = 0;
		except KeyError:
			pass;

		j = j_list[len(t)];
		ind = 0;
		# if the j is in neighbor list
		while j in adj_lists[u][0]:
			j = np.random.choice(node_ids,p=prob); #randint(1, N-1)
			#j = random.randint(1, N-1);
			ind += 1;
			if ind == 100:
				print "Error: Training batch generation, cannot find negative sample for u = ", u;
				break;

		t.append([u, i, j])
	return np.asarray(t)


def uij_to_feat(uij,node_feat,max_feat_id):
	'''
	Generate same length attribiute set for each triple with max-attribute_ID+2 as padding
	'''
	uij_feat = [];
	max_feat_count = 0;
	for each in uij:
		u = each[0];
		i = each[1];
		j = each[2];
		u_feat_list = [];
		i_feat_list = [];
		j_feat_list = [];
		try:
			u_feat_list = node_feat[u];
		except KeyError:
			print "Error: features not found... u = ", u ;

		try:
			i_feat_list = node_feat[i];
		except KeyError:
			print "Error: features not found... i = ", i ;

		try:
			j_feat_list = node_feat[j];
		except KeyError:
			print "Error: features not found... j = ", j ;


		if max_feat_count < len(u_feat_list):
			max_feat_count = len(u_feat_list);
		if max_feat_count < len(i_feat_list):
			max_feat_count = len(i_feat_list);
		if max_feat_count < len(j_feat_list):
			max_feat_count = len(j_feat_list);

		uij_feat.append([u_feat_list,i_feat_list,j_feat_list]);

	uij_feat_list = [];
	#add padding of max_feat_id+2
	for each in uij_feat:
		new_u_list = each[0] + [max_feat_id+2]*(max_feat_count-len(each[0]));
		new_i_list = each[1] + [max_feat_id+2]*(max_feat_count-len(each[1]));
		new_j_list = each[2] + [max_feat_id+2]*(max_feat_count-len(each[2]));

		uij_feat_list.append([new_u_list,new_i_list,new_j_list]);

	return np.asarray(uij_feat_list)		

def uij_to_adj(uij,adj_lists,test_pos,N):
	'''
	Generate same length neighborhood set for each triple with N+2 as padding
	'''
	uij_adj = [];
	max_nei_count = 0;
	for each in uij:
		u_adj = [];
		i_adj = [];
		j_adj = [];
		try:
			u_adj = adj_lists[each[0]][0];
		except KeyError:
			print "Error: adj list not found... u = ", each ;

		try:
			i_adj = adj_lists[each[1]][0];
		except KeyError:
			print "Error: adj list not found... i = ", each ;

		try:
			j_adj = adj_lists[each[2]][0];
		except KeyError:
			print "Error: adj list not found... j = ", each ;



		if max_nei_count < len(u_adj):
			max_nei_count = len(u_adj);
		if max_nei_count < len(i_adj):
			max_nei_count = len(i_adj);
		if max_nei_count < len(j_adj):
			max_nei_count = len(j_adj);

		uij_adj.append([u_adj,i_adj,j_adj]);

	uij_adj_list = [];
	#add padding of N+2
	for each in uij_adj:
		new_u_list = each[0] + [N+2]*(max_nei_count-len(each[0]));
		new_i_list = each[1] + [N+2]*(max_nei_count-len(each[1]));
		new_j_list = each[2] + [N+2]*(max_nei_count-len(each[2]));

		uij_adj_list.append([new_u_list,new_i_list,new_j_list]);

	return np.asarray(uij_adj_list)		


def neural_bpr(N, max_feat_id, node_feat, hidden_dim,num_neurons,learning_rate = 0.01, regulation_rate = 0.0001):
	'''
	Neural-Brane architecture: intputs -> Embedding layer (lookup+max-pooling+integration) -> hidden layer -> output layer ->BPR layer
	'''
	#inputs: For triplets (u,i,j)
	#neighborhood information (adjacency vectors)
	u = tf.placeholder(tf.int32, [None,None])
	i = tf.placeholder(tf.int32, [None,None])
	j = tf.placeholder(tf.int32, [None,None])
	#attribute information
	u_att = tf.placeholder(tf.int32, [None,None])
	i_att = tf.placeholder(tf.int32, [None,None])
	j_att = tf.placeholder(tf.int32, [None,None])

	'''
	------------------------------------------- Embedding Layer -------------------------------------------------------------
	'''
	with tf.device("/gpu:0"):
		#----------------------------- attribute embedding --------------------------------------------------------------

		node_att_emb_w = tf.get_variable("node_feat_emb_w", [max_feat_id+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.01))
		# Can add bias for each attribtes
		#node_att_b = tf.get_variable("node_att_b", [max_feat_id+1, 1],initializer=tf.constant_initializer(0.0))

		# CONCAT-LOOKUP (.)
		u_emb1 = tf.reduce_max(tf.nn.embedding_lookup(node_att_emb_w,u_att),1)
		i_emb1 = tf.reduce_max(tf.nn.embedding_lookup(node_att_emb_w,i_att),1)
		j_emb1 = tf.reduce_max(tf.nn.embedding_lookup(node_att_emb_w,j_att),1)

		#i_b1 = tf.reduce_max(tf.nn.embedding_lookup(node_att_b, i_att),1)
		#j_b1 = tf.reduce_max(tf.nn.embedding_lookup(node_att_b, j_att),1)


		#----------------------------- topological embedding --------------------------------------------------------------

		node_emb_w = tf.get_variable("node_emb_w", [N+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.01))

		# Can add bias for each attribtes
		#node_b = tf.get_variable("node_b", [N+1, 1],initializer=tf.constant_initializer(0.0))

		# CONCAT-LOOKUP (.)
		u_emb2 = tf.reduce_max(tf.nn.embedding_lookup(node_emb_w,u),1)
		i_emb2 = tf.reduce_max(tf.nn.embedding_lookup(node_emb_w,i),1)
		j_emb2 = tf.reduce_max(tf.nn.embedding_lookup(node_emb_w,j),1)

		#i_b2 = tf.reduce_max(tf.nn.embedding_lookup(node_b, i),1)
		#j_b2 = tf.reduce_max(tf.nn.embedding_lookup(node_b, j),1)


		#----------------------------- Integration component --------------------------------------------------------------

		u_emb3 = tf.concat([u_emb1,u_emb2],1);
		i_emb3 = tf.concat([i_emb1,i_emb2],1);
		j_emb3 = tf.concat([j_emb1,j_emb2],1);
#		i_b = tf.concat([i_b1,i_b2],1);
#		j_b = tf.concat([j_b1,j_b2],1);
		
		'''
		----------------------------- NN Hidden Layer --------------------------------------------------------------
		'''		
		NN_w1 = tf.get_variable("nn_w1", [hidden_dim*2,num_neurons], initializer=tf.random_normal_initializer(0, 0.1))
		NN_b1 = tf.get_variable("nn_b1", [num_neurons],initializer=tf.constant_initializer(0.0))

		u_emb = tf.nn.relu(tf.matmul(u_emb3, NN_w1));
		i_emb = tf.nn.relu(tf.matmul(i_emb3, NN_w1));
		j_emb = tf.nn.relu(tf.matmul(j_emb3, NN_w1));
#		u_emb = tf.nn.relu(tf.matmul(u_emb3, NN_w1) + NN_b1);
#		i_emb = tf.nn.relu(tf.matmul(i_emb3, NN_w1) + NN_b1);
#		j_emb = tf.nn.relu(tf.matmul(j_emb3, NN_w1) + NN_b1);


	'''
	------------------------------------------- Output + BPR Layer -------------------------------------------------------------
	'''
	x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)
	l2_norm = tf.add_n([tf.reduce_sum(tf.multiply(u_emb, u_emb)), tf.reduce_sum(tf.multiply(i_emb, i_emb)),tf.reduce_sum(tf.multiply(j_emb, j_emb))]);

	bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))


	'''
				Back-propogation learning using gradient descent 
	'''
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(bprloss)
	
	return u, i, j, u_att, i_att, j_att, bprloss, train_op, node_emb_w, node_att_emb_w;


def main(hidden_dim,num_epoch,num_iteration,batch_size,N,adj_lists,test_pos, learning_rate, regulation_rate, num_neurons, max_feat_id, node_feat,node_ids,prob):
	'''
	tensorflow session
	'''
	with tf.Graph().as_default(), tf.Session() as session:

		u, i, j, u_att, i_att, j_att, loss, train_op, node_emb_w, node_att_emb_w = neural_bpr(N, max_feat_id, node_feat, hidden_dim,num_neurons,learning_rate, regulation_rate)
		print 'construct tensorflow computational graph done'
		session.run(tf.global_variables_initializer())

		for epoch in range(num_epoch):
			_batch_bprloss = 0
			for k in range(num_iteration): # uniform samples from training set
				uij = generate_train_batch(adj_lists, test_pos, N,node_ids,prob, batch_size);
				#print k;
				uij_feat_list = uij_to_feat(uij,node_feat,max_feat_id);
				uij_adjList = uij_to_adj(uij,adj_lists,test_pos,N);
				
#				ui_label = get_paired_input(uij);
#				ui_feat_list, labels = ui_to_feat(ui_label,node_feat,max_feat_id);
#				print "training batch generated...";

				_bprloss, _ = session.run([loss, train_op], feed_dict={u:uij_adjList[:,0], i:uij_adjList[:,1], j:uij_adjList[:,2], u_att:uij_feat_list[:,0], i_att:uij_feat_list[:,1], j_att:uij_feat_list[:,2]})

				_batch_bprloss += _bprloss

			print str(epoch) + "," + str(_batch_bprloss / k);

		node_emb, node_att_emb = session.run([node_emb_w,node_att_emb_w])
		return node_emb, node_att_emb;
