# Neural-Brane: Neural Bayesian Personalized Ranking for Attributed Network Embedding

### Requirements:
Python packages: NumPy (version 1.13.* or above), tensoflow-GPU (tested Version 1.1.0)

### Run:
python Neural-Brane_embedding.py ./Datasets/citeseer_graph.txt ./Datasets/citeseer_nodeAtt.txt 75 150

### Inputs:
> 1st input: grpah filename
 - Format: First row specify "#nodes #edges". 
 - From the 2nd row, each row specify an edge in space delimited format: "node_id1 node_id2".
 - Node_id need to be integer and node_id starts with 0.

> 2nd input: attribute filename:
  - Format: "node_id attribute_id1:1 attribute_id5:1 attribute_id7:1 ..."
  - Each row contains atributes for a node. 
  - Row only contains positive attribute_ids.
		attribtue_ids are positive integer starts with 0.

> 3rd input: Embedding dimensionality. [integer number]

> 4th input: hidden layer neuron counts (hidden layer dimension) [integer number].

### Note: Tensorflow-cpu version will be available soon.
