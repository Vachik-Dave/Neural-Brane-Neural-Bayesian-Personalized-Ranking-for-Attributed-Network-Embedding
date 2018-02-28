Inputs:
1st input: grpah filename
	Format: First row specify "#nodes #edges". 
		From the 2nd row, each row specify an edge in space delimited format: "node_id1 node_id2".
		Node_id need to be integer and node_id starts with 0.

2nd input: attribute filename:
	Format: "node_id attribute_id1:1 attribute_id5:1 attribute_id7:1 ..."
		Each row contains atributes for a node. 
		Row only contains positive attribute_ids.
		attribtue_ids are positive integer starts with 0.

3rd input: Embedding dimensionality. [integer number]

4th input: hidden layer neuron counts (hidden layer dimension) [integer number].
