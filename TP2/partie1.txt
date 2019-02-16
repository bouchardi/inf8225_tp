1.

D = 100
L hiddens layers of size
K dimension to the output

Parameters initialization:
	# Initialize the weights with the correct number of units per layer
	weights = [1 matrix(x_size, D),
			   L-1 matrices(D, D),
			   1 matrix (D, K)]

	# Initialize the weights with the correct number of units per layer
	weights = [1 matrix(1, D),
			   L-1 matrices(1, D),
			   1 matrix (1, K)]

Forward pass:

	# Initialize h with the input. h is the output of the previous layer.
	h = X
	# Initialize the cache
	cache = [(h, None)]
	# For each layer, we get the set of weights and bias
	for W, b in weights, bias:
		# Concatenate W and b
		Wb = [W, b]
		# Add a line of one to the input
		hb = [h, 1]
    	# Compute the dot product of the weights and bias with the previous layer output
        a = Wb.T x hb
        # Compute the activation function
        h = Sigmoid(a)
		# Keep the results of each layer in a cache
        cache += (h, a)
	# Return the output of the last layer and the cache
	return h, cache

Backward pass:

	# Initialize an empty list of gradients
	grads = []
    # Compute the gradient of the loss  before the activation
	grad_a = - (y - y')
	# For each layer
    for k in layers:
		# Get the previous layer output from the cache
		h = cache['h'][k-1]
  		# Compute the gradient of hidden layers weights
    	grad_W = grad_a x h
  		# Compute the gradient of hidden layers bias
        grad_b = grad_a
        # Compute the gradient of the hidden layer bellow
        grad_h = W[k] x grad_a
		# Get the previous layer output before activation from the cache
		a = cache['a'][k-1]
        # Compute the gradient of the hidden layer bellow (before activation)
		# Element-wise multiplication!
		grad_a = grad_h x sigmoid_deriv(a)
		# Keep the gradient of de weights and bias for each layer
        grads.append((grad_W.T, grad_b))
    # Return the gradient for each layers
	return grads

b.

- Split the dataset in train/valid/test set
- Run hyper-parameters search (GridSearch, RandomSearch) with different learning rates and batch size
- Add regularization methods (Dropout, ElasticNet)
- Use more sophisticated optimization methods (momentum, adams, weight decay)
- Use early stopping to avoid underfitting

Training:

	shuffle(dataset)
	N = 500000
	# Split the dataset in train (70%)/ valid (15%)/ test (15%)
	trainset = dataset[0: 0.7*N]
	validset = dataset[0.7*N: 0.85*N]
	testset = dataset[0.85*N: N]

    # For each epoch
	for epoch in epochs:

		# For each batch in the trainset
        for batch_x, batch_y in trainset:
			# Compute the forward pass
            prediction, cache = forward(batch_x)
			# Compute the backward pass
            grads = backward(y, prediction, cache)
            # Update parameters
            W = W - grads * learning_rate
            b = b - grads * learning_rate
			# Compute the loss to see the model learning curve
			L = loss(prediction, batch_y)
			print(L)

		# For each batch in the validset
        for batch_x, batch_y in validset:
			# Compute the forward pass
        	prediction = forward(batch_x)
    		# Compute the loss to see the model learning curve
			L = loss(prediction, batch_y)
			print(L)
            # Compute the accuracy which is the metric we are actually interested in
			accuracy = get_accuracy(prediction, batch_y)
			print(accuracy)

		# If the accuracy computed is the best one so far,
		# keep track of the current weights and bias
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_W, best_b = W, b

    # For each batch in the test set
	for batch_x, batch_y in testset:
		# Compute the forward pass
        prediction = forward(batch_x)
        test_accuracy = get_accuracy(prediction, batch_y)
