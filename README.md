Output of Forwar Propagation ad Reverse-mode Automatic Differentiation (Back Propagation)
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    FORWARD PROPAGATION     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


	Layer 0:
        	         f(x) = summation weights[i] * inputs[i] + bias 
        	                f(x) =  0.7550000000000001
	
        	                        neuron activation: 0.6802671966986485
	
        	         f(x) = summation weights[i] * inputs[i] + bias 
        	                f(x) =  0.68
	
        	                        neuron activation: 0.6637386974043528
	
        	        layer activation: [0.6802671966986485, 0.6637386974043528]
	Layer 1:
                 	f(x) = summation weights[i] * inputs[i] + bias 
                        	f(x) =  -0.6122404770287837

                        	        neuron activation: 0.35154828437998153
	
                 	f(x) = summation weights[i] * inputs[i] + bias 
                        	f(x) =  0.256352829837228

                        	        neuron activation: 0.5637395261839213
	
                	layer activation: [0.35154828437998153, 0.5637395261839213]
	Layer 2:	
        	         f(x) = summation weights[i] * inputs[i] + bias 
        	                f(x) =  0.23252677616116818
	
        	                        neuron activation: 0.5578711770959379
	
        	         f(x) = summation weights[i] * inputs[i] + bias 
        	                f(x) =  0.42977249676674306
	
        	                        neuron activation: 0.60581934145203
	
        	        layer activation: [0.5578711770959379, 0.60581934145203]
	Output layer:
        	         f(x) = summation weights[i] * inputs[i] + bias 
        	                f(x) =  0.7125987604356083

	0.6709751363686631
	########## Observeved Activations Per layer ##########
        	                [[0.6802671966986485, 0.6637386974043528], [0.35154828437998153, 0.5637395261839213], [0.5578711770959379, 0.60581934145203], 0.6709751363686631]



	@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    BACK PROPAGATION        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                        Loss = 0.17097513636866313
	Consumed Weights and activations consumed by back prop ::::  [[0.3, 0.9], [[0.02, 0.4], [0.1, 0.7]], [[-0.9, 0], [0.25, 0.13]], [[0.1, 0.8], [0.4, 0.6]]] :::: [[0.5578711770959379, 0.60581934145203], [0.35154828437998153, 0.5637395261839213], [0.6802671966986485, 0.6637386974043528]] 

        	                                OUTPUT LAYER
        	                delta at output layer : 0.17097513636866313

	changeinweight args 1           0.17097513636866313             0.6709751363686631

	new weight0.4147200654406145

	changeinweight args 1           0.17097513636866313             0.6709751363686631

	new weight1.0147200654406145
        	                                New weights for out put layer [[0.4147200654406145, 1.0147200654406145]]
        	                                
        	                                HIDDEN LAYERS
	Consumed Weights and activations consumed by hidden layers ::::  [[0.3, 0.9], [[0.02, 0.4], [0.1, 0.7]], [[-0.9, 0], [0.25, 0.13]], [[0.1, 0.8], [0.4, 0.6]]] :::: [[0.5578711770959379, 0.60581934145203], [0.35154828437998153, 0.5637395261839213], [0.6802671966986485, 0.6637386974043528]] 

	input weights [[0.02, 0.4], [0.1, 0.7]] : activations [0.5578711770959379, 0.60581934145203]
	Neuron 0: Weights [0.02, 0.4]  activations : [0.5578711770959379, 0.60581934145203] 


        	                        activation 0.5578711770959379           Weight [0.02, 0.4]               delta :0.17097513636866313
	
        	                new delta : 0.000843423517112147
        	                        This neurons delta: 0.000843423517112147
	
	changeinweight args 1           0.000843423517112147            0.5578711770959379

	new weight0.02047052167028175

	changeinweight args 1           0.000843423517112147            0.5578711770959379

	new weight0.4004705216702818
	[0.02047052167028175, 0.4004705216702818]
	Neuron 1: Weights [0.1, 0.7]  activations : [0.5578711770959379, 0.60581934145203] 


        	                        activation 0.60581934145203             Weight [0.1, 0.7]                delta :0.17097513636866313

        	                new delta : 0.004082925016113817
        	                        This neurons delta: 0.004082925016113817
	
	changeinweight args 1           0.004082925016113817            0.60581934145203

	new weight0.1024735149444601

	changeinweight args 1           0.004082925016113817            0.60581934145203

	new weight0.7024735149444601
	[0.1024735149444601, 0.7024735149444601]
        	                                New weights for last hidden layer into output [[0.02047052167028175, 0.4004705216702818], [0.1024735149444601, 0.7024735149444601]]
        	                                deltas for last hidden layer [0.000843423517112147, 0.004082925016113817]



	Deltas object updated in hidden layer [0.17097513636866313, [0.000843423517112147, 0.004082925016113817]]



	Hidden layer 1
	activations                                     [0.35154828437998153, 0.5637395261839213]                       deltas: [0.17097513636866313, [0.000843423517112147, 0.004082925016113817]]



	using deltas : [0.000843423517112147, 0.004082925016113817]



	Neuron 0: Weights [-0.9, 0]  previous activations : [0.35154828437998153, 0.5637395261839213] downstream deltas : [0.000843423517112147, 0.004082925016113817]

	loop 0
	loop 1

	changeinweight args 1           0.00017304172752455053          0.35154828437998153

	new weight-0.8999391674775626

	changeinweight args 1           0.00017304172752455053          0.35154828437998153

	new weight6.083252243740397e-05
	Neuron 1: Weights [0.25, 0.13]  previous activations : [0.35154828437998153, 0.5637395261839213] downstream deltas : [0.000843423517112147, 0.004082925016113817]

	loop 0
	loop 1

	changeinweight args 1           -0.0001823959675611485          0.5637395261839213

	new weight0.24989717618366922

	changeinweight args 1           -0.0001823959675611485          0.5637395261839213

	new weight0.12989717618366922
                                        New weights for hidden layer into next layer [[-0.8999391674775626, 6.083252243740397e-05], [0.24989717618366922, 0.12989717618366922]]



                                        deltas for hidden layer [0.00017304172752455053, -0.0001823959675611485]



Hidden layer 2
	activations                                     [0.6802671966986485, 0.6637386974043528]                        deltas: [0.17097513636866313, [0.000843423517112147, 0.004082925016113817], [0.00017304172752455053, -0.0001823959675611485]]



	using deltas : [0.00017304172752455053, -0.0001823959675611485]



	Neuron 0: Weights [0.1, 0.8]  previous activations : [0.6802671966986485, 0.6637386974043528] downstream deltas : [0.00017304172752455053, -0.0001823959675611485]

	loop 0
	loop 1

	changeinweight args 1           2.7973721509443145e-05          0.6802671966986485
	
	new weight0.10001902960511247

	changeinweight args 1           2.7973721509443145e-05          0.6802671966986485

	new weight0.8000190296051125
	Neuron 1: Weights [0.4, 0.6]  previous activations : [0.6802671966986485, 0.6637386974043528] downstream deltas : [0.00017304172752455053, -0.0001823959675611485]

	loop 0
	loop 1

	changeinweight args 1           8.976885812647663e-06           0.6637386974043528

	new weight0.40000595830649605
	
	changeinweight args 1           8.976885812647663e-06           0.6637386974043528

	new weight0.6000059583064961
                                        	New weights for hidden layer into next layer [[0.10001902960511247, 0.8000190296051125], [0.40000595830649605, 0.6000059583064961]]



                                       		deltas for hidden layer [2.7973721509443145e-05, 8.976885812647663e-06]






	New weights after back propagation : [[[0.10001902960511247, 0.8000190296051125], [0.40000595830649605, 0.6000059583064961]], [[-0.8999391674775626, 6.083252243740397e-05], [0.24989717618366922, 0.12989717618366922]], [[0.02047052167028175, 0.4004705216702818], [0.1024735149444601, 0.7024735149444601]], [0.4147200654406145, 1.0147200654406145]]
