"""
PROOF OF CONCEPT
Author : Solomon Muchiri
Education: Bsc Applied Computing

This is proof of my understanding on Neural Networks (theory) and ability to (practical) produce code with respect to the forward and back pass (propagation) algorithm 

1.) Implimentation of Forward Propagation
2.) Implimentation of Backward Propagation/reverse-mode automatic differentiation
3.) Perform Gradient Descent(update parameters/weights)

no libraries like pytorch, numpy etc used
"""
import math
import random

nueronsPerLayer = 2
Layers = 1 

ExpectedOutcome = 0.5
learningrate = 1
#when learning rate is 24 the neuron network learns faster i.eweights are tweaked much faster with less forward and back passes and achieves 0.0 loss( neuron network output - expected outcome )
X = [0.35, 0.9] #inputs 
W11 = [0.1, 0.8] #w[i:n]wieghts from the input layer into the first neuron   
W12 = [0.4, 0.6] #w[i:n]]wieghts from the input layer into the second neuron
WO = [0.3, 0.9] # w[i:n]wieghts from the hidden layer into the output neuron 

WL1 = [ W11, W12]
WL2 = [ [ -0.9, 0], [0.25, 0.13]]
WL3 = [ [ 0.02, 0.4 ], [ 0.1, 0.7] ]
WNa = [ WL1, WO]
BNa = [ 0, 0]
WNb = [WL1, WL2,WO]
WNc = [WL1, WL2, WL3 ,WO]
BNx =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

ObservedActivationsPerLayer = []
ObservedDeltasPerLayer = []

#dot product function
def weightedSumPlusBias(inputs, weights, bias):
	summation = 0;
	print("\t\t f(x) = summation weights[i] * inputs[i] + bias ")
	for i in range(0, len(inputs)):
		wx = inputs[i] * weights[i]
		summation += wx
		#print(f"\t\t\t{weights[i]} * {inputs[i]} = {wx}")
	a = (summation + bias) * 1
	print(f"\t\t\tf(x) =  {a}\n")
	return a
	
#sigmoid function
def activationFunction(x):
	return 1 / (1 + math.exp(-x))
	
def activateNeuron(inputs, weights, bias):
	a = activationFunction(weightedSumPlusBias(inputs, weights, bias))
	print(f"\t\t\t\tneuron activation: {a}\n")
	return a
	
def hiddenLayer(inputs, weights, bias):
	layeractivation = []
	#here the len of the weights tell us how many neurons we have for this layer	

	for i in range (0, len(weights)):
		w = weights[i]
		layeractivation.append(activateNeuron(inputs, w, bias))
	print(f"\t\tlayer activation: {layeractivation}")
	ObservedActivationsPerLayer.append(layeractivation)
	return layeractivation
	
def outputLayer(inputs, weights, bias):
	output = activationFunction(weightedSumPlusBias(inputs, weights, bias))
	ObservedActivationsPerLayer.append(output)
	return output

#NOTE :: let layers argument be the number of hidden layers present
#layers = 0 if no hidden layer present only  anoutput layer 
	# weights is a single dimension array
	# bias is an int 
#layers = 1 if 1 hidden layer is present and an output layer
	# weights is a multy dimensional array  where weights[0] are the weights for layer 1
	# bias is an array b[i] is the bias for layer 1
#layer  > 1 if at least  2 hidden layers are present and an output layer
	# weights is a multy dimensional array  where weights[i] are the weights for layer i and weights [n] are the weights of the output layer
	# bias is an array b[i] is the bias for layer i 
def forwardPropagate(layers, X, weights, biases):
	print(f"\n\n{'@'*60}\tFORWARD PROPAGATION\t{'@'*79}\n\n")
	if layers <= 0 :
		#print("Output layer:")
		output = outputLayer(X, weights, biases)
		#print(output)
		#print(f"{ObservedActivationsPerLayer}\n")
		return output
	if layers == 1:	
		B1 = biases[0]
		BO = biases[1]
		print(X)
		WL1 = weights[0]
		WO = weights[1]
		print("Layer 1:")
		act = hiddenLayer(X, WL1, B1)
		print("Output layer:")
		output = outputLayer(act, WO, BO)
		print(output)
		print(f"{'#' * 10} Observeved Activations Per layer {'#' * 10}\n\n\t\t\t{ObservedActivationsPerLayer}\n")
		return output 
	elif layers > 1:
		rounds = layers
		WO = weights[layers]
		BO = biases[layers]
		activation = X
		for l in range(0, rounds):
			W = weights[l]
			B = biases[l]
			print(f"Layer {l}:")
			activation = hiddenLayer(activation, W, B)
		print("Output layer:")
		output = outputLayer(activation, WO, BO)
		print(output)
		print(f"{'#' * 10} Observeved Activations Per layer {'#' * 10}\n\t\t\t{ObservedActivationsPerLayer}\n")
		return output
	else:
		print("Invalid layer count provided")
		return 0xff
		
#cost function
def lossFunction(expected, outcome):
	return outcome - expected

#change in cost / change in weightss
def deltaAtOutputLayerNeuron(expected, activation):
	#error term
	delta =  (expected - activation) *activation * (1 - activation)
	print(f"\t\t\tdelta at output layer : {delta}")
	return delta
	
#here delta is the error term for the last neuron since we are propagating back
def deltaAtLastHiddenLayerNeuron(activation, weight, previousdelta):	
	delta = activation * (1 - activation)  * (weight * previousdelta) 
	print(f"\t\t\tnew delta : {delta}")
	return delta
	
#change in cost / change in weights 
def deltaAtHiddenLayerNeuron(activation, weightstochange, previouslayerdeltas):
    deltasummation = 0
    previouslayerneurondelta = previouslayerdeltas
    for i in range(0, len(previouslayerneurondelta)):
        weighttochange = weightstochange[i]
        print(f"loop {i}")

        delta = - previouslayerneurondelta[i] * weighttochange * activation * (1 - activation)
        deltasummation += delta

    return deltasummation
def changeInWeight(learingrate, activation, delta):
	print(f"\nchangeinweight args { learningrate}\t\t{activation}\t\t{delta}\n")
	return learningrate * delta * activation
	
def newWeight(old, change):
	print(f"new weight{change+old}")
	return change + old
	
def changedWeightsAtOutputNeuron(expected, outcome, weights):
	newweights = []
	delta = deltaAtOutputLayerNeuron(expected, outcome)
	for i in range(0, len(weights)):	
		change = changeInWeight(learningrate, delta, outcome)
		newweights.append(newWeight(weights[i], change))
	return newweights, delta
	
def changedWeightsAtEachLastHiddenLayerNeuron(weightstochange, previousactivation, learningrate, delta):
	newweights = []
	print(f"\n\t\t\t\tactivation { previousactivation}\t\tWeight {weightstochange}\t\t delta :{delta}\n")
	deltanew = deltaAtLastHiddenLayerNeuron(previousactivation, weightstochange[0], delta)
	print(f"\t\t\t\tThis neurons delta: {deltanew}")
	for i in range(0, len(weightstochange)):	
		change = changeInWeight(learningrate, deltanew, previousactivation)
		newweights.append(newWeight(weightstochange[i], change))
	return newweights, deltanew
	
def changedWeightsAtEachHiddenLayerNeuron(weightstochange, previousactivation, previouslayerdeltas, learningrate):
    newweights = []

    delta = deltaAtHiddenLayerNeuron(previousactivation, weightstochange, previouslayerdeltas)


    for j in range(0, len(weightstochange)):
        change = changeInWeight(learningrate, delta, previousactivation)
        newweights.append(newWeight(weightstochange[j], change))

    return newweights, delta
    
def changedWeightsAtLastHiddenLayer(layerweights, activations, learningrate, delta):
	LW = layerweights
	newlayerweights = []
	layerdeltas = []
	print(f"input weights {LW} : activations {activations}")
	for i in range(0, len(LW)):
		print(f"Neuron {i}: Weights {LW[i]}  activations : {activations} \n")
		weight, neurondelta = changedWeightsAtEachLastHiddenLayerNeuron(LW[i], activations[i], learningrate, delta)		
		print(weight)
		layerdeltas.append(neurondelta)
		newlayerweights.append(weight)
	print(f"\t\t\t\t\tNew weights for last hidden layer into output {newlayerweights}")
	print(f"\t\t\t\t\tdeltas for last hidden layer {layerdeltas}")
	return newlayerweights, layerdeltas
	
def changedWeightsAtEachHiddenLayer(layerweights, previousactivations, learningrate, downstreamdeltas):
	LW = layerweights
	newlayerweights = []
	layerdeltas = []
	for i in range(0, len(LW)):
		print(f"Neuron {i}: Weights {LW[i]}  previous activations : {previousactivations} downstream deltas : {downstreamdeltas}\n")
		changedweights, delta =  changedWeightsAtEachHiddenLayerNeuron(LW[i],previousactivations[i], downstreamdeltas,learningrate)
		newlayerweights.append(changedweights)
		layerdeltas.append(delta)
	print(f"\t\t\t\t\tNew weights for hidden layer into next layer {newlayerweights}\n\n\n")
	print(f"\t\t\t\t\tdeltas for hidden layer {layerdeltas}\n\n\n")
	return newlayerweights, layerdeltas
	
def backPropagation(expected, outcome, neuronweights, layers):
	deltas = []
	nextdeltas = []
	newWeights = []
	print(f"\n\n{'@'*60}\tBACK PROPAGATION\t{'@'*79}\n\n")
	loss = 0
	UpdatedWeights = []
	if outcome == 0xff:
		print("Error occurred during forward propagation.\nEXITING PROCESS")
		return 0
	
	loss = lossFunction(expected, outcome)
	if loss == 0:
		print("We got expected values, weights are not going to be altered again.Not Perfoming Back Propagation.\n")
		return neuronweights
	print(f"\t\t\tLoss = {loss}")
	neuronweights.reverse()
	ObservedActivationsPerLayer.pop()#remove the last item since it is our outcome and we already have it
	ObservedActivationsPerLayer.reverse()#reverse the list  and consume it in reverse "back prop"
	print(f"Consumed Weights and activations consumed by back prop ::::  {neuronweights} :::: {ObservedActivationsPerLayer} \n")
	print(f"\t\t\t\t\tOUTPUT LAYER")
	newoutputlayerweights, delta = changedWeightsAtOutputNeuron(expected, outcome, neuronweights[0])
	deltas.append(delta)
	newWeights.append(newoutputlayerweights)
	print(f"\t\t\t\t\tNew weights for out put layer {newWeights}")
	print(f"\t\t\t\t\tHIDDEN LAYERS")
	print(f"Consumed Weights and activations consumed by hidden layers ::::  {neuronweights} :::: {ObservedActivationsPerLayer} \n")
	if layers == 0:
		return newWeights
	elif layers == 1:
		LW = neuronweights[1]
		activations = ObservedActivationsPerLayer[0]
		weights, deltasforlayer = changedWeightsAtLastHiddenLayer(LW, activations, learningrate, delta)
		deltas.append(deltasforlayer) 
		newWeights.append(weights)
		newWeights.reverse()
		print(f"\n\n\nNew weights after back propagation : {newWeights}\n\n\n")
		return newWeights
	elif layers > 1:
		LW = neuronweights[1]
		activations = ObservedActivationsPerLayer[0]
		hiddenlayerweights, deltasforhiddenlayer = changedWeightsAtLastHiddenLayer(LW, activations, learningrate, delta)
		deltas.append(deltasforhiddenlayer) 
		newWeights.append(hiddenlayerweights)
		print(f"\n\n\nDeltas object updated in hidden layer {deltas}\n\n\n")
		for i in range(1, layers):
			print(f"Hidden layer {i}")
			LW = neuronweights[i + 1]
			activations = ObservedActivationsPerLayer[i]
			print(f"activations\t\t\t\t\t{activations}\t\t\tdeltas: {deltas}\n\n\n")
			downstreamdeltas = deltas[i]
			print(f"using deltas : {downstreamdeltas}\n\n\n")
			layerweights, streamdeltas = changedWeightsAtEachHiddenLayer(LW, activations, learningrate, downstreamdeltas)
			newWeights.append(layerweights)
			deltas.append(streamdeltas)
		newWeights.reverse()
		print(f"\n\n\nNew weights after back propagation : {newWeights}\n\n\n")
		return newWeights
	else:
		return "ERROR BAD ARGUMENTS"
		
def main():
	NewNW = WNc
	Train = True
	runtimeerror = False
	count = 0
	trainingweights = []
	trainedweghts = []
	while(Train):
		if NewNW == None  or NewNW == 0:
			print("Error")
			runtimeerror = True
			Train = False
		#outcome = forwardPropagate(1,X, NewNW, BNa)
		#outcome = forwardPropagate(2,X, NewNW, BNx)
		outcome = forwardPropagate(3,X, NewNW, BNx)
		if outcome == ExpectedOutcome:
			print(f"\n\n\n{'*' * 15}\n\n\t\t\t\t Outcome = {outcome}\n\n\t\t\t\t loss = 0.0000....\n\n {'*' * 15}\n\n")
			trainedweights = NewNW
			Train = False
			break;
		#NewNW = backPropagation(ExpectedOutcome, outcome, NewNW, 1)
		#NewNW = backPropagation(ExpectedOutcome, outcome, NewNW, 2)
		NewNW = backPropagation(ExpectedOutcome, outcome, NewNW, 3)
		print(f"{'#' * 10} New Weights After Back Propagation {'#' * 10}\n\n\t\t\t\t{NewNW}\n\n")
		trainingweights.reverse()
		print(f"old wights: {trainingweights} <----------------------> {NewNW}\n\n")
		if trainingweights == NewNW:
			print(f"\n\n\n{'*' * 15}\n\n\t\t\t\tNetwork Weights updated to achieve minimum loss \n\n{'*' * 15}\n\n")
			print("Weights never changed.....")
			trainedweights = NewNW
			Train = False
			break;
		
		trainingweights = NewNW
		print(f"updated wights: {trainingweights} __________ {NewNW}\n\n")
		ObservedActivationsPerLayer.clear()

		count += 1
		print(f"propagated forward and backwards {count} times")
	print(f"looped {count} times")
	if not Train and not runtimeerror:
#Neuron network trained
		print(f"{'#' * 10} Infrence Weights After Back Propagation {'#' * 10}\n\n\t\t\t\t{trainedweights}\n\n")
		ObservedActivationsPerLayer.clear()
		#outcome = forwardPropagate(1,X, NewNW, BNa)
		#outcome = forwardPropagate(2,X, trainedweights, BNx)
		outcome = forwardPropagate(3,X, NewNW, BNx)
		print(f"\n\n\n{'*' * 15}\n\n\t\t\t\trefrence time: \n\n\t\t\t\t{outcome}\n\n")
		
		loss = lossFunction(ExpectedOutcome, outcome)
		print(f"INFRENCE LOSS: {activationFunction(loss)}")
		print(f"INFRENCE LOSS: {loss}")
		if loss > 0:
			print("Under predicted")
		else:
			print("Over predicted")
		ObservedActivationsPerLayer.clear()
		Xrandom = [0.6, 0.3]
		print(f"{'#' * 10} Infrence Weights After Back Propagation {'#' * 10}\n\n\t\t\t\t{trainedweights}\n\n")
		#outcome = forwardPropagate(1,X, NewNW, BNa)
		#outcome = forwardPropagate(2,Xrandom, trainedweights, BNx)
		outcome = forwardPropagate(3,X, NewNW, BNx)
		print(f"\n\n\n{'*' * 15}\n\n\t\t\t\trefrence time: \n\n\t\t\t\t{outcome}\n\n")
		
		loss = lossFunction(ExpectedOutcome, outcome)
		print(f"INFRENCE LOSS: {activationFunction(loss)}")
		print(f"INFRENCE LOSS: {loss}")
		if loss > 0:
			print("Under predicted")
		else:
			print("Over predicted")
""""""
if __name__ == "__main__":
	main()
