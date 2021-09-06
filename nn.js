// This file contains SequentialNeuralNetwork, Neuron and Weight classes for visualizing each one of them

// SequentialNeuralNetwork: Built on top of tf.Sequential class, for fully visualizing it
class SequentialNeuralNetwork extends tf.Sequential{
	constructor(
		sequentialArgs
	){
		super(sequentialArgs);
	};

	// Override add method
	add = (layer) => {
		// While adding first layer, check if it's "inputLayer"
		if((this.layers.length === 0) && !layer.name.startsWith("input")){
			console.error("First layer should be tf.layers.inputLayer (checking from the layer name)");
			return;
		}
		// While adding hidden/output layers, check if it's "dense"
		if((this.layers.length > 0) && !layer.name.startsWith("dense")){
			console.error("Hidden and output layers should be tf.layers.dense (checking from the layer name)");
			return;
		}

		// Add layer to the Sequential model
		super.add(layer);
	}

	// Override compile method
	compile = (compileArgs) => {
		// Compile the tf.Sequential side
		super.compile(compileArgs);

		// Create a nested-list for keeping 'Neuron' objects of each layer
		this.layersNeuron = this.layers.map(layer => []);
		// Create Neuron objects for each layer
		this.layers.forEach((layer, layerIndex) => {
			// Get neuron counts
			let neuronCount = (layer.units) || (layer.batchInputShape && layer.batchInputShape[1]);

			// Check if next layer uses bias
			let useBias = ((this.layers[layerIndex+1]) && (this.layers[layerIndex+1].useBias === true));
			if(useBias) neuronCount += 1;

			// Each neuron
			[...Array(neuronCount).keys()].forEach(neuronIndex => {
				// Create Neuron object & push it to the list
				this.layersNeuron[layerIndex].push(new Neuron());
			});
		});

		// Create a nested-list for keeping 'Weight' objects of each layer
		this.layersWeight = this.layers.slice(1).map(layer => []);
		// Create Weight objects for each layer (with pairing layers)
		[...Array(this.layersNeuron.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.layersNeuron[layerIndex];
			let toLayerNeurons = this.layersNeuron[layerIndex+1];

			// Check if next layer uses bias
			let useBias = (this.layers[layerIndex+2] && (this.layers[layerIndex+2].useBias === true));
			// If using bias, there's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(useBias){
				toLayerNeurons = toLayerNeurons.slice(1);
			};
			
			// Get the 2D weight tensor of the layer, turn it into a nested-list
			let layerWeightMatrix = this.getLayerWeightMatrix(layerIndex+1).arraySync();

			// Create each Weight object between the layers
			fromLayerNeurons.forEach((fromNeuron, fromNeuronIndex) => {
				this.layersWeight[layerIndex].push([]);

				toLayerNeurons.forEach((toNeuron, toNeuronIndex) => {
					// Create Weight object & push it to the nested-list
					this.layersWeight[layerIndex][fromNeuronIndex].push(
						new Weight(
							fromNeuron,
							toNeuron,
							layerWeightMatrix[fromNeuronIndex][toNeuronIndex]
						)
					);
				});
			});
		});
	};

	// Gets weight matrix of a dense layer
	getLayerWeightMatrix = (layerIdx) => {
		// Get all weights of the layer (list)
		let layerWeights = this.layers[layerIdx].getWeights();

		// Get kernel
		let kernel = layerWeights[0];
		// Directly return the kernel if there's no bias
		if(layerWeights.length === 1){
			return kernel;
		}

		// Expanding bias vector for concatenation. 1D to 2D
		let bias = tf.expandDims(layerWeights[1], 0);

		// Concat bias&kernel values for getting the final layer weight matrix
		return tf.concat([bias, kernel], 0);
	};
	
	// Override predict method
	predict = (X) => {
		// Feed layer by layer
		let layerOutput = X;
		this.layers.forEach(layer => {
			layerOutput = layer.call(layerOutput);
			layerOutput.print();
		});

		// Forward-propagation with given tensor
		return super.predict(X);
	};

	// Draws the whole network
	draw = (canvas, {gapRateX, gapRateY}) => {
		// Get maximum neuron count
		let maxUnitCount = Math.max(...(this.layers.map(layer => 
			(layer.units) || (layer.batchInputShape && layer.batchInputShape[1])
		)));
		
		// Calculate step per neuron in +Y direction
		let perNeuronY = ((canvas.height*gapRateY) / maxUnitCount);
		// Calculate step per layer in +X direction
		let perLayerX = ((canvas.width*gapRateX) / (this.layers.length-1));
		// Calculate X coordinate of starting point of the network
		let startLayerX = ((canvas.width/2) - (canvas.width*gapRateX/2));
		// Calculate each neuron size with using step per neuron value
		Neuron.r = (perNeuronY / 1.25);

		// Draw neurons
		// Each layer
		this.layersNeuron.forEach((layer, layerIndex) => {
			// Calculate starting point (Y-coordinate of first neuron) of the layer
			// Top of the layer in Y = ((Center of the neural network in Y) - (layer size in Y / 2)) + (applying Y shift a bit for centering)
			let startNeuronY = ((canvas.height/2) - ((perNeuronY * layer.length) / 2)) + (perNeuronY/2);

			// Each neuron
			layer.forEach((neuron, neuronIndex) => {
				// Set position of neuron & draw it
				neuron.x = (startLayerX + (perLayerX * layerIndex));
				neuron.y = (startNeuronY + (perNeuronY * neuronIndex));
				neuron.draw(canvas);
			});
		});

		// Draw weights
		// Each layer
		this.layersWeight.forEach((layer) => {
			// Each neuron
			layer.forEach((neuron) => {
				// Each weight
				neuron.forEach(weight => {
					weight.draw(canvas);
				});
			});
		});
	};
};

class Neuron{
	x = 0;
	y = 0;
	static r = 30.0;

	constructor(){};

	draw = (canvas) => {
		canvas.noFill();
		canvas.strokeWeight(1);
		canvas.stroke(255);
		canvas.circle(this.x, this.y, Neuron.r);
	};
};

let RANGE_MAX = 4.0;
let RANGE_MIN = -4.0;
class Weight{
	value = 0;
	visualValue = 0;

	constructor(from, to, value){
		this.from = from;
		this.to = to;

		this.value = value;
		this.visualValue = value;
	};

	update = () => {
		this.visualValue += (this.value - this.visualValue) * 0.1;
	};

	draw = (canvas) => {
		// Color indicates the carried value
		let indicatorValue = max(min(RANGE_MAX, this.visualValue), RANGE_MIN);
		canvas.strokeWeight(abs(indicatorValue));
		canvas.stroke(255);

		// Line between neurons
		canvas.line(
			// from (neuron)
			(this.from.x+(Neuron.r/2)), this.from.y,
			// to (neuron)
			(this.to.x-(Neuron.r/2)), this.to.y
		);
	};
};
