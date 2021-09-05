// This file contains SequentialNeuralNetwork, Neuron and Weight classes for visualizing each one of them

// SequentialNeuralNetwork: Built on top of tf.Sequential class, for fully visualizing it
class SequentialNeuralNetwork extends tf.Sequential{
	constructor(
		sequentialArgs,
		customArgs
	){
		super(sequentialArgs);
		this.customArgs = customArgs;
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

		// Get maximum neuron count
		this.maxUnitCount = Math.max(...(this.layers.map(layer => 
			(layer.units) || (layer.batchInputShape && layer.batchInputShape[1])
		)));

		// Calculate step per neuron in +Y direction
		this.perNeuronY = (this.customArgs.height / this.maxUnitCount);
		// Calculate step per layer in +X direction
		this.perLayerX = (this.customArgs.width / (this.layers.length-1));
		// Calculate X coordinate of starting point of the network
		this.startLayerX = (this.customArgs.centerX - this.customArgs.width/2);
		// Calculate each neuron size with using step per neuron value
		Neuron.r = (this.perNeuronY / 1.25);

		// Create a nested-list for keeping 'Neuron' objects of each layer
		this.layersNeuron = this.layers.map(layer => []);
		// Create Neuron objects for each layer
		this.layers.forEach((layer, layerIndex) => {
			// Get neuron counts
			let neuronCount = (layer.units) || (layer.batchInputShape && layer.batchInputShape[1]);

			// Check if next layer uses bias
			let useBias = ((this.layers[layerIndex+1]) && (this.layers[layerIndex+1].useBias === true));
			if(useBias) neuronCount += 1;

			// Calculate starting point (Y-coordinate of first neuron) of the layer
			// Top of the layer in Y = ((Center of the neural network in Y) - (layer size in Y / 2)) + (applying Y shift a bit for centering)
			let startNeuronY = (this.customArgs.centerY - ((this.perNeuronY*neuronCount) / 2)) + (this.perNeuronY/2);

			// Each neuron
			[...Array(neuronCount).keys()].forEach(neuronIndex => {
				// Create Neuron object & push it to the list
				let posX = (this.startLayerX + (this.perLayerX * layerIndex));
				let posY = (startNeuronY + (this.perNeuronY * neuronIndex));
				this.layersNeuron[layerIndex].push(new Neuron(posX, posY));
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

			// There's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(useBias){
				toLayerNeurons = toLayerNeurons.slice(1);
			};

			// Concat kernel&bias matrices for getting the layer weight matrix
			let kernel = this.layers[layerIndex+1].getWeights()[0];
			// Expanding bias vector for concatenation. 1D to 2D
			let bias = tf.expandDims(
				this.layers[layerIndex+1].getWeights()[1],
				0
			);
			let layerWeightTensor = tf.concat([bias, kernel], 0);
			
			// Get the 2D tensor in a nested-list
			let layerWeightMatrix = layerWeightTensor.arraySync();

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

	// Draws the whole network
	draw = () => {
		// Draw neurons
		// Each layer
		this.layersNeuron.forEach(layer => {
			// Each neuron
			layer.forEach(neuron => {
				neuron.draw();
			});
		});

		// Draw weights
		// Each layer
		this.layersWeight.forEach(layer => {
			// Each neuron
			layer.forEach(neuron => {
				// Each weight
				neuron.forEach(weight => {
					weight.draw();
				});
			});
		});
	};
};

class Neuron{
	static r = 30.0;
	constructor(x, y){
		this.x = x;
		this.y = y;
	};

	draw = () => {
		noFill();
		strokeWeight(1);
		stroke(255);
		circle(this.x, this.y, Neuron.r);
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

	draw = () => {
		// Color indicates the carried value
		let indicatorValue = max(min(RANGE_MAX, this.visualValue), RANGE_MIN);
		strokeWeight(abs(indicatorValue));
		stroke(255);

		// Line between neurons
		line(
			// from (neuron)
			(this.from.x+(Neuron.r/2)), this.from.y,
			// to (neuron)
			(this.to.x-(Neuron.r/2)), this.to.y
		);
	};
};
