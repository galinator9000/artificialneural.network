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

	// Override compile method for creating visual objects on our side
	compile = (compileArgs) => {
		// Compile the tf.Sequential side
		super.compile(compileArgs);
		// this.layers.forEach(layer => {console.log(layer)});

		// Check if first layer is input layer and the others are dense layers
		if(
			this.layers.length !== (
				this.layers.map((layer, layerIndex) => (
					(layerIndex==0 && layer.name.startsWith("input")) || (layerIndex>0 && layer.name.startsWith("dense"))
				)).reduce((a, b) => (a+b), 0)
			)
		){
			console.error("First layer should be tf.layers.inputLayer and the rest should be tf.layers.dense (checking from the layer name)");
			return;
		}

		// Get layers' properties
		let layerUseBias = this.layers.map(layer => (layer.useBias === true));
		let inputLayerUnitCount = this.layers.map(layer => (layer.batchInputShape && layer.batchInputShape[1]))[0];
		let hiddenLayerUnitCounts = this.layers.map(layer => layer.units).filter(i => i !== undefined);
		this.layerUnitCounts = [inputLayerUnitCount, ...hiddenLayerUnitCounts];

		// Then let's create our visual objects
		let {centerX, centerY, width, height} = this.customArgs;

		// First, create Neuron objects of each layer
		let maxLayerUnitCount = Math.max(...this.layerUnitCounts);
		Neuron.r = (height / maxLayerUnitCount / 1.25);
		let perLayerX = (width / (this.layerUnitCounts.length-1));
		let perNeuronY = (height / maxLayerUnitCount);
		let startLayerX = (centerX - width/2);

		// Each layer
		this.allNeurons = this.layerUnitCounts.map((neuronCount, layerIndex) => {
			// Calculate starting point (Y-coordinate of first neuron) of the layer
			// Top of the layer in Y = ((Center of the neural network in Y) - (layer size in Y / 2)) + (applying Y shift a bit for centering)
			let startNeuronY = (centerY - ((perNeuronY*neuronCount) / 2)) + (perNeuronY/2);

			// List of neurons
			return [...Array(neuronCount).keys()].map(neuronIndex => (
				// Each neuron of layer
				new Neuron(
					(startLayerX+(perLayerX*layerIndex)),
					(startNeuronY+(perNeuronY*neuronIndex))
				)
			));
		});

		// Create Weights between Neurons
		this.allWeights = [];
		[...Array(this.allNeurons.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.allNeurons[layerIndex];
			let toLayerNeurons = this.allNeurons[layerIndex+1];

			// Check if using bias value
			let toLayerNeurons_useBias = layerUseBias[layerIndex+1];
			if (toLayerNeurons_useBias && ((layerIndex+1) !== (this.layers.length-1))){
				toLayerNeurons = toLayerNeurons.slice(1);
			}

			// Nested-for loop for connecting neurons with weights
			fromLayerNeurons.forEach((fromNeuron, idxFromNeuron) => {
				toLayerNeurons.forEach((toNeuron, idxToNeuron) => {
					this.allWeights.push(
						new Weight(fromNeuron, toNeuron)
					);
				});
			});
		});
	}

	draw = () => {
		// Each layer
		this.allNeurons.forEach(layer => {
			// Each neuron
			layer.forEach(neuron => {
				neuron.draw();
			})
		});

		// Each weight
		this.allWeights.forEach(weight => {
			weight.draw();
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

class Weight{
	constructor(from, to){
		this.from = from;
		this.to = to;
		this.val = random(0, 1);
	};

	draw = () => {
		// Color indicates the carried value
		strokeWeight(1);
		stroke(
			this.val * 255,
			this.val * 255,
			this.val * 255
		);

		// Line between neurons
		line(
			// from (neuron)
			(this.from.x+(Neuron.r/2)), this.from.y,
			// to (neuron)
			(this.to.x-(Neuron.r/2)), this.to.y
		);
	};
};
