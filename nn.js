// This file contains NeuralNetwork, Neuron and Weight classes for visualizing each one of them

class NeuralNetwork{
	constructor(layers, centerX, centerY, width, height){
		// First, create Neuron objects of each layer
		let maxNeuron = Math.max(...layers);
		Neuron.r = (height / maxNeuron / 1.25);
		let perLayerX = (width / (layers.length-1));
		let perNeuronY = (height / maxNeuron);
		let startLayerX = (centerX - width/2);

		// Each layer
		this.layers = layers.map((neuronCount, layerIndex) => {
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
		this.weights = [];
		[...Array(this.layers.length-1).keys()].forEach((layerIndex) => {
			let fromLayer = this.layers[layerIndex];
			let toLayer = this.layers[layerIndex+1];

			let useBias = true;
			if (useBias && ((layerIndex+1) !== (this.layers.length-1))) toLayer = toLayer.slice(1);

			// Nested-for loop for connecting neurons with weights
			fromLayer.forEach((fromNeuron, idxFromNeuron) => {
				toLayer.forEach((toNeuron, idxToNeuron) => {
					this.weights.push(
						new Weight(fromNeuron, toNeuron)
					);
				});
			});
		});
	};

	draw = () => {
		// Each layer
		this.layers.forEach(layer => {
			// Each neuron
			layer.forEach(neuron => {
				neuron.draw();
			})
		});

		// Each weight
		this.weights.forEach(weight => {
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
