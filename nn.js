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

		// First reset own values
		this.reset();

		// Create a nested-list for keeping 'Neuron' objects of each layer
		this.layerNeurons = this.layers.map(layer => []);
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
				this.layerNeurons[layerIndex].push(new Neuron());
			});
		});

		// Create a nested-list for keeping 'Weight' objects of each layer
		this.layerWeights = this.layers.slice(1).map(layer => []);
		// Create Weight objects for each layer (with pairing layers)
		[...Array(this.layerNeurons.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.layerNeurons[layerIndex];
			let toLayerNeurons = this.layerNeurons[layerIndex+1];

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
				this.layerWeights[layerIndex].push([]);

				toLayerNeurons.forEach((toNeuron, toNeuronIndex) => {
					// Create Weight object & push it to the nested-list
					this.layerWeights[layerIndex][fromNeuronIndex].push(
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
	
	// Override predict method
	predict = (X) => {
		// Fire the neuron visually ;)
		this.propagationHighlight.cur = 0.0;
		this.propagationHighlight.curAnim = 0.0;
		this.propagationHighlight.tar = 1.0;

		// Feed layer by layer
		let layerOutput = X;
		this.layers.forEach(layer => {
			layerOutput = layer.call(layerOutput);
			// layerOutput.print();
		});

		// Forward-propagation with given tensor
		return super.predict(X);
	};

	//// Custom methods
	reset = () => {
		this.propagationHighlight = {
			cur: 0.0, tar: 0.0,
			width: 0.03, speed: 0.03,
			anim: AnimationUtils.easeInQuad
		};
		this.layerNeurons = [];
		this.layerWeights = [];
	};

	// Gets weight matrix of a dense layer
	getLayerWeightMatrix = (layerIdx) => {
		// Get all weights of the layer (list)
		let w = this.layers[layerIdx].getWeights();

		// Get kernel
		let kernel = w[0];
		// Directly return the kernel if there's no bias
		if(w.length === 1){
			return kernel;
		}

		// Expanding bias vector for concatenation. 1D to 2D
		let bias = tf.expandDims(w[1], 0);

		// Concat bias&kernel values for getting the final layer weight matrix
		return tf.concat([bias, kernel], 0);
	};

	// Updates all values of weight objects
	updateWeights = () => {
		[...Array(this.layerNeurons.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.layerNeurons[layerIndex];
			let toLayerNeurons = this.layerNeurons[layerIndex+1];

			// Check if next layer uses bias
			let useBias = (this.layers[layerIndex+2] && (this.layers[layerIndex+2].useBias === true));
			// If using bias, there's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(useBias){
				toLayerNeurons = toLayerNeurons.slice(1);
			};
			
			// Get the 2D weight tensor of the layer, turn it into a nested-list
			let layerWeightMatrix = this.getLayerWeightMatrix(layerIndex+1).arraySync();

			fromLayerNeurons.forEach((fromNeuron, fromNeuronIndex) => {
				toLayerNeurons.forEach((toNeuron, toNeuronIndex) => {
					// Update value of Weight object
					let newWeightValue = layerWeightMatrix[fromNeuronIndex][toNeuronIndex];
					this.layerWeights[layerIndex][fromNeuronIndex][toNeuronIndex].value = newWeightValue;
				});
			});
		});
	};

	// Draws the whole network, gets called at each frame
	draw = (canvas, {gapRateX, gapRateY}) => {
		// Update propagation highlight position
		this.propagationHighlight.cur += (this.propagationHighlight.tar - this.propagationHighlight.cur) * this.propagationHighlight.speed;
		this.propagationHighlight.curAnim = this.propagationHighlight.anim(this.propagationHighlight.cur);

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
		this.layerNeurons.forEach((layer, layerIndex) => {
			// Calculate starting point (Y-coordinate of first neuron) of the layer
			// Top of the layer in Y = ((Center of the neural network in Y) - (layer size in Y / 2)) + (applying Y shift a bit for centering)
			let startNeuronY = ((canvas.height/2) - ((perNeuronY * layer.length) / 2)) + (perNeuronY/2);

			// Each neuron
			layer.forEach((neuron, neuronIndex) => {
				// Set position of neuron & draw it
				neuron.x = (startLayerX + (perLayerX * layerIndex));
				neuron.y = (startNeuronY + (perNeuronY * neuronIndex));
				neuron.draw(canvas, this.propagationHighlight);
			});
		});

		// Draw weights
		// Each layer
		this.layerWeights.forEach((layer) => {
			// Each neuron
			layer.forEach((neuron) => {
				// Each weight
				neuron.forEach(weight => {
					weight.draw(canvas, this.propagationHighlight);
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

	draw = (canvas, highlight) => {
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

	// Draws the weight
	draw = (canvas, highlight) => {
		// Smoothly go towards the actual value visually
		this.visualValue += ((this.value - this.visualValue) * 0.1);

		// Color indicates the carried value
		let weightStroke = abs(max(min(this.visualValue, RANGE_MAX), RANGE_MIN));

		// Calculate line's start&end positions
		let fromX = (this.from.x + (Neuron.r/2));
		let fromY = this.from.y;
		let toX = (this.to.x - (Neuron.r/2));
		let toY = this.to.y;

		// Draw weight between neurons
		canvas.stroke(255);
		canvas.strokeWeight(weightStroke);
		canvas.line(
			// from (neuron)
			fromX, fromY,
			// to (neuron)
			toX, toY
		);

		// Highlight the connection if firing
		// Calculate highlight area
		let highlightStartX = (canvas.width * highlight.curAnim);
		let highlightEndX = (canvas.width * Math.min(1.0, (highlight.curAnim + highlight.width)));
		let hFromX = max(highlightStartX, fromX);
		let hToX = min(highlightEndX, toX);

		// Check bounds of the highlight area if it intersects with weight line. If not, return.
		if(hFromX > toX || hToX > toX || hFromX < fromX || hToX < fromX) return;

		// Calculate highlighting line's Y coordinates with linear interpolation
		let hFromY = lerp(fromY, toY, ((hFromX-fromX) / (toX-fromX)));
		let hToY = lerp(fromY, toY, ((hToX-fromX) / (toX-fromX)));

		// Highlight the line!
		canvas.stroke(128);
		canvas.strokeWeight(weightStroke*3);
		canvas.line(
			// from (neuron)
			hFromX, hFromY,
			// to (neuron)
			hToX, hToY
		);
	};
};
