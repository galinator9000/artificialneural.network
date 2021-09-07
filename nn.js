// This file contains SequentialNeuralNetwork, Neuron and Weight classes for visualizing each one of them

// SequentialNeuralNetwork: Built on top of tf.Sequential class, for fully visualizing it
class SequentialNeuralNetwork extends tf.Sequential{
	constructor(
		sequentialArgs,
		vArgs
	){
		super(sequentialArgs);
		// vArgs holds our custom values (for visual purposes usually) for our class
		this.vArgs = {
			...vArgs
		};
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

		// Initially call them.
		this.onChangeWeights();
		this.onChangeNeurons();
	};
	
	// Override predict method
	predict = (X) => {
		// Feed layer by layer
		let layerOutput = X;
		this.layers.forEach((layer, layerIndex) => {
			// Feed to layer & get output
			layerOutput = layer.call(layerOutput);

			// Set each neuron's output
			layerOutput.arraySync()[0].forEach((neuronOutput, neuronIndex) => {
				// Set neuron's output
				this.layerNeurons[layerIndex][neuronIndex].value = neuronOutput;
			});
		});

		// Neuron values changed, call it!
		this.onChangeNeurons();
		
		// Fire the neurons forward ;)
		this.vArgs.propagationHighlight.x = 0.0;
		this.vArgs.propagationHighlight.xAnim = 0.0;
		this.vArgs.propagationHighlight.xTarget = 1.0;

		// Log final output
		layerOutput.print();

		// Forward-propagation with given tensor
		// super.predict(X);
	};
	
	// Override fit method
	fit = (X, y, args) => {
		super.fit(X, y, args).then(({history}) => {
			console.log(history.loss[history.loss.length-1]);

			// Fire the neurons backward ;)
			this.vArgs.propagationHighlight.x = 1.0;
			this.vArgs.propagationHighlight.xAnim = 1.0;
			this.vArgs.propagationHighlight.xTarget = 0.0;

			// Update all Weight objects' values
			this.updateAllWeights();
		});
	};

	//// Custom methods
	reset = () => {
		// Propagation highlight values
		this.vArgs.propagationHighlight = {
			// x: propagation position
			// xAnim: smoothed x, xAnim = animFn(x)
			// x's target to smoothly go towards
			x: 0.0, xAnim: 0.0, xTarget: 0.0,
			// Width of the highlight and speed of propagation (ratio value for width of the canvas)
			width: 0.02, speed: 0.05,
			// Animation smoothing function
			animFn: AnimationUtils.easeInExpo
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
	updateAllWeights = () => {
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

		// Weight values changed, call it!
		this.onChangeWeights();
	};

	// Should be called when Weight values change
	onChangeWeights = () => {
		// Get all weights in a 1D tensor
		let allWeights = this.getAllWeights().arraySync();

		// Update weight stats
		this.vArgs.weightsStats = {
			min: Math.min(...allWeights),
			max: Math.max(...allWeights),
			mean: (arrSum(allWeights) / allWeights.length)
		};
	};

	// Should be called when Neuron values change
	onChangeNeurons = () => {
		// Get all outputs in an array
		let allOutputs = this.layerNeurons.map(
			neurons => neurons.map(neuron => neuron.value)
		).reduce((a, b) => [...a, ...b], []);

		// Update neuron output stats
		this.vArgs.neuronStats = {
			min: Math.min(...allOutputs),
			max: Math.max(...allOutputs),
			mean: (arrSum(allOutputs) / allOutputs.length)
		};
	};

	// Gets all weights in a 1D tensor
	getAllWeights = () => {
		// Concat all 1D tensors, return it
		return tf.concat(
			// Convert each layer's weights to 1D tensor, put them in a list
			// Slice 1 for excluding the input layer (has no kernel&bias)
			[...Array(this.layers.length).keys()].slice(1).map((layerIndex) => {
				// Get the 2D weight tensor of the layer
				let layerWeightMatrix = this.getLayerWeightMatrix(layerIndex);
				// Flatten the tensor (2D to 1D) and return it
				return layerWeightMatrix.reshape([layerWeightMatrix.size]);
			}),
			0
		);
	};

	// Gets total parameter count of the network
	getTotalParameterCount = () => {
		return this.getAllWeights().size;
	};

	// Draws the whole network, gets called at each frame
	draw = (canvas) => {
		// Get current highlight position, it's target and speed
		let {x, xTarget, speed} = this.vArgs.propagationHighlight;

		// Update propagation highlight position (towards target, smoothly)
		this.vArgs.propagationHighlight.x += (xTarget - x) * speed;

		// Calculate current point with using the animation function
		// Reversing the animation function if going towards negative
		if((xTarget - x) > 0){
			this.vArgs.propagationHighlight.xAnim = this.vArgs.propagationHighlight.animFn(this.vArgs.propagationHighlight.x);
		}else{
			this.vArgs.propagationHighlight.xAnim = 1 - (this.vArgs.propagationHighlight.animFn(1 - this.vArgs.propagationHighlight.x));
		}

		// Get maximum neuron count
		let maxUnitCount = Math.max(...(this.layers.map(layer => 
			(layer.units) || (layer.batchInputShape && layer.batchInputShape[1])
		)));
		// Calculate step per neuron in +Y direction
		let perNeuronY = ((canvas.height * this.vArgs.gapRateY) / maxUnitCount);
		// Calculate step per layer in +X direction
		let perLayerX = ((canvas.width * this.vArgs.gapRateX) / (this.layers.length-1));
		// Calculate X coordinate of starting point of the network
		let startLayerX = ((canvas.width / 2) - (canvas.width * this.vArgs.gapRateX / 2));
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
				neuron.draw(canvas, this.vArgs);
			});
		});

		// Draw weights
		// Each layer
		this.layerWeights.forEach((layer) => {
			// Each neuron
			layer.forEach((neuron) => {
				// Each weight
				neuron.forEach(weight => {
					weight.draw(canvas, this.vArgs);
				});
			});
		});
	};
};

class Neuron{
	value = null;
	visualValue = 0;
	
	x = 0;
	y = 0;
	static r = 30.0;

	constructor(){};

	draw = (canvas, vArgs) => {
		canvas.noFill();

		if(this.value !== null){
			// Smoothly go towards the actual value visually
			this.visualValue += ((this.value - this.visualValue) * vArgs.neuronVisualChangeSpeed);
			// Stroke weight indicates the output value
			let strokeWeight = ((this.visualValue - vArgs.neuronStats.min) / (vArgs.neuronStats.max - vArgs.neuronStats.min));
			
			canvas.stroke(255);
			canvas.strokeWeight(strokeWeight);
		}
		// No output yet
		else{
			canvas.stroke(64);
			canvas.strokeWeight(1);
		}

		canvas.circle(this.x, this.y, Neuron.r);
	};
};

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
	draw = (canvas, vArgs) => {
		// Smoothly go towards the actual value visually
		this.visualValue += ((this.value - this.visualValue) * vArgs.weightVisualChangeSpeed);

		// Stroke weight indicates the carried value
		let strokeWeight = AnimationUtils.easeInExpo((this.visualValue - vArgs.weightsStats.min) / (vArgs.weightsStats.max - vArgs.weightsStats.min));

		// Calculate line's start&end positions
		let fromX = (this.from.x + (Neuron.r/2));
		let fromY = this.from.y;
		let toX = (this.to.x - (Neuron.r/2));
		let toY = this.to.y;

		// Draw weight between neurons
		canvas.stroke(255);
		canvas.strokeWeight(strokeWeight);
		canvas.line(
			// from (neuron)
			fromX, fromY,
			// to (neuron)
			toX, toY
		);

		//// Highlight the connection during propagation
		// If highlighting point reached to the destination, no need to draw anything anymore
		if(abs(vArgs.propagationHighlight.xTarget - vArgs.propagationHighlight.x) < 0.001) return;

		// Calculate highlight area
		let highlightStartX = (canvas.width * vArgs.propagationHighlight.xAnim);
		let highlightEndX = (canvas.width * Math.min(1.0, (vArgs.propagationHighlight.xAnim + vArgs.propagationHighlight.width)));
		let hFromX = max(highlightStartX, fromX);
		let hToX = min(highlightEndX, toX);

		// Check bounds of the highlight area if it intersects with weight line. If not, simply return.
		if(hFromX > toX || hToX > toX || hFromX < fromX || hToX < fromX) return;

		// Calculate highlighting line's Y coordinates with linear interpolation
		// AAAAAAAAH MY MIND
		let hFromY = lerp(fromY, toY, ((hFromX-fromX) / (toX-fromX)));
		let hToY = lerp(fromY, toY, ((hToX-fromX) / (toX-fromX)));

		// Draw the highlighting line!
		canvas.stroke(128);
		canvas.strokeWeight(strokeWeight*2);
		canvas.line(
			// from (highlighter line start point)
			hFromX, hFromY,
			// to (highlighter line end point)
			hToX, hToY
		);
	};
};
