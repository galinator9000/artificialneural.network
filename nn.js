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
			let nextLayerUsesBias = ((this.layers[layerIndex+1]) && (this.layers[layerIndex+1].useBias === true));
			if(nextLayerUsesBias){
				neuronCount++;
			}

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
			let nextLayerUsesBias = (this.layers[layerIndex+2] && (this.layers[layerIndex+2].useBias === true));
			// If next layer is using bias, there's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(nextLayerUsesBias){
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
							// Carried value, get it from the matrix
							layerWeightMatrix[fromNeuronIndex][toNeuronIndex]
						)
					);
				});
			});
		});

		// Initially call them.
		this.onChangeWeights();
		this.onChangeNeurons();

		// Update all weights' outer look initially by calling their updateOuterLook method
		this.layerWeights.forEach(layer => {
			layer.forEach(neuron => {
				neuron.forEach(weight => {
					weight.updateOuterLook(this.vArgs);
				});
			});
		});
	};
	
	// Override predict method
	predict = (X) => {
		// If provided more than one sample, predict it and simply return, no need to visualize
		if(X.shape[0] > 1){
			// Forward-propagation with given tensor
			return super.predict(X);
		}

		// Feed layer by layer
		let layerOutput = X;
		this.layers.forEach((layer, layerIndex) => {
			// Feed to current layer & get output as tensor
			layerOutput = layer.call(layerOutput);

			// Get the output in a nested-list
			let neuronOutputs = layerOutput.arraySync()[0];

			// Check if next layer uses bias
			let nextLayerUsesBias = (this.layers[layerIndex+1] && (this.layers[layerIndex+1].useBias === true));
			// If next layer is using bias, also add 1 value to the top of the current output
			if(nextLayerUsesBias){
				neuronOutputs = [1, ...neuronOutputs];
			};

			// Set each neuron's output
			neuronOutputs.forEach((neuronOutput, neuronIndex) => {
				// Set neuron's output
				this.layerNeurons[layerIndex][neuronIndex].value = neuronOutput;
			});
		});

		// Neuron values changed, call it!
		this.onChangeNeurons();
		
		// Fire the neurons forward ;)
		this.vArgs.propagation.x = 0.0;
		this.vArgs.propagation.xAnim = 0.0;
		this.vArgs.propagation.xTarget = 1.0;

		// Log final output
		console.log("Prediction", layerOutput.toString());
	};
	
	// Override fit method
	fit = (X, y, args) => {
		super.fit(X, y, args).then(({history}) => {
			// Update all Weight objects' values
			this.updateAllWeights();

			// Weight values changed, call it!
			this.onChangeWeights();

			// Fire the neurons backward ;)
			this.vArgs.propagation.x = 1.0;
			this.vArgs.propagation.xAnim = 1.0;
			this.vArgs.propagation.xTarget = 0.0;
			
			// Log loss output
			console.log("Loss", history.loss[history.loss.length-1]);
		});
	};

	//// Custom methods
	reset = () => {
		// Propagation wave values
		this.vArgs.propagation = {
			...this.vArgs.propagation,
			// x: propagation position
			x: 0.0,
			// xAnim: smoothed x, xAnim = animFn(x)
			xAnim: 0.0,
			// x's target to smoothly go towards
			xTarget: 0.0
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

	// Updates all carried values of weight objects
	updateAllWeights = () => {
		[...Array(this.layerNeurons.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.layerNeurons[layerIndex];
			let toLayerNeurons = this.layerNeurons[layerIndex+1];

			// Check if next layer uses bias
			let nextLayerUsesBias = (this.layers[layerIndex+2] && (this.layers[layerIndex+2].useBias === true));
			// If next layer is using bias, there's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(nextLayerUsesBias){
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
		//// Update propagation related values
		// Get current wave position, target and speed
		let {x, xTarget, speed} = this.vArgs.propagation;

		// Update propagation wave position (towards target, smoothly)
		if(xTarget > x) this.vArgs.propagation.x += speed;
		if(xTarget < x) this.vArgs.propagation.x -= speed;
		
		// Limit between 0 and 1
		this.vArgs.propagation.x = Math.min(
			1,
			Math.max(
				0,
				this.vArgs.propagation.x
			)
		);

		// Calculate current point with using the animation function
		// Reversing the animation function if going towards negative
		if((xTarget - x) > 0){
			this.vArgs.propagation.xAnim = this.vArgs.propagation.animFn(this.vArgs.propagation.x);
		}else{
			this.vArgs.propagation.xAnim = 1 - (this.vArgs.propagation.animFn(1 - this.vArgs.propagation.x));
		}

		//// Calculate values for drawing
		// Get maximum neuron count
		let maxUnitCount = Math.max(...(this.layers.map(layer => 
			(layer.units) || (layer.batchInputShape && layer.batchInputShape[1])
		)));
		// Calculate step per neuron in +Y direction
		let perNeuronY = ((canvas.height * this.vArgs.gapRateY) / maxUnitCount);
		// Calculate step per layer in +X direction
		let perLayerX = ((canvas.width * this.vArgs.gapRateX) / (this.layers.length-1));
		// Calculate X coordinate of starting point of the network
		let startLayerX = (canvas.width * ((1-this.vArgs.gapRateX) / 2));
		// Calculate each neuron size with using step per neuron value
		Neuron.r = (perNeuronY / 1.25);

		// Update the function for calculating the real X position of the wave with canvas width/gap value
		this.vArgs.propagation.xToCanvasPosX = ((curX) => (
			(startLayerX - (Neuron.r/2)) + (curX * ((canvas.width * this.vArgs.gapRateX) + Neuron.r))
		));

		//// Draw neurons
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

		//// Draw weights
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
	// Visual values (indicates output)
	strokeWeight = 0.5;
	stroke = 255;
	fill = 0;

	value = null;
	visualValue = 0;
	x = 0;
	y = 0;
	static r = 30.0;
	constructor(){};

	// Updates visual values
	updateOuterLook = (vArgs) => {
		// These values indicates the output value
		this.strokeWeight = ((this.visualValue - vArgs.neuronStats.min) / (vArgs.neuronStats.max - vArgs.neuronStats.min));
		this.stroke = (1-this.strokeWeight)*255;
		this.fill = this.strokeWeight*255;
	};

	// Gets called every frame
	draw = (canvas, vArgs) => {
		// If no output yet, simply don't update any value
		if(this.value !== null){
			// Adjust the visible value after propagation (forward) wave passes over it
			if(
				(vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim) >= this.x)
				&& ((vArgs.propagation.xTarget - vArgs.propagation.xAnim) > 0)
			){
				// Smoothly go towards the actual value visually
				this.visualValue += ((this.value - this.visualValue) * vArgs.neuronVisualChangeSpeed);
				// Set directly if it's close enough to the target
				if(abs(this.value - this.visualValue) < 0.001){
					this.visualValue = this.value;
				}

				this.updateOuterLook(vArgs);
			}
		}

		// Neuron as circle
		canvas.strokeWeight(this.strokeWeight);
		canvas.stroke(this.stroke);
		canvas.fill(this.fill);
		canvas.circle(this.x, this.y, Neuron.r);

		// Draw the value as text
		canvas.fill(this.stroke);
		canvas.textAlign(CENTER, CENTER);
		canvas.textFont(vArgs.neuronValueFont);
		canvas.textSize(12);
		canvas.text(
			this.visualValue.toFixed(2),
			this.x, this.y
		);
	};
};

class Weight{
	// Visual values (indicates carried value)
	strokeWeight = 0.5;
	stroke = 255;

	value = 0;
	visualValue = 0;
	constructor(from, to, value){
		this.from = from;
		this.to = to;
		this.value = value;
		this.visualValue = value;
	};

	// Updates visual values
	updateOuterLook = (vArgs) => {
		// These values indicates the carried value
		this.strokeWeight = AnimationUtils.easeInExpo((this.visualValue - vArgs.weightsStats.min) / (vArgs.weightsStats.max - vArgs.weightsStats.min));
		this.stroke = (1-this.strokeWeight)*255;
	};

	// Gets called every frame
	draw = (canvas, vArgs) => {
		// Adjust the visual value after propagation (backward) wave passes over it
		if(
			(vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim) <= this.from.x)
			&& ((vArgs.propagation.xTarget - vArgs.propagation.xAnim) < 0)
		){
			// Smoothly go towards the actual value visually
			this.visualValue += ((this.value - this.visualValue) * vArgs.weightVisualChangeSpeed);
			// Set directly if it's close enough to the target
			if(abs(this.value - this.visualValue) < 0.001){
				this.visualValue = this.value;
			}

			this.updateOuterLook(vArgs);
		}

		// Calculate line's start&end positions
		let fromX = (this.from.x + (Neuron.r/2));
		let fromY = this.from.y;
		let toX = (this.to.x - (Neuron.r/2));
		let toY = this.to.y;

		// Draw weight between neurons
		canvas.stroke(this.stroke);
		canvas.strokeWeight(this.strokeWeight);
		canvas.line(
			// from (neuron)
			fromX, fromY,
			// to (neuron)
			toX, toY
		);

		//// Highlight the connection during propagation
		// If highlighting point reached to the destination, no need to draw anything anymore
		if(abs(vArgs.propagation.xTarget - vArgs.propagation.x) < 0.001) return;

		// Calculate highlight area
		let highlightStartX = vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim);
		let highlightEndX = vArgs.propagation.xToCanvasPosX(Math.min(1.0, (vArgs.propagation.xAnim + vArgs.propagation.width)));
		let hFromX = max(highlightStartX, fromX);
		let hToX = min(highlightEndX, toX);

		// Check bounds of the highlight area if it intersects with weight line. If not, simply return.
		if(hFromX > toX || hToX > toX || hFromX < fromX || hToX < fromX) return;

		// Calculate highlighting line's Y coordinates with linear interpolation
		// AAAAAAAAH MY MIND
		let hFromY = lerp(fromY, toY, ((hFromX-fromX) / (toX-fromX)));
		let hToY = lerp(fromY, toY, ((hToX-fromX) / (toX-fromX)));

		// Draw the highlighting line!
		canvas.stroke(this.stroke/2);
		canvas.strokeWeight(this.strokeWeight*5);
		canvas.line(
			// from (highlighter line start point)
			hFromX, hFromY,
			// to (highlighter line end point)
			hToX, hToY
		);
	};
};
