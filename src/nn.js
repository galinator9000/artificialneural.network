// Neural network related variables, functions & classes.

// Creates a dense layer config object with given values
createDenseLayerConfig = (denseArgs={activation: "sigmoid"}) => ({
	class: tf.layers.dense,
	args: {
		// Min 4, max 15 neurons, if not specified
		units: ((denseArgs.units) ? denseArgs.units : getRandomInt(4, 16)),
		// Randomly choose if use bias or not, if not specified
		useBias: ((denseArgs.useBias) ? denseArgs.useBias : (getRandomInt(0, 2)===1)),
		// No activation, if not specified
		activation: denseArgs.activation
	}
});

// Builds tf layer with given config
layerConfigToLayer = (layer) => layer.class(layer.args);

// Our main neural network model
let nn;

// Specifies our neural network structure (layers, losses etc.)
let nnStructure = {
	// Input layer config
	inputLayerConfig: {
		class: tf.layers.inputLayer,
		// (set input unit to 1 initially)
		args: {inputShape: [1]}
	},

	// Hidden layers config
	hiddenLayersConfig: [],

	// Output layer config
	outputLayerConfig: createDenseLayerConfig({
		// (set output unit to 1 initially)
		units: 1,
		useBias: true,
		// Regression
		// activation: null,
		// Classification (binary)
		activation: "sigmoid",
	}),

	// Compile arguments (optimizer, loss)
	compileArgs: {
		optimizer: tf.train.sgd(0.001),

		// Regression
		// loss: "meanSquaredError",
		// Classification (binary)
		loss: "binaryCrossentropy"
	},
};

// Resets neural network
resetNeuralNetwork = () => {
	// Reset hidden layers
	nnStructure.hiddenLayersConfig = [];
	// Remove current nn object
	nn = undefined;
};

// Initializes neural network
initializeNeuralNetwork = () => {
	// Build NN sequentially with our custom class
	nn = new SequentialNeuralNetwork(
		// Arguments which will be passed to tf.Sequential
		sequentialArgs={},

		// Various visual arguments
		vArgs={
			scaleX: 0.75, scaleY: 0.80,
			translateX: -0.025, translateY: -0.035,
			showBiasNeurons: false,
			weightVisualChangeSpeed: 0.25,
			neuronVisualChangeSpeed: 0.25,
			animatePropagation: true,
			propagation: {
				// Width and step values (ratio value for width of the canvas) of the propagation wave
				width: 0.005, step: 0.01,
				// Animation smoothing function
				animFn: AnimationUtils.easeOutQuad,
				// Apply animation layer by layer or to whole network?
				animationApplyType: (1 ? "layer" : "network")
			}
		}
	);

	// Put all layer configs in a list, add each of them to the model
	[
		nnStructure.inputLayerConfig,
		...nnStructure.hiddenLayersConfig,
		nnStructure.outputLayerConfig
	].forEach(layerConfig => nn.add(layerConfigToLayer(layerConfig)));

	// "Precompile" the network for being able to draw it
	nn.precompile();
};

// Compiles neural network
compileNeuralNetwork = () => {
	// Compile the model with args
	nn.compile(nnStructure.compileArgs);
	console.log("NN compiled", nnStructure);
};

// SequentialNeuralNetwork: Built on top of tf.Sequential class, for fully visualizing it
class SequentialNeuralNetwork extends tf.Sequential{
	// vArgs holds our custom values (for visual purposes) for our class
	vArgs = {};

	// These variable holds our Neuron and Weight objects for our layers as a nested-list
	layerNeurons = [];
	layerWeights = [];

	constructor(
		sequentialArgs,
		initialVArgs
	){
		super(sequentialArgs);
		this.vArgs = initialVArgs;
		this.isCompiled = false;
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

		// "Precompile" the network for being able to draw it
		nn.precompile();
	}

	// Override compile method
	compile = (compileArgs) => {
		// Compile the tf.Sequential side
		super.compile(compileArgs);

		// First reset own (custom class) values
		this.reset();

		// Create Neuron & Weight objects
		this.createNeurons();
		this.createWeights(true);

		// Initially call them
		this.onChangeWeights(true);
		this.onChangeNeurons();

		// Mark as compiled
		this.isCompiled = true;
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
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
				neuronOutputs = [1.0, ...neuronOutputs];
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
			this.onChangeWeights(true);

			// Fire the neurons backward ;)
			this.vArgs.propagation.x = 1.0;
			this.vArgs.propagation.xAnim = 1.0;
			this.vArgs.propagation.xTarget = 0.0;
			
			// Log loss output
			console.log("Loss", history.loss[history.loss.length-1]);
		});
	};

	//// Custom methods
	// Create Neuron objects for each of neuron in the network
	createNeurons = () => {
		// Create a nested-list for keeping 'Neuron' objects of each layer
		this.layerNeurons = this.layers.map(layer => []);

		// Create Neuron objects for each layer
		this.layers.forEach((layer, layerIndex) => {
			// Get neuron counts
			let neuronCount = (layer.units) || (layer.batchInputShape && layer.batchInputShape[1]);

			// Check if next layer uses bias
			let nextLayerUsesBias = ((this.layers[layerIndex+1]) && (this.layers[layerIndex+1].useBias === true));
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
				neuronCount++;
			}

			// Each neuron
			[...Array(neuronCount).keys()].forEach(neuronIndex => {
				// Create Neuron object & push it to the list
				this.layerNeurons[layerIndex].push(new Neuron());
			});
		});
	};

	// Create Neuron objects for each of weight in the network
	createWeights = (withRealValues) => {
		// Create a nested-list for keeping 'Weight' objects of each layer
		this.layerWeights = this.layers.slice(1).map(layer => []);

		// Create Weight objects for each layer (with pairing layers)
		[...Array(this.layerNeurons.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.layerNeurons[layerIndex];
			let toLayerNeurons = this.layerNeurons[layerIndex+1];

			// Check if next layer uses bias
			let nextLayerUsesBias = (this.layers[layerIndex+2] && (this.layers[layerIndex+2].useBias === true));
			// If next layer is using bias, there's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
				toLayerNeurons = toLayerNeurons.slice(1);
			};

			// Get or build layer weight matrix
			let layerWeightMatrix;
			if(withRealValues){
				// Get the 2D weight tensor of the layer, turn it into a nested-list
				layerWeightMatrix = this.getLayerWeightMatrix(layerIndex+1).arraySync();
			}else{
				// Build a tensor that has fake values (for being able to draw them before compiling)
				layerWeightMatrix = tf.ones([fromLayerNeurons.length, toLayerNeurons.length]).arraySync();
			}

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
	}

	// Precompile method prepares the variables for being able to draw the network before compiling
	precompile = () => {
		this.createNeurons();
		this.createWeights(false);
		this.onChangeWeights(false);
	};

	reset = () => {
		this.isCompiled = false;

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
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
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
	onChangeWeights = (withRealValues) => {
		// Get all weights in a 1D tensor
		let allWeights;
		if(withRealValues){
			allWeights = this.getAllWeights().arraySync();
		}else{
			allWeights = [1, 1];
		}

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

	// Updates the network, gets called at each main loop of the sketch
	update = (canvas) => {
		this.vArgs.nnIsCompiled = nn.isCompiled;

		//// Calculate values for drawing
		// Get maximum neuron count
		this.vArgs.maxUnitCount = Math.max(...(this.layers.map(layer => 
			(layer.units) || (layer.batchInputShape && layer.batchInputShape[1])
		)));
		// Calculate step per neuron in +Y direction
		this.vArgs.perNeuronY = ((canvas.height * this.vArgs.scaleY) / (
			// Limit maximum neuron size when there's few neurons
			Math.max(10, this.vArgs.maxUnitCount)
		));
		// Calculate step per layer in +X direction
		this.vArgs.perLayerX = ((canvas.width * this.vArgs.scaleX) / (this.layers.length-1));
		// Calculate X coordinate of starting point of the network
		this.vArgs.startLayerX = (canvas.width * ((1-this.vArgs.scaleX) / 2));
		// Calculate each neuron size with using step per neuron value
		Neuron.r = (this.vArgs.perNeuronY / 1.25 / 2);

		//// Update propagation related values
		// Update the function for calculating the real X position of the wave with canvas width/gap value
		this.vArgs.propagation.xToCanvasPosX = ((curX) => (
			(this.vArgs.startLayerX - Neuron.r) + (curX * ((canvas.width * this.vArgs.scaleX) + Neuron.r*2)) + (canvas.width * this.vArgs.translateX)
		));

		// Update propagation wave position (set directly or go towards to target smoothly)
		if(this.vArgs.animatePropagation){
			if(this.vArgs.propagation.xTarget > this.vArgs.propagation.x){
				this.vArgs.propagation.x += this.vArgs.propagation.step;
			}
			else if(this.vArgs.propagation.xTarget < this.vArgs.propagation.x){
				this.vArgs.propagation.x -= this.vArgs.propagation.step;
			}
			// Limit between 0 and 1
			this.vArgs.propagation.x = Math.min(1, Math.max(0, this.vArgs.propagation.x));
		}else{
			this.vArgs.propagation.x = this.vArgs.propagation.xTarget;
		}

		//// Calculate current propagation wave point with using the animation function
		// Apply animation layer by layer (looks nicer ;D)
		if(this.vArgs.animatePropagation){
			if(this.vArgs.propagation.animationApplyType == "layer"){
				let eachLayerX = 1 / (this.layers.length-1);
				let currentLayerIdx = Math.floor(this.vArgs.propagation.x / eachLayerX);
				let currentLayerX = (this.vArgs.propagation.x % eachLayerX) / eachLayerX;
	
				// Reversing the animation function if going towards negative
				if((this.vArgs.propagation.xTarget - this.vArgs.propagation.x) > 0){
					this.vArgs.propagation.xAnim = (currentLayerIdx * eachLayerX) + (this.vArgs.propagation.animFn(currentLayerX) / (this.layers.length - 1));
				}else{
					this.vArgs.propagation.xAnim = (currentLayerIdx * eachLayerX) + ((1 - this.vArgs.propagation.animFn(1 - currentLayerX)) / (this.layers.length - 1));
				}
			}
			// Apply animation to whole network
			else if(this.vArgs.propagation.animationApplyType == "network"){
				// Reversing the animation function if going towards negative
				if((this.vArgs.propagation.xTarget - this.vArgs.propagation.x) > 0){
					this.vArgs.propagation.xAnim = this.vArgs.propagation.animFn(this.vArgs.propagation.x);
				}else{
					this.vArgs.propagation.xAnim = (1 - this.vArgs.propagation.animFn(1 - this.vArgs.propagation.x));
				}
			}
		}

		//// Call update methods of Neuron&Weight objects
		// Each layer
		this.layerNeurons.forEach((layer, layerIndex) => {
			// Calculate starting point (Y-coordinate of first neuron) of the layer
			// Top of the layer in Y = ((Center of the neural network in Y) - (layer size in Y / 2)) + (applying Y shift a bit for centering)
			let startNeuronY = ((canvas.height/2) - ((this.vArgs.perNeuronY * layer.length) / 2)) + (this.vArgs.perNeuronY/2);

			// Each neuron
			layer.forEach((neuron, neuronIndex) => {
				// Calculate new position of neuron & update it
				let newXpos = (this.vArgs.startLayerX + (this.vArgs.perLayerX * layerIndex) + (canvas.width * this.vArgs.translateX));
				let newYpos = (startNeuronY + (this.vArgs.perNeuronY * neuronIndex) + (canvas.height * this.vArgs.translateY));
				neuron.update(newXpos, newYpos, this.vArgs);
			});
		});

		// Each layer
		this.layerWeights.forEach((layer) => {
			// Each neuron
			layer.forEach((neuron) => {
				// Each weight
				neuron.forEach(weight => {
					weight.update(this.vArgs);
				});
			});
		});
	};

	// Process mouse click on the network
	mousePressed = (mouseX, mouseY) => {
		// Check if clicked on any of the neurons
		this.layerNeurons.forEach(
			layer => layer.forEach(
				neuron => {
					// Check if distance is less than the radius
					let distanceToPress = dist(
						neuron.x, neuron.y,
						mouseX, mouseY
					);
					// If it's already in focus, defocus it
					neuron.isFocused = ((!neuron.isFocused) && (distanceToPress < Neuron.r));
				}
			)
		);
	};

	// Draws the whole network on the given canvas, gets called if subcanvas is active
	draw = (canvas, sample) => {
		// Update the value of: if any of the neurons are focused
		this.vArgs.focusedAnyNeuron = arrBoolAny(
			this.layerNeurons.map(
				layer => arrBoolAny(
					layer.map(
						neuron => neuron.isFocused
					)
				)
			)
		);

		//// Draw weights
		// Each layer
		this.layerWeights.forEach(layer => {
			// Each neuron
			layer.forEach((neuron) => {
				// Each weight
				neuron.forEach(weight => {
					weight.draw(canvas, this.vArgs);
				});
			});
		});

		//// Draw neurons
		// Each layer
		this.layerNeurons.forEach(layer => {
			// Each neuron
			layer.forEach((neuron) => {
				neuron.draw(canvas, this.vArgs);
			});
		});

		//// Draw sample input/targets to the side of the network
		if(this.isCompiled && (sample && sample.input && sample.target)){
			// Get the sample (currently tensor) in a nested-list
			let inputVec = sample.input.arraySync()[0];
			let targetVec = sample.target.arraySync()[0];

			// Check if next layer uses bias
			let nextLayerUsesBias = (this.layers[1] && (this.layers[1].useBias === true));
			// If next layer is using bias, add value 1 to the top of the input
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
				inputVec = [1.0, ...inputVec];
			};

			// Draw input values to the left side
			this.layerNeurons[0].forEach((inputNeuron, neuronIndex) => {
				let posX = (inputNeuron.x - (Neuron.r*3));
				let posY = inputNeuron.y;

				// Value rect outline
				canvas.push();
				canvas.rect(posX, posY, Neuron.r*3, Neuron.r*2);
				canvas.pop();

				// Value as text
				let vText = inputVec[neuronIndex].toFixed(2);
				canvas.push();
				canvas.fill(255);
				canvas.textSize(calculateTextSize(vText, Neuron.r*3, Neuron.r*2));
				canvas.text(
					vText,
					posX, posY
				);
				canvas.pop();
			});

			// Draw target values to the right side
			this.layerNeurons[this.layerNeurons.length-1].forEach((outputNeuron, neuronIndex) => {
				let posX = (outputNeuron.x + (Neuron.r*3));
				let posY = outputNeuron.y;

				// Value rect outline
				canvas.push();
				canvas.rect(posX, posY, Neuron.r*3, Neuron.r*2);
				canvas.pop();

				// Value as text
				let vText = targetVec[neuronIndex].toFixed(2);
				canvas.push();
				canvas.fill(255);
				canvas.textSize(calculateTextSize(vText, Neuron.r*3, Neuron.r*2));
				canvas.text(
					vText,
					posX, posY
				);
				canvas.pop();
			});
		}
	};
};

// Neuron object represents every output unit of the fully-connected layer, holds output value and XY position on the subcanvas.
class Neuron{
	value = null;
	visualValue = 0;
	isFocused = false;

	x = 0;
	y = 0;
	static r = 30.0;

	// Sets visual value
	updateVisualValue = (vArgs) => {
		// Smoothly go towards the actual value visually
		this.visualValue += ((this.value - this.visualValue) * vArgs.neuronVisualChangeSpeed);
		// Set directly if it's close enough to the target
		if(abs(this.value - this.visualValue) < 0.001){
			this.visualValue = this.value;
		}
	};

	// Gets called every frame
	update = (newXpos, newYpos, vArgs) => {
		this.x = newXpos;
		this.y = newYpos;

		// If no output yet, simply don't update any value
		if(this.value !== null){
			// Adjust the visible value after propagation (forward) wave passes over it
			if(
				(
					(vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim) >= this.x)
					&& ((vArgs.propagation.xTarget - vArgs.propagation.xAnim) > 0)
				) || (!vArgs.animatePropagation)
			){
				this.updateVisualValue(vArgs);
			}
		}
	};

	draw = (canvas, vArgs) => {
		// Neuron circle
		canvas.push();
		canvas.fill(BG_COLOR);
		canvas.stroke(255);
		canvas.strokeWeight(
			(this.isFocused && vArgs.focusedAnyNeuron)
			// When focused
			? 2
			// When isn't
			: 1
		);
		canvas.circle(this.x, this.y, Neuron.r*2);
		canvas.pop();

		// Draw the output value as text
		if(vArgs.nnIsCompiled){
			canvas.push();
			canvas.fill(255);
			canvas.stroke(255);
			canvas.strokeWeight(
				(this.isFocused && vArgs.focusedAnyNeuron)
				// When focused
				? 2
				// When isn't
				: 1
			);

			let vText = this.visualValue.toFixed(2);
			canvas.textSize(calculateTextSize(vText, Neuron.r*2, Neuron.r*2));
			canvas.text(
				vText,
				this.x, this.y
			);
			canvas.pop();
		}
	};
};

// Weight objects are connected to two separate Neuron objects: "from" and "to".
class Weight{
	value = null;
	visualValue = null;
	isFocused = false;

	constructor(from, to, value){
		this.from = from;
		this.to = to;
		this.value = value;
		this.visualValue = value;
	};

	// Sets visual value
	updateVisualValue = (vArgs) => {
		// Smoothly go towards the actual value visually
		this.visualValue += ((this.value - this.visualValue) * vArgs.weightVisualChangeSpeed);
		// Set directly if it's close enough to the target
		if(abs(this.value - this.visualValue) < 0.001){
			this.visualValue = this.value;
		}
	};

	// Gets called every frame
	update = (vArgs) => {
		// Adjust the visual value after propagation (backward) wave passes over it
		if(
			(
				(vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim) <= this.from.x)
				&& ((vArgs.propagation.xTarget - vArgs.propagation.xAnim) < 0)
			) || (!vArgs.animatePropagation)
		){
			this.updateVisualValue(vArgs);
		}
	}

	draw = (canvas, vArgs) => {
		// If focused on a neuron, but none of the connected neurons are focused ones, then no need to draw this weight
		if(vArgs.focusedAnyNeuron && !(this.from.isFocused || this.to.isFocused)){
			return;
		}

		// Check if weight is focused
		this.isFocused = (vArgs.focusedAnyNeuron && (this.from.isFocused || this.to.isFocused));

		// Check if should write the carried value as text
		let writeCarriedValue = (this.isFocused && vArgs.nnIsCompiled);

		// Calculate line's start&end positions
		let fromX = (this.from.x + Neuron.r);
		let fromY = this.from.y;
		let toX = (this.to.x - Neuron.r);
		let toY = this.to.y;
		let gapStartX, gapStartY, gapEndX, gapEndY;

		// Draw weight between neurons (from -> to) as a line
		if(!writeCarriedValue){
			canvas.push();
			canvas.fill(255);
			canvas.stroke(255);
			canvas.strokeWeight(
				this.isFocused
				// When focused
				? 2
				// When isn't
				: 1
			);
			canvas.line(
				// from (neuron)
				fromX, fromY,
				// to (neuron)
				toX, toY
			);
			canvas.pop();
		}
		// Draw the carried value as text between the connection if focused
		else{
			// Text value gap center point and width
			let gapCenter = 0.50;
			// Adjust gapcenter value for drawing text values in a circular path
			if(this.from.isFocused) gapCenter = 0.70;
			if(this.to.isFocused) gapCenter = 0.30;

			let gapWidth = 0.15;

			// Text value gap XY points, linear interpolation from neurons' XY values
			gapStartX = lerp(fromX, toX, (gapCenter - (gapWidth/2)));
			gapStartY = lerp(fromY, toY, (gapCenter - (gapWidth/2)));
			gapEndX = lerp(fromX, toX, (gapCenter + (gapWidth/2)));
			gapEndY = lerp(fromY, toY, (gapCenter + (gapWidth/2)));

			// Draw the line with a gap on the center
			canvas.push();
			canvas.fill(255);
			canvas.stroke(255);
			canvas.strokeWeight(1);
			canvas.line(fromX, fromY, gapStartX, gapStartY);
			canvas.line(gapEndX, gapEndY, toX, toY);
			canvas.pop();

			// Draw the value as text on the center of the gap with rotating accordingly
			canvas.push();
			canvas.fill(255);
			canvas.stroke(255);
			canvas.strokeWeight(1);
			canvas.translate(
				lerp(fromX, toX, gapCenter),
				lerp(fromY, toY, gapCenter)
			);
			canvas.rotate(
				createVector(toX-fromX, 0).normalize().angleBetween(
					createVector(toX-fromX, toY-fromY).normalize()
				)
			);
			let vText = this.visualValue.toFixed(2);
			canvas.textSize(
				calculateTextSize(
					"    ",
					// Get width&height as distance between gap endpoints
					(gapEndX-gapStartX),
					(gapEndY-gapStartY)
				)
			);
			canvas.text(vText, 0, 0);
			canvas.pop();
		}

		//// Highlight the connection during propagation
		// If highlighting point reached to the destination, no need to draw anything anymore
		if(abs(vArgs.propagation.xTarget - vArgs.propagation.x) < 0.001) return;

		// Calculate highlight area
		let hFromX = vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim);
		let hToX = vArgs.propagation.xToCanvasPosX(Math.min(1.0, (vArgs.propagation.xAnim + vArgs.propagation.width)));
		hFromX = max(hFromX, fromX);
		hToX = min(hToX, toX);

		// Check bounds of the highlight area if it intersects with weight line. If not, simply return. (also checks the gap)
		if(
			(hFromX > toX || hFromX < fromX) || (hToX > toX || hToX < fromX)
			// Gap condition
			|| (writeCarriedValue && (
				(hFromX > gapStartX && hFromX < gapEndX)
				&& (hToX > gapStartX && hToX < gapEndX)
			))
		) return;

		// Adjust highlight position values according to gap :P
		if(writeCarriedValue){
			if(hFromX < gapStartX && hToX > gapStartX){
				hToX = gapStartX;
			}else if(hFromX < gapEndX && hToX > gapEndX){
				hFromX = gapEndX;
			}
		}

		// Calculate highlighting line's Y coordinates with linear interpolation
		// AAAAAAAAH MY MIND
		let hFromY = lerp(fromY, toY, ((hFromX-fromX) / (toX-fromX)));
		let hToY = lerp(fromY, toY, ((hToX-fromX) / (toX-fromX)));

		// Draw the highlighting line!
		canvas.push();
		canvas.stroke(255);
		canvas.strokeWeight(3);
		canvas.line(
			// from (highlighter line start point)
			hFromX, hFromY,
			// to (highlighter line end point)
			hToX, hToY
		);
		canvas.pop();
	};
};
