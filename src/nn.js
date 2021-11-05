// Neural network related variables, functions & classes.

// Various (configurable) visual arguments
var nnVArgs = {
	scaleX: 0.80, scaleY: 0.66,
	translateX: -0.035, translateY: -0.025,
	showBiasNeurons: false,
	weightVisualChangeSpeed: 0.25,
	neuronVisualChangeSpeed: 0.25,
	animatePropagation: true,
	propagation: {
		// Width and step values (ratio value for width of the canvas) of the propagation wave
		width: 0.005, step: 0.0125,
		// Animation smoothing function
		animFn: AnimationUtils.easeOutQuad,
		// Apply animation layer by layer or to whole network?
		animationApplyType: (1 ? "layer" : "network"),
		inProgress: false
	},
	predicted: false,
	backpropagated: false,
	autoTrain: {
		isEnabled: false,
		inProgress: false,
	},
	status: {
		text: "",
		defaultText: "",
	},
	isDummy: false,
};

// Creates a dense layer config object with given values
createDenseLayerConfig = (denseArgs={}) => ({
	class: tf.layers.dense,
	args: {
		// Min 4, max 12 neurons, if not specified
		units: ((denseArgs.units) ? denseArgs.units : getRandomInt(4, 12)),
		// Use bias if not specified
		useBias: ((denseArgs.useBias) ? denseArgs.useBias : true),
		// No activation, if not specified
		activation: ((denseArgs.activation) ? denseArgs.activation : null),
	}
});

// Builds tf layer with given config
layerConfigToLayer = (layer) => layer.class(layer.args);

// Our main neural network model
var nn;

// Specifies our neural network structure (layers, losses etc.)
var nnStructure = {
	// Input layer config
	inputLayerConfig: {
		class: tf.layers.inputLayer,
		// (set input unit to 1 initially)
		args: {inputShape: [1]}
	},

	// Hidden layers config, add two layer initially
	hiddenLayersConfig: [
		createDenseLayerConfig({activation: "linear"}),
		createDenseLayerConfig({activation: "linear"})
	],

	// Output layer config
	outputLayerConfig: createDenseLayerConfig({
		// (set output unit to 1 initially)
		units: 1,
		useBias: true,
		activation: "linear"
	}),

	// Compile arguments (optimizer, loss)
	compileArgs: {
		optimizer: "sgd",
		loss: "meanSquaredError",
		learningRate: 0.0001
	},

	//// Options
	activationFunctionOptions: {
		"Linear": "linear",
		"ReLU": "relu",
		"Sigmoid": "sigmoid",
		"Tanh": "tanh",
		"Softmax": "softmax",
		"Softplus": "softplus",

		// "ELU": "elu",
		// "Hard Sigmoid": "hardSigmoid",
		// "ReLU6": "relu6",
		// "SeLU": "selu",
		// "Softsign": "softsign",
		// "Swish": "swish",
		// "Mish": "mish",
	},
	lossFunctionOptions: {
		"Mean Squared Error": "meanSquaredError",
		"SigmoidCrossEntropy": "sigmoidCrossEntropy",
		"SoftmaxCrossEntropy": "softmaxCrossEntropy",

		// "Absolute Difference": "absoluteDifference",
		// "Compute Weighted": "computeWeightedLoss",
		// "Cosine Dist": "cosineDistance",
		// "Hinge Loss": "hingeLoss",
		// "Huber Loss": "huberLoss",
		// "Log Loss": "logLoss",
	},
	optimizerOptions: {
		"SGD": "sgd",
		"Adam": "adam",
		
		// "RMSprop": "rmsprop",
		// "Momentum": "momentum",
		// "Adagrad": "adagrad",
		// "Adadelta": "adadelta",
		// "Adamax": "adamax",
	},

	// Whether apply limits to some configurations of the network or not
	applyLimits: true,
	limits: {
		maxHiddenUnitCount: 16,
		maxHiddenLayerCount: 3
	}
};

// Resets neural network
resetNeuralNetwork = () => {
	// Remove current nn object
	nn = undefined;
	// Reset NN GUI (dynamic components)
	resetNeuralNetworkGUI();
};

// Builds neural network
buildNeuralNetwork = () => {
	// Build NN sequentially with our custom class
	nn = new SequentialNeuralNetwork(
		// Arguments which will be passed to tf.Sequential
		sequentialArgs={},
		// Pass visual arguments
		vArgs=nnVArgs
	);

	// Put all layer configs in a list, add each of them to the model
	[
		nnStructure.inputLayerConfig,
		...nnStructure.hiddenLayersConfig,
		nnStructure.outputLayerConfig
	].forEach(layerConfig => nn.add(layerConfigToLayer(layerConfig)));

	// "Precompile" the network for being able to draw it
	nn.precompile();

	// Update network once initially
	nn.update(subCanvas.c[NN_SUBCANVAS_INDEX].obj);

	// Reset NN GUI (dynamic components)
	resetNeuralNetworkGUI();
};

// Compiles neural network
compileNeuralNetwork = () => {
	// Compile the model with args
	nn.compile({
		// Call optimizer fn with learning rate
		optimizer: tf.train[nnStructure.compileArgs.optimizer](nnStructure.compileArgs.learningRate),
		// Direcly pass the loss fn class
		loss: tf.losses[nnStructure.compileArgs.loss]
	});
	console.log("NN compiled", nnStructure);
	// Reset NN GUI (dynamic components)
	resetNeuralNetworkGUI();
};

// Resets dynamic NN GUI components
resetNeuralNetworkGUI = () => {
	// Remove all GUI components that configures NN
	getGUIComponentIDs().filter(gcId => gcId.startsWith("nn_cfg")).map(gcId => {removeGUIComponentWithID(gcId)});

	// Add new GUI components if NN is built (buildNeuralNetwork fn)
	if((nn === undefined)) return;

	// ... all goes into new GUI components' configs
	let nnGUIComponentDefaults = {
		subCanvasIndex: NN_SUBCANVAS_INDEX,
		attributes: [
			// "Disabled" attribute if NN is built
			{
				name: "disabled", value: "",
				condition: () => ((nn && (nn.isCompiled === true)))
			}
		],
	};

	// Dense layer config getter&setter util fn
	denseConfigGetter = (denseLayerIdx, cfgName) => {
		// Return hidden layer's config
		if(denseLayerIdx < nn.layerNeurons.length-1) return nnStructure.hiddenLayersConfig[denseLayerIdx-1].args[cfgName];
		// Return output layer's  config
		else return nnStructure.outputLayerConfig.args[cfgName];
	};
	denseConfigSetter = (denseLayerIdx, cfgName, newValue) => {
		// Set hidden layer's config
		if(denseLayerIdx < nn.layerNeurons.length-1){
			nnStructure.hiddenLayersConfig[denseLayerIdx-1].args[cfgName] = newValue;
		}
		// Set output layer's config
		else{
			nnStructure.outputLayerConfig.args[cfgName] = newValue;
		}
	};

	//// Network structure GUI components (disappears after compilation)
	if(!(nn && (nn.isCompiled === true))){
		// Loss function selector
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_cost",
			obj: createSelect(),
			initCalls: [
				// Add all losses as option
				...(Object.entries(nnStructure.lossFunctionOptions).map(([key, value]) => ({fnName: "option", args: [key, value]}))),
				// Set the current one as selected
				{fnName: "selected", args: [nnStructure.compileArgs.loss]},

				// changed event
				{fnName: "changed", args: [
					(event) => {
						// Get new value and set config
						let value = getGUIComponentWithID("nn_cfg_cost").obj.value();
						nnStructure.compileArgs.loss = value;
						// Rebuild the network?
						// buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.71, 0.96],
			canvasRelativeSize: [0.12, 0.04]
		});
		// Cost function text
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title_cost",
			obj: createButton(`Cost function`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.71, 0.92],
			canvasRelativeSize: [0.12, 0.04]
		});

		// Optimizer selector
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_optimizer",
			obj: createSelect(),
			// "Disabled" attribute for optimizer select
			attributes: [{name: "disabled", value: "", condition: () => true}],
			initCalls: [
				// Add all optimizers as option
				...(Object.entries(nnStructure.optimizerOptions).map(([key, value]) => ({fnName: "option", args: [key, value]}))),
				// Set the current one as selected
				{fnName: "selected", args: [nnStructure.compileArgs.optimizer]},

				// changed event
				{fnName: "changed", args: [
					(event) => {
						// Get new value and set config
						let value = getGUIComponentWithID("nn_cfg_optimizer").obj.value();
						nnStructure.compileArgs.optimizer = value;
						// Rebuild the network?
						// buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.55, 0.96],
			canvasRelativeSize: [0.06, 0.04]
		});

		// Learning rate input
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_learning_rate",
			obj: createInput(nnStructure.compileArgs.learningRate.toString()),
			initCalls: [
				// changed event
				{fnName: "changed", args: [
					(event) => {
						// Get new value and set layer config
						let gc = getGUIComponentWithID("nn_cfg_learning_rate").obj;
						let value = gc.value();

						// Try to convert the string to number and check the validity of the input
						if(typeof(value) !== "number") value = Number(value);
						if(typeof(value) !== "number") return;
						nnStructure.compileArgs.learningRate = value;
						// Rebuild the network?
						// buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.61, 0.96],
			canvasRelativeSize: [0.04, 0.04]
		});

		// Optimizer text
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title_optimizer",
			obj: createButton(`Optimizer`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.58, 0.92],
			canvasRelativeSize: [0.12, 0.04]
		});

		let centerY = 0.96;
		let centerYstep = 0.06;

		// "Add hidden layer" button
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_add_hidden_layer",
			obj: createButton("+ Add hidden layer"),
			initCalls: [
				{fnName: "mousePressed", args: [
					(() => {
						// Check hidden layer count limit
						if(nnStructure.applyLimits && (nnStructure.hiddenLayersConfig.length >= nnStructure.limits.maxHiddenLayerCount)){
							return;
						}

						// Add (randomly configured) hidden layer to the list
						nnStructure.hiddenLayersConfig.push(createDenseLayerConfig());
						// Rebuild the network
						buildNeuralNetwork();
					})
				]},
			],
			canvasRelativePosition: [0.90, centerY],
			canvasRelativeSize: [0.12, 0.04]
		});
		centerY -= centerYstep;

		// Output layer text
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title_output",
			obj: createButton(`Output (${nnStructure.outputLayerConfig.args.units})`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.825, centerY],
			canvasRelativeSize: [0.07, 0.04]
		});

		// Output layer "Use bias" button
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_dense_layer_bias_output",
			obj: createButton(`${denseConfigGetter((nn.layerNeurons.length-1), "useBias") ? "true" : "false"}`),
			initCalls: [
				{fnName: "mousePressed", args: [
					(event) => {
						// Get new value and set layer config
						denseConfigSetter((nn.layerNeurons.length-1), "useBias", !denseConfigGetter((nn.layerNeurons.length-1), "useBias"));
						// Rebuild the network
						buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.89, centerY],
			canvasRelativeSize: [0.04, 0.04]
		});

		// Output layer activation function selector
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_dense_layer_activation_output",
			obj: createSelect(),
			initCalls: [
				// Add all activations as option
				...(Object.entries(nnStructure.activationFunctionOptions).map(([key, value]) => ({fnName: "option", args: [key, value]}))),
				// Set the current one as selected
				...((denseConfigGetter(nn.layerNeurons.length-1, "activation") === null) ? [] : [{fnName: "selected", args: [denseConfigGetter(nn.layerNeurons.length-1, "activation")]}]),

				// changed event
				{fnName: "changed", args: [
					(event) => {
						// Get new value and set layer config
						let value = getGUIComponentWithID("nn_cfg_dense_layer_activation_output").obj.value();

						// Convert "null" string to null
						if(value === "null") value = null;
						denseConfigSetter(nn.layerNeurons.length-1, "activation", value);

						// Rebuild the network
						buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.95, centerY],
			canvasRelativeSize: [0.06, 0.04]
		});
		centerY -= centerYstep;

		// Process every hidden layer for GUI component placement
		[...Array(nn.layerNeurons.length).keys()].slice(1, -1).reverse().forEach(denseLayerIndex => {
			let activationSelectId = ("nn_cfg_dense_layer_activation_"+(denseLayerIndex.toString()));
			let neuronCountInputId = ("nn_cfg_dense_layer_unitCount_"+(denseLayerIndex.toString()));
			let biasButtonId = ("nn_cfg_dense_layer_bias_"+(denseLayerIndex.toString()));

			//// Place layer config components below the hidden&output layers
			// Place removal button below the hidden layer
			addGUIComponent({
				...nnGUIComponentDefaults,
				id: ("nn_cfg_remove_hidden_layer_"+(denseLayerIndex.toString())),
				obj: createButton("-"),
				initCalls: [
					{fnName: "mousePressed", args: [
						(() => {
							// Remove hidden layer from the config
							nnStructure.hiddenLayersConfig.splice((denseLayerIndex-1), 1);
							// Rebuild the network
							buildNeuralNetwork();
						})
					]},
				],
				canvasRelativePosition: [0.80, centerY],
				canvasRelativeSize: [0.02, 0.03]
			});

			// Place neuron size input below the hidden layers
			addGUIComponent({
				...nnGUIComponentDefaults,
				id: neuronCountInputId,
				obj: createInput(denseConfigGetter(denseLayerIndex, "units").toString()),
				initCalls: [
					// changed event
					{fnName: "changed", args: [
						(event) => {
							// Get new value and set layer config
							let value = getGUIComponentWithID(neuronCountInputId).obj.value();

							// Try to convert the string to number and check the validity of the input
							if(typeof(value) !== "number") value = Number(value);
							if(typeof(value) !== "number") return;
							if(!Number.isInteger(value)) return;
							if(value <= 0) return;

							// Check new neuron unit count limit
							if(nnStructure.applyLimits && (value > nnStructure.limits.maxHiddenUnitCount)){
								value = nnStructure.limits.maxHiddenUnitCount;
							}

							denseConfigSetter(denseLayerIndex, "units", value);
							
							// Rebuild the network
							buildNeuralNetwork();
						}
					]},
				],
				canvasRelativePosition: [0.84, centerY],
				canvasRelativeSize: [0.04, 0.04]
			});

			// "Use bias" button
			addGUIComponent({
				...nnGUIComponentDefaults,
				id: biasButtonId,
				obj: createButton(`${denseConfigGetter(denseLayerIndex, "useBias") ? "true" : "false"}`),
				initCalls: [
					{fnName: "mousePressed", args: [
						(event) => {
							// Get new value and set layer config
							denseConfigSetter(denseLayerIndex, "useBias", !denseConfigGetter(denseLayerIndex, "useBias"));
							// Rebuild the network
							buildNeuralNetwork();
						}
					]},
				],
				canvasRelativePosition: [0.89, centerY],
				canvasRelativeSize: [0.04, 0.04]
			});

			// Activation function selector
			addGUIComponent({
				...nnGUIComponentDefaults,
				id: activationSelectId,
				obj: createSelect(),
				initCalls: [
					// Add all activations as option
					...(Object.entries(nnStructure.activationFunctionOptions).map(([key, value]) => ({fnName: "option", args: [key, value]}))),
					// Set the current one as selected
					...((denseConfigGetter(denseLayerIndex, "activation") === null) ? [] : [{fnName: "selected", args: [denseConfigGetter(denseLayerIndex, "activation")]}]),

					// changed event
					{fnName: "changed", args: [
						(event) => {
							// Get new value and set layer config
							let value = getGUIComponentWithID(activationSelectId).obj.value();

							// Convert "null" string to null
							if(value === "null") value = null;
							denseConfigSetter(denseLayerIndex, "activation", value);

							// Rebuild the network
							buildNeuralNetwork();
						}
					]},
				],
				canvasRelativePosition: [0.95, centerY],
				canvasRelativeSize: [0.06, 0.04]
			});

			centerY -= centerYstep;
		});

		// Neuron, bias, activation texts
		centerY += (centerYstep*0.33);
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title_units",
			obj: createButton(`Units`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.84, centerY],
			canvasRelativeSize: [0.04, 0.04]
		});
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title_usebias",
			obj: createButton(`Bias`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.89, centerY],
			canvasRelativeSize: [0.04, 0.04]
		});
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title_activation",
			obj: createButton(`Activation`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.95, centerY],
			canvasRelativeSize: [0.06, 0.04]
		});
		centerY -= (centerYstep);

		// Input layer text
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title_input",
			obj: createButton(`Input (${nnStructure.inputLayerConfig.args.inputShape[0]})`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.90, centerY],
			canvasRelativeSize: [0.16, 0.04]
		});
		centerY -= centerYstep;
		
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_title",
			obj: createButton("Network Structure"),
			initCalls: [
				{fnName: "class", args: ["text-button-border"]},
			],
			canvasRelativePosition: [0.90, centerY],
			canvasRelativeSize: [0.16, 0.04]
		});

		//// Left-bottom components
		// "Show bias" button
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_show_bias_neurons",
			obj: createButton(nnVArgs.showBiasNeurons ? "Don't show bias" : "Show bias"),
			initCalls: [
				{fnName: "addClass", args: ["button-bottom-border"]},
				// mousePressed event
				{fnName: "mousePressed", args: [
					() => {
						nnVArgs.showBiasNeurons = !nnVArgs.showBiasNeurons;
						// Rebuild the network
						buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.08, 0.96],
			canvasRelativeSize: [0.12, 0.04]
		});

		// "Animate" button
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_animate_propagation",
			obj: createButton(nnVArgs.animatePropagation ? "Disable animation" : "Enable animation"),
			initCalls: [
				{fnName: "addClass", args: ["button-bottom-border"]},
				// mousePressed event
				{fnName: "mousePressed", args: [
					() => {
						nnVArgs.animatePropagation = !nnVArgs.animatePropagation;
						// Rebuild the network
						buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.21, 0.96],
			canvasRelativeSize: [0.12, 0.04]
		});

		// "Apply limits" button
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_apply_limits",
			obj: createButton(nnStructure.applyLimits ? "Disable limits" : "Enable limits"),
			initCalls: [
				{fnName: "addClass", args: ["button-bottom-border"]},
				// mousePressed event
				{fnName: "mousePressed", args: [
					() => {
						nnStructure.applyLimits = !nnStructure.applyLimits;

						// Apply limits immediately
						if(nnStructure.applyLimits){
							// Hidden layer count max limit
							if(nnStructure.hiddenLayersConfig.length > nnStructure.limits.maxHiddenLayerCount){
								nnStructure.hiddenLayersConfig = nnStructure.hiddenLayersConfig.slice(0, nnStructure.limits.maxHiddenLayerCount);
							}
							// Hidden layer neuron unit max limit
							[...Array(nnStructure.hiddenLayersConfig.length).keys()].forEach((hiddenLayerIdx) => {
								nnStructure.hiddenLayersConfig[hiddenLayerIdx].args.units = Math.min(
									nnStructure.limits.maxHiddenUnitCount,
									nnStructure.hiddenLayersConfig[hiddenLayerIdx].args.units
								);
							});
						}

						// Rebuild the network
						buildNeuralNetwork();
					}
				]},
			],
			canvasRelativePosition: [0.34, 0.96],
			canvasRelativeSize: [0.12, 0.04]
		});
	}

	// Parameter count
	if(nn.isCompiled){
		addGUIComponent({
			...nnGUIComponentDefaults,
			id: "nn_cfg_parameter_count",
			obj: createButton(`Total parameter count: ${nn.getTotalParameterCount().toString()}`),
			initCalls: [{fnName: "addClass", args: ["text-button"]}],
			canvasRelativePosition: [0.10, 0.0625],
			canvasRelativeSize: [0.16, 0.04]
		});
	}
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
		this.precompile();
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
		this.summary();
	};

	//// Feed-forward / Backpropagation / Applying gradient functionalities
	// Feed-forward method
	feedForward = (X) => {
		if(this.vArgs.propagation.inProgress || (this.vArgs.animatePropagation && this.vArgs.propagation.inProgress)) return false;

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
		this.vArgs.propagation.inProgress = true;
		this.vArgs.status.text = "Prediction >>";

		// Set necessary values
		this.vArgs.predicted = true;
		this.vArgs.backpropagated = false;

		// Log final output
		console.log("Prediction", layerOutput.toString());
		return true;
	};

	// Backpropagate method
	backpropagate = (X, y) => {
		if(this.vArgs.propagation.inProgress || (this.vArgs.animatePropagation && this.vArgs.propagation.inProgress)) return false;

		// Calculate & assign gradient values to Weight objects
		[...Array(this.layerNeurons.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.layerNeurons[layerIndex];
			let toLayerNeurons = this.layerNeurons[layerIndex+1];

			// Check if next layer uses bias
			let nextLayerUsesBias = (this.layers[layerIndex+2] && (this.layers[layerIndex+2].useBias === true));
			// If next layer is using bias, there's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
				toLayerNeurons = toLayerNeurons.slice(1);
			};
			
			// Get the 2D gradient tensor of the layer, turn it into a nested-list
			let layerGradientMatrix = this.getLayerGradientMatrix(X, y, layerIndex+1, this.vArgs.showBiasNeurons).arraySync();

			fromLayerNeurons.forEach((fromNeuron, fromNeuronIndex) => {
				toLayerNeurons.forEach((toNeuron, toNeuronIndex) => {
					// Invert real value for descent value
					let newWeightGradientValue = -layerGradientMatrix[fromNeuronIndex][toNeuronIndex];
					// Update gradient of Weight object
					this.layerWeights[layerIndex][fromNeuronIndex][toNeuronIndex].gradientValue = newWeightGradientValue;
				});
			});
		});

		// Gradient values changed
		this.onChangeWeights(true);

		// Fire the neurons backward ;)
		this.vArgs.propagation.x = 1.0;
		this.vArgs.propagation.xAnim = 1.0;
		this.vArgs.propagation.xTarget = 0.0;
		this.vArgs.propagation.inProgress = true;
		this.vArgs.status.text = "<< Backpropagation";

		// Set necessary values
		this.vArgs.backpropagated = true;

		// Log
		console.log("Backpropagation");
		return true;
	};

	// Reset gradients method
	resetGradients = () => {
		// Reset gradient values of Weight objects
		[...Array(this.layerNeurons.length-1).keys()].forEach((layerIndex) => {
			let fromLayerNeurons = this.layerNeurons[layerIndex];
			let toLayerNeurons = this.layerNeurons[layerIndex+1];

			// Check if next layer uses bias
			let nextLayerUsesBias = (this.layers[layerIndex+2] && (this.layers[layerIndex+2].useBias === true));
			// If next layer is using bias, there's no weight connected to the next layer's first neuron from this layer, ignore it!!
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
				toLayerNeurons = toLayerNeurons.slice(1);
			};
			
			fromLayerNeurons.forEach((fromNeuron, fromNeuronIndex) => {
				toLayerNeurons.forEach((toNeuron, toNeuronIndex) => {
					// Reset gradient of Weight object
					this.layerWeights[layerIndex][fromNeuronIndex][toNeuronIndex].gradientValue = null;
					this.layerWeights[layerIndex][fromNeuronIndex][toNeuronIndex].gradientVisualValue = null;
				});
			});
		});
	};

	// Apply gradients method
	applyGradients = (X, y) => {
		if(this.vArgs.propagation.inProgress || (this.vArgs.animatePropagation && this.vArgs.propagation.inProgress)) return false;
		
		// Calculate gradients for each weight
		let {value, grads} = this.optimizer.computeGradients(
			() => this.loss(
				this.predict(X),
				y
			)
		);

		// Apply gradients to weights
		this.optimizer.applyGradients(grads);

		// Update all Weight objects' values
		this.updateAllWeights();
		// Weight values changed, call it!
		this.onChangeWeights(true);

		// Set necessary values
		this.vArgs.predicted = false;
		this.vArgs.backpropagated = false;

		// Log
		console.log("Applied gradients");
		return true;
	};

	// Override fit method
	fit = async (X, y, args) => {
		if(this.vArgs.propagation.inProgress || (this.vArgs.animatePropagation && this.vArgs.propagation.inProgress)) return false;
		this.resetGradients();

		let {history} = await super.fit(X, y, args);
		
		// Update all Weight objects' values
		this.updateAllWeights();
		// Weight values changed, call it!
		this.onChangeWeights(true);

		// Set necessary values
		this.vArgs.predicted = false;
		this.vArgs.backpropagated = false;

		// Fire the neurons backward ;)
		this.vArgs.propagation.x = 1.0;
		this.vArgs.propagation.xAnim = 1.0;
		this.vArgs.propagation.xTarget = 0.0;
		this.vArgs.propagation.inProgress = true;
		this.vArgs.status.text = `${args.epochs} Epoch, Cost: ${history.loss[history.loss.length-1].toFixed(4)}`;

		// Log loss output
		console.log("Loss", history.loss[history.loss.length-1]);
		return true;
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
				this.layerNeurons[layerIndex].push(new Neuron(0, 0, layerIndex===0));
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
				layerWeightMatrix = this.getLayerWeightMatrix(layerIndex+1, this.vArgs.showBiasNeurons).arraySync();
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
			xTarget: 0.0,
			inProgress: false
		};
		this.layerNeurons = [];
		this.layerWeights = [];
	};

	// Gets weight matrix of a dense layer
	getLayerWeightMatrix = (layerIdx, withBias) => {
		// Get all weights of the layer (list)
		let w = this.layers[layerIdx].getWeights();

		// Get kernel
		let kernel = w[0];
		// Directly return the kernel if there's no bias or bias not wanted
		if((w.length === 1) || (!withBias)){
			return kernel;
		}

		// Expanding bias vector for concatenation. 1D to 2D
		let bias = tf.expandDims(w[1], 0);

		// Concat bias&kernel values for getting the final layer weight matrix
		return tf.concat([bias, kernel], 0);
	};

	// Gets gradient matrix of a dense layer, with given data
	getLayerGradientMatrix = (X, y, layerIdx, withBias) => {
		// Calculate gradient with given input/target values
		let {value, grads} = this.optimizer.computeGradients(
			() => this.loss(
				this.predict(X),
				y
			),
			this.layers[layerIdx].getWeights()
		);
		// Get all gradient values of the layer (list)
		let w = Object.values(grads);

		// Get kernel
		let kernel = w[0];
		// Directly return the kernel if there's no bias or bias not wanted
		if((w.length === 1) || (!withBias)){
			return kernel;
		}

		// Expanding bias vector for concatenation. 1D to 2D
		let bias = tf.expandDims(w[1], 0);

		// Concat bias&kernel values for getting the final layer gradient matrix
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
			let layerWeightMatrix = this.getLayerWeightMatrix(layerIndex+1, this.vArgs.showBiasNeurons).arraySync();

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
		//// Get all weights in a 1D tensor
		let allWeights = (withRealValues) ? (this.getAllWeights(this.vArgs.showBiasNeurons).arraySync()) : [1, 1];
		// Update weight stats
		this.vArgs.weightsStats = {
			min: Math.min(...allWeights),
			max: Math.max(...allWeights),
			mean: (arrSum(allWeights) / allWeights.length)
		};
		this.vArgs.weightsStats.wMax = Math.max(
			Math.abs(this.vArgs.weightsStats.min),
			Math.abs(this.vArgs.weightsStats.max)
		);

		//// Get all gradients in an array
		let allGradients = [];
		this.layerWeights.forEach((_, layerIndex) => {
			this.layerWeights[layerIndex].forEach((_, fromNeuronIndex) => {
				this.layerWeights[layerIndex][fromNeuronIndex].forEach((_, toNeuronIndex) => {
					allGradients.push(this.layerWeights[layerIndex][fromNeuronIndex][toNeuronIndex].gradientValue);
				});
			});
		});

		// Update gradient stats
		this.vArgs.gradientStats = {
			min: Math.min(...allGradients),
			max: Math.max(...allGradients),
			mean: (arrSum(allGradients) / allGradients.length)
		};
		this.vArgs.gradientStats.gMax = Math.max(
			Math.abs(this.vArgs.gradientStats.min),
			Math.abs(this.vArgs.gradientStats.max)
		);
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
		this.vArgs.neuronStats.nMax = Math.max(
			Math.abs(this.vArgs.neuronStats.min),
			Math.abs(this.vArgs.neuronStats.max)
		);
	};

	// Gets all weights in a 1D tensor
	getAllWeights = (withBias) => {
		// Concat all 1D tensors, return it
		return tf.concat(
			// Convert each layer's weights to 1D tensor, put them in a list
			// Slice 1 for excluding the input layer (has no kernel&bias)
			[...Array(this.layers.length).keys()].slice(1).map((layerIndex) => {
				// Get the 2D weight tensor of the layer
				let layerWeightMatrix = this.getLayerWeightMatrix(
					layerIndex,
					withBias
				);
				// Flatten the tensor (2D to 1D) and return it
				return layerWeightMatrix.reshape([layerWeightMatrix.size]);
			}),
			0
		);
	};

	// Gets total parameter count of the network
	getTotalParameterCount = () => {
		return this.getAllWeights(true).size;
	};

	// Updates the network, gets called at each main loop of the sketch
	update = (canvas) => {
		this.vArgs.nnIsCompiled = this.isCompiled;

		//// Calculate values for drawing
		// Get maximum neuron count
		this.vArgs.maxUnitCount = Math.max(...(this.layers.map(layer => 
			(layer.units) || (layer.batchInputShape && layer.batchInputShape[1])
		)));
		// Calculate step per neuron in +Y direction
		this.vArgs.perNeuronY = ((canvas.height * this.vArgs.scaleY) / (
			// Limit maximum neuron size when there's few neurons
			Math.max(7, this.vArgs.maxUnitCount)
		));
		// Calculate step per layer in +X direction
		this.vArgs.perLayerX = ((canvas.width * this.vArgs.scaleX) / (this.layers.length-1));
		// Calculate X coordinate of starting point of the network
		this.vArgs.startLayerX = (canvas.width * ((1-this.vArgs.scaleX) / 2)) + (canvas.width * this.vArgs.translateX);

		//// Adjust "per layer X" and "per neuron Y" step values for expanding the network out of to the canvas
		// Layer step value on the X axis cannot be smaller than %25 of the canvas width
		this.vArgs.perLayerX = Math.max(this.vArgs.perLayerX, (canvas.width * 0.25));
		// Neuron step value on the Y axis cannot be smaller than %5 of the canvas height
		this.vArgs.perNeuronY = Math.max(this.vArgs.perNeuronY, (canvas.height * 0.05));

		// Calculate each neuron size with using step per neuron value
		Neuron.r = (this.vArgs.perNeuronY / 1.25 / 2);

		this.vArgs.layersTopRowY = ((canvas.height/2) - ((this.vArgs.perNeuronY * (this.vArgs.maxUnitCount+1)) / 2)) + (canvas.height * this.vArgs.translateY);
		this.vArgs.layersBottomRowY = this.vArgs.layersTopRowY + (this.vArgs.perNeuronY * (this.vArgs.maxUnitCount+1));
		this.vArgs.canvasWidth = canvas.width;
		this.vArgs.canvasHeight = canvas.height;

		//// Update propagation related values
		// Update the function for calculating the real X position of the wave with canvas width/gap value
		this.vArgs.propagation.xToCanvasPosX = ((curX) => (
			(this.vArgs.startLayerX - Neuron.r) + (curX * ((this.vArgs.perLayerX * (this.layers.length-1)) + Neuron.r*2))
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

		//// Update inProgress value of propagation
		let nextInProgress = (this.vArgs.propagation.x != this.vArgs.propagation.xTarget);
		let prevInProgress = this.vArgs.propagation.inProgress;

		// Propagation done event
		if(!nextInProgress && prevInProgress){
			// Reset status text after some time
			setTimeout(
				() => {
					this.vArgs.status.text = this.vArgs.status.defaultText;
				},
				250
			)
		}
		// Propagation started event
		if(nextInProgress && !prevInProgress){}

		this.vArgs.propagation.inProgress = nextInProgress;

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

		//// Call update methods of Neuron&Weight objects
		// Each layer
		this.layerNeurons.forEach((layer, layerIndex) => {
			// Calculate starting point (Y-coordinate of first neuron) of the layer
			// Top of the layer in Y = ((Center of the neural network in Y) - (layer size in Y / 2)) + (applying Y shift a bit for centering) + translation of the network
			let startNeuronY = ((canvas.height/2) - ((this.vArgs.perNeuronY * layer.length) / 2)) + (this.vArgs.perNeuronY/2) + (canvas.height * this.vArgs.translateY);

			// Each neuron
			layer.forEach((neuron, neuronIndex) => {
				// Calculate new position of neuron & update it
				let newXpos = (this.vArgs.startLayerX + (this.vArgs.perLayerX * layerIndex));
				let newYpos = (startNeuronY + (this.vArgs.perNeuronY * neuronIndex));
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
	mouseClicked = (mouseX, mouseY) => {
		// Check if clicked on any of the neurons
		let clickedOnAnyNeuron = arrBoolAny(
			this.layerNeurons.map(
				layer => arrBoolAny(
					layer.map(
						neuron => (
							dist(
								neuron.x, neuron.y,
								mouseX, mouseY
							) < Neuron.r
						)
					)
				)
			)
		);

		// Process neuron clicks if user clicked on any of them
		if(clickedOnAnyNeuron){
			this.layerNeurons.forEach(
				layer => layer.forEach(
					neuron => {
						// Check neuron center position to mouse position distance
						let distanceToClick = dist(
							neuron.x, neuron.y,
							mouseX, mouseY
						);
						// If it's already in focus, defocus it
						neuron.isFocused = ((!neuron.isFocused) && (distanceToClick < Neuron.r));
					}
				)
			);
		}
	};

	// Draws the network's baseline (basically weights), gets called time-to-time for optimization purposes
	initDraw = (canvas) => {
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
	}

	// Draws the whole network on the given canvas, gets called if subcanvas is active
	draw = (canvas, sample) => {
		//// Draw neurons
		// Each layer
		this.layerNeurons.forEach(layer => {
			// Each neuron
			layer.forEach((neuron) => {
				neuron.draw(canvas, this.vArgs);
			});
		});

		//// Draw sample input/targets to the side of the network
		if(this.isCompiled && (!this.vArgs.isDummy) && (sample && sample.input && sample.target)){
			// Draw the status text
			canvas.push();
			canvas.fill(255);
			canvas.translate(
				((this.vArgs.propagation.xToCanvasPosX(0) + this.vArgs.propagation.xToCanvasPosX(1)) / 2),
				(this.vArgs.layersTopRowY - (this.vArgs.perNeuronY))
			);
			canvas.textSize(Neuron.r);
			canvas.text(
				this.vArgs.status.text,
				0, 0
			);
			canvas.pop();

			// Get the sample (currently tensor) in a nested-list
			let inputVec = sample.input.arraySync()[0];
			let targetVec = sample.target.arraySync()[0];

			// Check if next layer uses bias
			let nextLayerUsesBias = (this.layers[1] && (this.layers[1].useBias === true));
			// If next layer is using bias, add value 1 to the top of the input
			if(nextLayerUsesBias && this.vArgs.showBiasNeurons){
				inputVec = [1.0, ...inputVec];
			};

			//// Input
			// Set input neurons' output values to input values (gets animated smoothly)
			let neuronInputChanged = false;
			this.layerNeurons[0].forEach((inputNeuron, neuronIndex) => {
				if(inputNeuron.value !== inputVec[neuronIndex]){
					inputNeuron.value = inputVec[neuronIndex];
					neuronInputChanged = true;
				}
			});
			// If any of the neuron input value is changed, call this thing for updating color
			if(neuronInputChanged) this.onChangeNeurons();

			//// Target
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

			//// Draw layer titles
			canvas.push();
			canvas.fill(255);
			canvas.textSize(calculateTextSize(" ", Neuron.r*2, Neuron.r*2));

			// "x" title of input layer
			canvas.text(
				"x",
				(this.layerNeurons[0][0].x),
				(this.layerNeurons[0][0].y - (this.vArgs.perNeuronY*1.25))
			);

			// "ŷ" title of output layer
			canvas.text(
				"ŷ",
				(this.layerNeurons[this.layerNeurons.length-1][0].x),
				(this.layerNeurons[this.layerNeurons.length-1][0].y - (this.vArgs.perNeuronY*1.25))
			);

			// "y" title at target value
			canvas.text(
				"y",
				(this.layerNeurons[this.layerNeurons.length-1][0].x) + (Neuron.r*3),
				(this.layerNeurons[this.layerNeurons.length-1][0].y - (this.vArgs.perNeuronY*1.25))
			);
			canvas.pop();
		}
	};
};

// Neuron object represents every output unit of the fully-connected layer, holds output value and XY position on the subcanvas.
class Neuron{
	value = null;
	visualValue = null;
	isFocused = false;
	isInput = false;

	x = 0;
	y = 0;
	static r = 30.0;

	constructor(x, y, isInput){
		this.x = x;
		this.y = y;
		this.isInput = isInput;
	}

	// Sets visual value
	updateVisualValue = (vArgs) => {
		// Set to zero initially
		if(this.visualValue === null) this.visualValue = 0.0;
		
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
					&& ((vArgs.propagation.xTarget - vArgs.propagation.xAnim) >= 0)
				) || (!vArgs.animatePropagation) || (this.isInput)
			){
				this.updateVisualValue(vArgs);
			}
		}
	};

	draw = (canvas, vArgs) => {
		//// Calculate color value
		let colorValue;

		// // Real output value
		if(vArgs.nnIsCompiled && (this.visualValue !== null)){
			let ratio = Math.min((((Math.abs(this.visualValue) / vArgs.neuronStats.nMax) * 0.67) + 0.33), 1.0);
			colorValue = ratio * 255;
		}
		// Dummy output
		else{
			colorValue = 255;
		}

		// Neuron circle
		canvas.push();
		canvas.fill(BG_COLOR);
		canvas.stroke(colorValue);
		canvas.strokeWeight((this.isFocused && vArgs.focusedAnyNeuron) ? 2 : 1);
		canvas.circle(this.x, this.y, Neuron.r*2);
		canvas.pop();

		// Draw the hidden/output neurons' output (activation value) as text
		if(vArgs.nnIsCompiled && (!vArgs.isDummy) && (this.visualValue !== null)){
			canvas.push();
			canvas.fill(colorValue);
			canvas.stroke(colorValue);

			let vText = this.visualValue.toFixed(2);
			canvas.textSize(calculateTextSize(vText, Neuron.r*3, Neuron.r*2));
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
	gradientValue = null;
	gradientVisualValue = null;

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

		// Update visual gradient value
		this.gradientVisualValue = this.gradientValue;
	};

	// Gets called every frame
	update = (vArgs) => {
		// Adjust the visual value after propagation (backward) wave passes over it
		if(
			(
				(vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim) <= this.from.x)
				&& ((vArgs.propagation.xTarget - vArgs.propagation.xAnim) <= 0)
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

		//// Calculate color value
		let colorValue;
		
		// Real weight value
		if(vArgs.nnIsCompiled){
			let ratio = Math.min((((Math.abs(this.visualValue) / vArgs.weightsStats.wMax) * 0.95) + 0.05), 1.0);
			colorValue = ratio * 255;
		}
		// Dummy weight
		else colorValue = 255;

		// Draw weight between neurons (from -> to) as a line
		if(!writeCarriedValue){
			// Draw gradient line (if gradient value exists)
			if(this.gradientVisualValue !== null && Math.abs(this.gradientVisualValue).toFixed(2) !== "0.00"){
				canvas.push();

				// Red/Green interpolation
				let gradientColorValue = (Math.abs(this.gradientVisualValue) / vArgs.gradientStats.gMax) * 255;
				if(this.gradientVisualValue > 0) canvas.stroke(0, gradientColorValue, 0, 192);
				else if(this.gradientVisualValue < 0) canvas.stroke(gradientColorValue, 0, 0, 192);

				canvas.strokeWeight(3);
				canvas.line(fromX, fromY, toX, toY);
				canvas.pop();
			}
			// Main weight line
			else{
				canvas.push();
				canvas.stroke(colorValue);
				canvas.strokeWeight(1);
				canvas.line(fromX, fromY, toX, toY);
				canvas.pop();
			}
		}
		// Draw the carried value as text between the connection if focused
		else{
			// Text value gap center point and width
			let gapCenter = 0.50;
			// Adjust gapcenter value for drawing text values in a "circular" path
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
			canvas.stroke(colorValue);
			canvas.line(fromX, fromY, gapStartX, gapStartY);
			canvas.line(gapEndX, gapEndY, toX, toY);
			canvas.pop();

			//// Draw the value as text on the center of the gap with rotating accordingly
			canvas.push();
			canvas.translate(
				lerp(fromX, toX, gapCenter),
				lerp(fromY, toY, gapCenter)
			);
			canvas.rotate(
				createVector(toX-fromX, 0).normalize().angleBetween(
					createVector(toX-fromX, toY-fromY).normalize()
				)
			);

			// Calculate text size
			let textSize = calculateTextSize(
				"    ",
				// Get width&height as distance between gap endpoints
				(gapEndX-gapStartX),
				(gapEndY-gapStartY)
			);

			/// Weight value
			canvas.push();
			canvas.fill(255);
			canvas.stroke(255);
			canvas.textSize(textSize);
			let vText = this.visualValue.toFixed(2);
			canvas.text(vText, 0, 0);
			canvas.pop();

			/// Gradient value
			if(this.gradientVisualValue !== null){
				// Draw below real value, with smaller text size
				canvas.push();
				canvas.translate(0, textSize);
				canvas.textSize(textSize*0.66);

				let gvText = this.gradientVisualValue.toFixed(2);
				if(Math.abs(this.gradientVisualValue).toFixed(2) === "0.00"){
					canvas.fill(255);
					canvas.stroke(255);
					// WTF is -0.00?
					gvText = "0.00";
				}
				// Red/Green
				else if(this.gradientVisualValue > 0){
					canvas.fill(0, 192, 0);
					canvas.stroke(0, 192, 0);
				}
				else if(this.gradientVisualValue < 0){
					canvas.fill(192, 0, 0);
					canvas.stroke(192, 0, 0);
				}
				canvas.text(gvText, 0, 0);
				canvas.pop();
			}

			canvas.pop();
		}

		//// Highlight the connection during propagation
		// If highlighting point reached to the destination, no need to draw anything anymore
		if(abs(vArgs.propagation.xTarget - vArgs.propagation.x) < 0.001) return;

		// Calculate highlight area
		let hFromX = vArgs.propagation.xToCanvasPosX(vArgs.propagation.xAnim);
		let hToX = vArgs.propagation.xToCanvasPosX(Math.min(1.0, (vArgs.propagation.xAnim + vArgs.propagation.width)));
		hFromX = Math.max(hFromX, fromX);
		hToX = Math.min(hToX, toX);

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
		canvas.stroke(colorValue);
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
