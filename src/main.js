// Main page related things

//// Dummy neural network
var dummynn;
initializeDummyNeuralNetwork = () => {
	// Various (configurable) visual arguments
	let dummynnVArgs = {
		scaleX: 0.66, scaleY: 0.33,
		translateX: -0.045, translateY: 0.15,
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
	};

	// Specifies our dummy neural network structure (layers, losses etc.)
	let dummynnStructure = {
		// Input layer config
		inputLayerConfig: {
			class: tf.layers.inputLayer,
			// (set input neuron count randomly)
			args: {inputShape: [getRandomInt(4, 6)]}
		},

		// Hidden layers config, add two layer initially
		hiddenLayersConfig: [
			createDenseLayerConfig({activation: "linear", units: getRandomInt(6, 9)}),
			createDenseLayerConfig({activation: "linear", units: getRandomInt(6, 9)})
		],

		// Output layer config
		outputLayerConfig: createDenseLayerConfig({
			// (set output neuron count randomly)
			units: getRandomInt(1, 4),
			useBias: true,
			activation: "linear"
		}),

		// Compile arguments (optimizer, loss)
		compileArgs: {
			optimizer: "sgd",
			loss: "meanSquaredError",
			learningRate: 0.0001
		},

		// Whether apply limits to some configurations of the network or not
		applyLimits: true,
		limits: {
			maxHiddenUnitCount: 16,
			maxHiddenLayerCount: 3
		}
	};

	// Build dummy neural network
	dummynn = new SequentialNeuralNetwork(
		sequentialArgs={},
		vArgs=dummynnVArgs,
		isDummy=true
	);

	// Put all layer configs in a list, add each of them to the fake model
	[
		dummynnStructure.inputLayerConfig,
		...dummynnStructure.hiddenLayersConfig,
		dummynnStructure.outputLayerConfig
	].forEach(layerConfig => dummynn.add(layerConfigToLayer(layerConfig)));

	// "Precompile" the dummy network for being able to draw it
	dummynn.precompile();

	// Update dummy network once initially
	dummynn.update(subCanvas.c[NN_SUBCANVAS_INDEX].obj);

	// Compile dummy neural network
	dummynn.compile({
		optimizer: tf.train[nnStructure.compileArgs.optimizer](nnStructure.compileArgs.learningRate),
		loss: tf.losses[nnStructure.compileArgs.loss]
	});
};

// Dummy nn class that inherits main one
// class DummySequentialNeuralNetwork extends SequentialNeuralNetwork{}