// Main page related things

//// Dummy neural network
var dummynn;
initializeDummyNeuralNetwork = () => {
	// Various (configurable) visual arguments
	let dummynnVArgs = {
		scaleX: 0.33, scaleY: 0.33,
		translateX: -0.090, translateY: 0.20,
		showBiasNeurons: false,
		weightVisualChangeSpeed: 0.25,
		neuronVisualChangeSpeed: 0.25,
		animatePropagation: true,
		propagation: {
			// Width and step values (ratio value for width of the canvas) of the propagation wave
			width: 0.02, step: 0.01,
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
		isDummy: true,
	};

	// Specifies our dummy neural network structure (layers, losses etc.)
	let dummynnStructure = {
		// Input layer config
		inputLayerConfig: {
			class: tf.layers.inputLayer,
			// (set input neuron count randomly)
			args: {inputShape: [getRandomInt(3, 4)]}
		},

		hiddenLayersConfig: [
			createDenseLayerConfig({activation: "linear", units: getRandomInt(6, 9)}),
		],

		// Output layer config
		outputLayerConfig: createDenseLayerConfig({
			// (set output neuron count randomly)
			units: getRandomInt(1, 3),
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
		vArgs=dummynnVArgs
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

	// Set it as it's training visually
	// Feed forwards with given matrix
	let dummynn_feedForward = (X) => {
		return dummynn.feedForward(X);
	};

	// Backpropagates with given input&target matrix
	let dummynn_backpropagate = (X, y) => {
		return dummynn.backpropagate(X, y);
	};

	// Applies gradients to weights, resets gradient values of Weight objects
	let dummynn_applyGradient = (X, y) => {
		if(dummynn.applyGradients(X, y)){
			dummynn.resetGradients();
			return true;
		}
		return false;
	};

	// Fake input&target for dummy nn
	let dummynn_generate_sample = () => {
		return [
			tf.randomNormal(
				[1, dummynnStructure.inputLayerConfig.args.inputShape[0]],
				mean=0.0, stddev=1.0
			),
			tf.randomNormal(
				[1, dummynnStructure.outputLayerConfig.args.units],
				mean=0.0, stddev=50.0
			)
		];
	};

	// Define as async and run fake training loop immediately
	(async () => {
		do{
			// Generate fake sample for dummy nn
			let [dummynn_input, dummynn_target] = dummynn_generate_sample();
			console.log(dummynn_input.toString());
			console.log(dummynn_target.toString());

			// Start fake training
			dummynn.vArgs.autoTrain.inProgress = true;
			await sleep(getRandomInt(1500, 6000));

			// Feed forward
			while(!shouldSubCanvasBeDrawn(HOME_SUBCANVAS_INDEX) || !dummynn_feedForward(dummynn_input)){
				await sleep(getRandomInt(1500, 6000));
			}
			await sleep(getRandomInt(1500, 6000));

			// Backpropagate
			while(!shouldSubCanvasBeDrawn(HOME_SUBCANVAS_INDEX) || !dummynn_backpropagate(dummynn_input, dummynn_target)){
				await sleep(getRandomInt(1500, 6000));
			}
			await sleep(getRandomInt(1500, 6000));

			// Apply gradient
			while(!shouldSubCanvasBeDrawn(HOME_SUBCANVAS_INDEX) || !dummynn_applyGradient(dummynn_input, dummynn_target)){
				await sleep(getRandomInt(1500, 6000));
			}

			// Done.
			dummynn.vArgs.autoTrain.inProgress = false;
			await sleep(1000);
		}while(true);
	})();
};
