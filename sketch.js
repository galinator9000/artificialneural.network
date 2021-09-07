//// Data
let n_samples = 16;
let n_features = 16;
let n_targets = 2;

// let dataFrame = {};
// initData = () => {
// 	dataFrame.X = tf.randomNormal([n_samples, n_features]);
// 	dataFrame.y = tf.randomNormal([n_samples, n_targets]);
// };

//// Neural network
// Our main neural network model
let nn;

// Creates a dense layer config object with given values
let createDenseLayerConfig = (units=null, useBias=null, activation=null) => (
	{
		class: tf.layers.dense,
		args: {
			// Min 4, max 15 neurons, if not specified
			units: ((units===null) ? getRandomInt(4, 16) : units),
			// Randomly choose if use bias or not, if not specified
			useBias: ((useBias===null) ? (getRandomInt(0, 2)===1) : useBias),
			// No activation, if not specified
			activation: activation
		}
	}
);

// Specifies our neural network structure (layers, losses etc.)
let nnStructure = {
	// Input layer
	inputLayer: {
		class: tf.layers.inputLayer,
		args: {inputShape: [n_features]}
	},

	// Hidden layers (create them randomly at start: min 1, max 5 layers)
	hiddenLayers: [...Array(getRandomInt(1, 6)).keys()].map(layer => (
		createDenseLayerConfig()
	)),

	// Output layer
	outputLayer: createDenseLayerConfig(
		// units
		n_targets,

		// useBias
		true,

		// activation
		// Regression
		null,
		// Classification
		// activation: "sigmoid",
	),

	// Compile arguments (optimizer, loss)
	compileArgs: {
		optimizer: "sgd",

		// Regression
		loss: "meanSquaredError",
		// Classification
		// loss: "categoricalCrossentropy"
	},
};

// Canvas which our neural network will be drawn
let nnCanvas;
let nnCanvasRatio = {x: 1.0, y: 1.0};
let nnCreateCanvas = () => {
	return createGraphics(
		(windowWidth*nnCanvasRatio.x),
		(windowHeight*nnCanvasRatio.y)
	);
};

// Builds neural network object at tf.js side with structure config, returns it
buildNeuralNetwork = () => {
	// Build NN sequentially with our custom class
	__nn__ = new SequentialNeuralNetwork(
		// Arguments which will be passed to tf.Sequential
		sequentialArgs={},
	);

	// Put all layer configs in a list, add each of them to the model
	[
		nnStructure.inputLayer,
		...nnStructure.hiddenLayers,
		nnStructure.outputLayer
	].forEach(layer => {
		__nn__.add(
			layer.class(layer.args)
		);
	});

	// Compile the model with args
	__nn__.compile(nnStructure.compileArgs);

	return __nn__;
};

// Initialize GUI components
initializeGUI = () => {
	// Predict
	let predictButton;
	predictButton = createButton("predict");
	predictButton.position(0, 0);
	predictButton.mousePressed(args => {
		nn.predict(
			tf.randomNormal([1, n_features])
		);
	});

	// Add hidden layer button
	let addHiddenLayerButton;
	addHiddenLayerButton = createButton("add hidden layer");
	addHiddenLayerButton.position(150, 0);
	addHiddenLayerButton.mousePressed(args => {
		// Add one layer to config & rebuild neural network
		nnStructure.hiddenLayers.push(createDenseLayerConfig());
		nn = buildNeuralNetwork();
	});

	// Remove hidden layers button
	let removeHiddenLayersButton;
	removeHiddenLayersButton = createButton("remove hidden layers");
	removeHiddenLayersButton.position(300, 0);
	removeHiddenLayersButton.mousePressed(args => {
		// Reset configs & rebuild neural network
		nnStructure.hiddenLayers = [];
		nn = buildNeuralNetwork();
	});

	// Reset neural network button
	let resetnnButton;
	resetnnButton = createButton("reset nn");
	resetnnButton.position(450, 0);
	resetnnButton.mousePressed(args => {
		// Reset configs & rebuild neural network
		nnStructure.hiddenLayers = [...Array(getRandomInt(1, 6)).keys()].map(layer => (createDenseLayerConfig()));
		nn = buildNeuralNetwork();
	});
};

// Setup
setup = () => {
	// Create main canvas
	createCanvas(windowWidth, windowHeight);

	// Create the canvas which will neural network be drawn
	nnCanvas = nnCreateCanvas();

	// Initialize GUI components
	initializeGUI();

	// Build neural network
	nn = buildNeuralNetwork();
};

// Main loop
draw = () => {
	// Clear backgrounds
	background(1, 0, 2, 255);
	nnCanvas.background(1, 0, 2, 255);

	// Draw the whole network on the given canvas
	nn.draw(
		nnCanvas,
		{gapRateX: 0.8, gapRateY: 0.8}
	);

	// Draw the network canvas to the main canvas 
	image(
		nnCanvas,
		// Position
		(windowWidth*(1-nnCanvasRatio.x)/2), (windowHeight*(1-nnCanvasRatio.y)),
		// Size
		(windowWidth*nnCanvasRatio.x), (windowHeight*nnCanvasRatio.y),
	);
};

// User-Events
// Resizes canvas' size when window is resized
windowResized = () => {
	resizeCanvas(windowWidth, windowHeight);
	nnCanvas = nnCreateCanvas();
};
