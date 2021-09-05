let n_samples = 16;
let n_features = 16;
let n_targets = 2;

// Data related
let dataFrame = {};
initData = () => {
	dataFrame.X = tf.randomNormal([n_samples, n_features]);
	dataFrame.y = tf.randomNormal([n_samples, n_targets]);
};

// Neural network related
let nn;
let nnHiddenLayers = [
	{"units": 8, "activation": "sigmoid"},
	{"units": 4, "activation": "sigmoid"},
];
initNN = () => {
	// Visualized NN class
	nn = new NeuralNetwork(
		// Layers
		[
			n_features,
			...nnHiddenLayers.map(hiddenLayer => (hiddenLayer.units)),
			n_targets
		],

		// Position & size
		windowWidth/2, windowHeight/2, (windowWidth*0.50), (windowHeight*0.66)
		// windowWidth/2, windowHeight/2, (windowWidth), (windowHeight)	// Fullscreen
	);

	// Build nn sequentially
	// nn = tf.sequential();

	// Input layer
	// nn.add(tf.layers.inputLayer({inputShape: [n_features]}));

	// Hidden layers
	// nnHiddenLayers.forEach(hiddenLayer => {
	// 	nn.add(tf.layers.dense({...hiddenLayer}));
	// });

	// Add output layer, compile model
	// Regression
	// nn.add(tf.layers.dense({units: n_targets, activation: null}));
	// nn.compile({loss: "meanSquaredError", optimizer: "sgd"});

	// Classification
	// nn.add(tf.layers.dense({units: n_targets, activation: "sigmoid"}));
	// nn.compile({loss: "categoricalCrossentropy", optimizer: "sgd"});

	// Print summary
	// nn.summary();
};

// Setup
setup = () => {
	// Create p5 canvas
	createCanvas(windowWidth, windowHeight);

	// Initialize neural network
	initNN();

	// Initialize dataset
	// initData();
};

// Main loop
draw = () => {
	// Background
	background(1, 0, 2, 255);

	// Draw neural network
	nn.draw();
};

// User-Events
// Resizes canvas' size when window is resized
windowResized = () => {
	resizeCanvas(windowWidth, windowHeight);
}
