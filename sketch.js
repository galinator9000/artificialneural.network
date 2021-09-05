let n_samples = 16;
let n_features = 16;
let n_targets = 2;

// Data related
// let dataFrame = {};
// initData = () => {
// 	dataFrame.X = tf.randomNormal([n_samples, n_features]);
// 	dataFrame.y = tf.randomNormal([n_samples, n_targets]);
// };

// Neural network related
let nn;
let nnHiddenLayers = [
	{"units": 8, "activation": "sigmoid", "useBias": false},
	{"units": 4, "activation": "sigmoid", "useBias": false},
];
initNN = () => {
	// Build NN sequentially with our custom class
	nn = new SequentialNeuralNetwork(
		// Arguments which will be passed to tf.Sequential
		sequentialArgs={},

		// Our args for visualizing
		customArgs={
			// Center & size of NN
			centerX: windowWidth*0.50,
			centerY: windowHeight*0.50,
			width: (windowWidth*0.50),
			height: (windowHeight*0.66)
		}
	);

	// Input layer
	nn.add(tf.layers.inputLayer({inputShape: [n_features]}));

	// Add hidden layers
	nnHiddenLayers.forEach(hiddenLayer => {
		nn.add(tf.layers.dense({...hiddenLayer}));
	});

	// Add output layer & compile the model
	// Regression
	// nn.add(tf.layers.dense({units: n_targets, activation: null}));
	// nn.compile({loss: "meanSquaredError", optimizer: "sgd"});

	// Classification
	nn.add(tf.layers.dense({units: n_targets, activation: "sigmoid"}));
	nn.compile({loss: "categoricalCrossentropy", optimizer: "sgd"});
};

// Setup
setup = () => {
	// Create p5 canvas
	createCanvas(windowWidth, windowHeight);

	// Initialize neural network
	initNN();
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
