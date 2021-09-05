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
let nnHiddenLayersStructure = [
	// Hidden layers
	{class: tf.layers.dense, args: {"units": 12, "activation": "sigmoid"}},
	{class: tf.layers.dense, args: {"units": 8, "activation": "sigmoid"}},
	{class: tf.layers.dense, args: {"units": 8, "activation": "sigmoid"}},
	{class: tf.layers.dense, args: {"units": 8, "activation": "sigmoid"}},
	{class: tf.layers.dense, args: {"units": 4, "activation": "sigmoid"}},
	{class: tf.layers.dense, args: {"units": 4, "activation": "sigmoid"}},
];
let nnStructure = {
	layers: [
		// Input layer
		{class: tf.layers.inputLayer, args: {inputShape: [n_features]}},

		// Hidden layer(s)
		...nnHiddenLayersStructure,

		// Output layer
		{
			class: tf.layers.dense,
			args: {
				"units": n_targets,

				// Regression
				"activation": null,
				// Classification
				// "activation": "sigmoid"
			}
		},
	],
	compileArgs: {
		optimizer: "sgd",

		// Regression
		loss: "meanSquaredError",
		// Classification
		// loss: "categoricalCrossentropy"
	},
};
buildNeuralNetwork = () => {
	// Build NN sequentially with our custom class
	__nn__ = new SequentialNeuralNetwork(
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

	// Add layers
	nnStructure.layers.forEach(layer => {
		__nn__.add(
			layer.class(layer.args)
		);
	});

	// Compile the model
	__nn__.compile(nnStructure.compileArgs);

	return __nn__;
};

// Setup
setup = () => {
	// Create p5 canvas
	createCanvas(windowWidth, windowHeight);

	// Build neural network
	nn = buildNeuralNetwork();
};

// Main loop
draw = () => {
	// Background
	background(1, 0, 2, 255);

	// Draw the network
	nn.draw();
};

// User-Events
// Resizes canvas' size when window is resized
windowResized = () => {
	resizeCanvas(windowWidth, windowHeight);
}
