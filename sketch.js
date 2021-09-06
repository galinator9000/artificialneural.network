// Canvas objects
let nnCanvas;

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
	{class: tf.layers.dense, args: {"units": 16, "activation": "sigmoid"}},
	{class: tf.layers.dense, args: {"units": 12, "activation": "sigmoid"}},
	{class: tf.layers.dense, args: {"units": 8, "activation": "sigmoid"}},
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
			centerX: nnCanvas.width/2,
			centerY: nnCanvas.height/2,
			width: nnCanvas.width*0.80,
			height: nnCanvas.height*0.80
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
	// Create canvas of neural network
	nnCanvas = createCanvas(windowWidth*0.80, windowHeight*0.80);

	let button;
	button = createButton("predict");
	button.position(0, 0);
	button.mousePressed(args => {
		nn.predict(
			tf.randomNormal([1, n_features])
		);
	});

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
