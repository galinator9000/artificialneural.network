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
let nnCanvas;
let nnCanvasRatio = {x: 1.0, y: 1.0};
let nnCreateCanvas = () => {
	return createGraphics(
		(windowWidth*nnCanvasRatio.x),
		(windowHeight*nnCanvasRatio.y)
	);
};

let nn;
let nnHiddenLayersStructure = [
	// Hidden layers
	{class: tf.layers.dense, args: {"units": 16, "activation": "sigmoid", "useBias": true}},
	{class: tf.layers.dense, args: {"units": 12, "activation": "sigmoid", "useBias": true}},
	{class: tf.layers.dense, args: {"units": 8, "activation": "sigmoid", "useBias": true}},
	{class: tf.layers.dense, args: {"units": 4, "activation": "sigmoid", "useBias": true}},
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
// DOM objects
setup = () => {
	// Create main canvas
	createCanvas(windowWidth, windowHeight);

	// Create the canvas which will neural network be drawn
	nnCanvas = nnCreateCanvas();

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
	// Clear backgrounds
	background(1, 0, 2, 255);
	nnCanvas.background(1, 0, 2, 255);

	// Draw the whole network on the given canvas
	nn.draw(
		nnCanvas,
		{"gapRateX": 0.8, "gapRateY": 0.8}
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
