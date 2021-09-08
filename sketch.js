let NEURON_VALUE_FONT;
preload = () => {
	NEURON_VALUE_FONT = loadFont("assets/Inconsolata-SemiBold.ttf");
};

//// Neural network
// Creates a dense layer config object with given values
let createDenseLayerConfig = (denseArgs={activation: "sigmoid"}) => ({
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

// Specifies our neural network structure (layers, losses etc.)
let nnStructure = {
	// Input layer
	inputLayer: {
		class: tf.layers.inputLayer,
		// (set input shape to random initially)
		args: {inputShape: [getRandomInt(4, 17)]}
	},

	// Hidden layers (create them randomly at start: min 1, max 3 layers)
	hiddenLayers: [...Array(getRandomInt(1, 4)).keys()].map(layer => (
		createDenseLayerConfig()
	)),

	// Output layer
	outputLayer: createDenseLayerConfig({
		units: getRandomInt(1, 5),
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

// Canvas which our neural network will be drawn
let nnCanvas;
let nnCanvasRatio = {x: 1.0, y: 1.0};
let nnCreateCanvas = () => {
	nnCanvas = createGraphics(
		(windowWidth*nnCanvasRatio.x),
		(windowHeight*nnCanvasRatio.y)
	);
};

// Our main neural network model
let nn;

// Gets called whenever neural network needs to rebuild
onChangeNeuralNetwork = () => {
	buildNeuralNetwork();
};

// Builds neural network object at tf.js side with structure config
buildNeuralNetwork = () => {
	// Build NN sequentially with our custom class
	nn = new SequentialNeuralNetwork(
		// Arguments which will be passed to tf.Sequential
		sequentialArgs={},

		// Various visual arguments
		vArgs={
			gapRateX: 0.8, gapRateY: 0.8,
			weightVisualChangeSpeed: 0.25,
			neuronVisualChangeSpeed: 0.25,
			propagation: {
				// Width and step values (ratio value for width of the canvas) of the propagation wave
				width: 0.005, step: 0.01,
				// Animation smoothing function
				animFn: AnimationUtils.easeOutExpo,
				// Apply animation layer by layer or to whole network?
				animationApplyType: (1 ? "layer" : "network")
			},
			neuronValueFont: NEURON_VALUE_FONT
		}
	);

	// Put all layer configs in a list, add each of them to the model
	[
		nnStructure.inputLayer,
		...nnStructure.hiddenLayers,
		nnStructure.outputLayer
	].forEach(layer => {
		nn.add(
			layer.class(layer.args)
		);
	});

	// Compile the model with args
	nn.compile(nnStructure.compileArgs);

	console.log("NN built", nnStructure);
};

//// Data
csvURLs = [
	"datasets/binary_classification_data.csv",
	"datasets/regression_data.csv",
	"https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv"
];

// Holds data related objects
let data = {
	dataset: null,
	structure: {
		n_samples: 0,
		n_features: 0,
		n_targets: 0
	},
	X: null,
	y: null,
	stageSample: {
		input: null,
		target: null
	}
};

getStageSampleFromDataset = (idx=null) => {
	// Sample randomly if index is not given
	idx = (idx !== null ? idx : getRandomInt(0, data.structure.n_samples));

	// Get input and target output of sample as tensor, set to stage
	data.stageSample.input = data.X.slice([idx, 0], [1, data.structure.n_features]);
	data.stageSample.target = data.y.slice([idx, 0], [1, data.structure.n_targets]);

	console.log("Sampled:");
	console.log(data.stageSample.input.toString());
	console.log(data.stageSample.target.toString());
};

// Gets called whenever dataset changes
onChangeDataset = () => {
	// Set neural network input/output layers' neuron count
	nnStructure.inputLayer.args.inputShape = [data.structure.n_features];
	nnStructure.outputLayer.args.units = data.structure.n_targets;
	onChangeNeuralNetwork();
};

// Initializes dataset with given URL
buildDataset = (csvURL) => {
	// Build CSVDataset & get full array
	tf.data.csv(csvURL).toArray().then(csvDataset => {
		// Set builded dataset as main
		data.dataset = csvDataset;

		// Set data structure values
		data.structure.n_samples = data.dataset.length;
		// Taking last column as target, others are X's
		data.structure.n_features = (Object.keys(data.dataset[0]).length-1);
		data.structure.n_targets = 1;

		// Get input and target tensors
		data.X = tf.tensor(
			// Get all feature values in a nested-list
			data.dataset.map((d) => {
				d = Object.values(d);
				return d.slice(1);
			}),
			// Shape
			[
				data.structure.n_samples,
				data.structure.n_features
			]
		);
		data.y = tf.tensor(
			// Get all target values in a nested-list
			data.dataset.map((d) => {
				d = Object.values(d);
				return d.slice(0, 1);
			}),
			// Shape
			[
				data.structure.n_samples,
				data.structure.n_targets
			]
		);

		console.log("Dataset built", data.structure);
		onChangeDataset();
	});
};

//// Sketch
// Initialize GUI components
initializeGUI = () => {
	// Create buttons
	let buttons = {
		sampleStageButton: createButton("Get sample"),
		predictButton: createButton("Predict"),
		sampleStagePredictButton: createButton("Get sample&predict"),
		fitButton: createButton("Fit dataset"),
		addHiddenLayerButton: createButton("Add hidden layer"),
		removeHiddenLayersButton: createButton("Remove hidden layers"),
		resetButton: createButton("Reset network"),
	}

	// Set positions of buttons
	Object.values(buttons).forEach((button, buttonIndex) => {
		button.position(50 + (buttonIndex * 150), 10);
	});

	// Sample button
	buttons.sampleStageButton.mousePressed(args => {
		// Get random sample from dataset for stage
		getStageSampleFromDataset();
	});

	// Predict button
	buttons.predictButton.mousePressed(args => {
		// Predict!
		nn.predict(data.stageSample.input);
	});

	// Sample&predict button
	buttons.sampleStagePredictButton.mousePressed(args => {
		// Get random sample from dataset for stage
		getStageSampleFromDataset();
		// Predict!
		nn.predict(data.stageSample.input);
	});

	// Train button
	buttons.fitButton.mousePressed(args => {
		// Train!
		nn.fit(
			data.X, data.y,
			{
				epochs: 100,
				batchSize: data.structure.n_samples
			}
		);
	});

	// Add hidden layer button
	buttons.addHiddenLayerButton.mousePressed(args => {
		// Add one layer to config & rebuild neural network
		nnStructure.hiddenLayers.push(createDenseLayerConfig());
		onChangeNeuralNetwork();
	});

	// Remove hidden layers button
	buttons.removeHiddenLayersButton.mousePressed(args => {
		// Remove hidden layers & rebuild neural network
		nnStructure.hiddenLayers = [];
		onChangeNeuralNetwork();
	});

	// Reset neural network button
	buttons.resetButton.mousePressed(args => {
		// Reset configs & rebuild neural network
		nnStructure.hiddenLayers = [...Array(getRandomInt(1, 4)).keys()].map(layer => (createDenseLayerConfig()));
		onChangeNeuralNetwork();
	});
};

// Setup
setup = () => {
	// Create main canvas
	createCanvas(windowWidth, windowHeight);

	// Create the canvas which will neural network be drawn
	nnCreateCanvas();

	// Initialize GUI components
	initializeGUI();

	// Initialize dataset
	buildDataset(csvURLs[0]);

	// Build neural network
	buildNeuralNetwork();
};

// Main loop
draw = () => {
	// Clear backgrounds
	background(1, 0, 2, 255);
	nnCanvas.background(1, 0, 2, 255);

	// Draw the whole network on the given canvas
	nn.draw(nnCanvas);

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
	nnCreateCanvas();
};
