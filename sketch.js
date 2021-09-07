let NEURON_VALUE_FONT;
preload = () => {
	NEURON_VALUE_FONT = loadFont("assets/Inconsolata-SemiBold.ttf");
};

//// Neural network
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
		// (set input shape to random initially)
		args: {inputShape: [getRandomInt(4, 17)]}
	},

	// Hidden layers (create them randomly at start: min 1, max 3 layers)
	hiddenLayers: [...Array(getRandomInt(1, 4)).keys()].map(layer => (
		createDenseLayerConfig()
	)),

	// Output layer
	outputLayer: createDenseLayerConfig(
		// units (set to random initially)
		getRandomInt(1, 5),

		// useBias
		true,

		// activation
		// Regression
		// null,
		// Classification (binary)
		"sigmoid",
	),

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
			weightVisualChangeSpeed: 1,
			neuronVisualChangeSpeed: 0.25,
			propagationHighlight: {
				// Width of the highlight and speed of propagation (ratio value for width of the canvas)
				width: 0.005, speed: 0.05,
				// Animation smoothing function
				animFn: AnimationUtils.easeInExpo
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

	console.log(nnStructure);
};

//// Data
csvURLs = [
	"datasets/binary_classification_data.csv",
	"https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv"
];

// Main dataset
let data = {
	structure: {
		n_samples: 0,
		n_features: 0,
		n_targets: 0
	},
	dataset: null
};

getSampleFromDataset = (idx=null) => {
	// Get sample
	let sampleData = Object.values(data.dataset[
		(idx !== null ? idx : getRandomInt(0, data.dataset.length))
	]);

	// Get input and target output as tensor
	let X = tf.tensor([sampleData.slice(0, sampleData.length-1)]);
	let y = tf.tensor([sampleData.slice(sampleData.length-1)]);

	X.print();
	y.print();

	return [X, y];
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

		console.log(data.structure);
		
		onChangeDataset();
	});
};

//// Sketch
// Initialize GUI components
initializeGUI = () => {
	// Predict button
	let predictButton;
	predictButton = createButton("predict");
	predictButton.position(0, 0);
	predictButton.mousePressed(args => {
		// Generate sample
		// let sampleData = tf.randomNormal([1, data.structure.n_features]);

		// Get random sample from dataset
		let [X, y] = getSampleFromDataset();

		// Predict!
		nn.predict(X);
	});

	// Train button
	let trainButton;
	trainButton = createButton("train");
	trainButton.position(150, 0);
	trainButton.mousePressed(args => {
		// Get random sample from dataset
		// let [X, y] = getSampleFromDataset(0);

		// Get all samples
		let Xy = tf.tensor(
			// Get all samples in a nested-list
			data.dataset.map((d) => (Object.values(d))),
			// Shape
			[
				data.structure.n_samples,
				data.structure.n_features+data.structure.n_targets
			]
		);
		let X = Xy.slice([0, 0], [data.structure.n_samples, data.structure.n_features]);
		let y = Xy.slice([0, Xy.shape[1]-(data.structure.n_targets)], [data.structure.n_samples, data.structure.n_targets]);

		// Train!
		nn.fit(
			X, y,
			{
				epochs: 100,
				batchSize: 32
			}
		);
	});

	// Add hidden layer button
	let addHiddenLayerButton;
	addHiddenLayerButton = createButton("add hidden layer");
	addHiddenLayerButton.position(300, 0);
	addHiddenLayerButton.mousePressed(args => {
		// Add one layer to config & rebuild neural network
		nnStructure.hiddenLayers.push(createDenseLayerConfig());
		onChangeNeuralNetwork();
	});

	// Remove hidden layers button
	let removeHiddenLayersButton;
	removeHiddenLayersButton = createButton("remove hidden layers");
	removeHiddenLayersButton.position(450, 0);
	removeHiddenLayersButton.mousePressed(args => {
		// Remove hidden layers & rebuild neural network
		nnStructure.hiddenLayers = [];
		onChangeNeuralNetwork();
	});

	// Reset neural network button
	let resetnnButton;
	resetnnButton = createButton("reset nn");
	resetnnButton.position(600, 0);
	resetnnButton.mousePressed(args => {
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
