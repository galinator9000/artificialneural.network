let MAIN_FONT;
preload = () => {
	MAIN_FONT = loadFont("assets/Inconsolata-SemiBold.ttf");
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
		// (set input unit to 1 initially)
		args: {inputShape: [1]}
	},

	// Hidden layers (create them randomly at start: min 1, max 3 layers)
	hiddenLayers: [...Array(getRandomInt(1, 4)).keys()].map(layer => (
		createDenseLayerConfig()
	)),

	// Output layer
	outputLayer: createDenseLayerConfig({
		// (set output unit to 1 initially)
		units: 1,
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
			scaleX: 0.65, scaleY: 0.60,
			translateX: -0.05,
			showBiasNeurons: false,
			weightVisualChangeSpeed: 0.25,
			neuronVisualChangeSpeed: 0.25,
			animatePropagation: true,
			propagation: {
				// Width and step values (ratio value for width of the canvas) of the propagation wave
				width: 0.005, step: 0.01,
				// Animation smoothing function
				animFn: AnimationUtils.easeOutQuad,
				// Apply animation layer by layer or to whole network?
				animationApplyType: (1 ? "layer" : "network")
			},
			neuronValueFont: MAIN_FONT
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
	// Get first sample on stage
	getStageSampleFromDataset(0);

	// Set neural network input/output layers' neuron count
	nnStructure.inputLayer.args.inputShape = [data.structure.n_features];
	nnStructure.outputLayer.args.units = data.structure.n_targets;
	onChangeNeuralNetwork();
};

// Initializes dataset with given URL
buildDataset = (csvURL) => {
	if(csvURL.length == 0 || !csvURL.endsWith(".csv")) return;

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

//// Sub canvas system
let subCanvas = {
	// Subcanvas objects 
	c: [
		{
			title: "Dataset",
			obj: null,
		},
		{
			title: "Neural Network",
			obj: null,
		},
		{
			title: "Stats",
			obj: null,
		},
	],

	currentIdx: 1,
	nextIdx: 1,

	// Animation of subcanvas transitions
	inTransition: false,
	transition: {
		x: 0.0,
		xAnim: 0.0,
		direction: 1,
		step: 0.05,
		animFn: AnimationUtils.easeOutExpo
	},
	leftTabWidthRatio: 0.05,
};

// Position converter functions
subCanvas.xToSubcanvasPosX = (x) => (x - (windowWidth * subCanvas.leftTabWidthRatio));
subCanvas.yToSubcanvasPosY = (y) => (y);

getSubCanvasWidthWithIndex = (cIdx) => {
	if(cIdx == -1) return windowWidth;
	return subCanvas.c[cIdx].obj.width;
};
getSubCanvasHeightWithIndex = (cIdx) => {
	if(cIdx == -1) return windowHeight;
	return subCanvas.c[cIdx].obj.height;
};

// Updates subcanvas related things
let updateSubCanvas = () => {
	// Update subcanvas transition value
	if(subCanvas.inTransition){
		// Step x
		subCanvas.transition.x += subCanvas.transition.step;

		// Limit between 0 and 1
		subCanvas.transition.x = Math.min(1, Math.max(0, subCanvas.transition.x));

		// Calculate animation value
		subCanvas.transition.xAnim = subCanvas.transition.animFn(subCanvas.transition.x);

		// Stop & reset transition when reached to the 1
		if(subCanvas.transition.x >= 1.0){
			subCanvas.transition.x = 0.0;
			subCanvas.transition.xAnim = 0.0;
			subCanvas.transition.direction = 1;
			subCanvas.inTransition = false;

			// Set current to the transitioned one
			subCanvas.currentIdx = subCanvas.nextIdx;
		}
	}
};

// Creates sub canvas objects
let createSubCanvas = () => {
	Object.entries(subCanvas.c).forEach(([k, v], cIdx) => {
		subCanvas.c[cIdx].obj = createGraphics(
			(windowWidth * (1 - subCanvas.leftTabWidthRatio)),
			windowHeight
		);
	});
};

//// GUI
let guiComponents = [];

// Initializes GUI components of main canvas & sub canvases
initializeGUI = () => {
	guiComponents = [
		//// Main components
		{
			subCanvasIndex: -1,
			obj: createImg(
				"./assets/join-globalaihub.png"
			),
			initCalls: [
				{fnName: "style", args: ["cursor", "pointer"]},
				{fnName: "mousePressed", args: [
					(() => openURLInNewTab("https://globalaihub.com"))
				]}
			],
			canvasRelativePosition: [0.805, 0.015],
			canvasRelativeSize: [0.14, 0.08]
		},

		//// NN components
		// Get sample button
		{
			subCanvasIndex: 1,
			obj: createButton("Get sample"),
			initCalls: [
				// Call for setting mouse press to getting random sample from dataset for stage
				{fnName: "mousePressed", args: [getStageSampleFromDataset]},
			],
			canvasRelativePosition: [0.03, 0.02],
			canvasRelativeSize: [0.10, 0.06]
		},
		// Predict button
		{
			subCanvasIndex: 1,
			obj: createButton("Predict"),
			initCalls: [
				// Call for setting mouse press to getting random sample from dataset for stage
				{fnName: "mousePressed", args: [
					(() => nn.predict(data.stageSample.input))
				]},
			],
			canvasRelativePosition: [0.14, 0.02],
			canvasRelativeSize: [0.10, 0.06]
		},
		// Add hidden layer button
		{
			subCanvasIndex: 1,
			obj: createButton("Add hidden layer"),
			initCalls: [
				// Call for setting mouse press to getting random sample from dataset for stage
				{fnName: "mousePressed", args: [
					(() => {
						// Add one layer to config & rebuild neural network
						nnStructure.hiddenLayers.push(createDenseLayerConfig());
						onChangeNeuralNetwork();
					})
				]},
			],
			canvasRelativePosition: [0.03, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		},
		// Remove hidden layer button
		{
			subCanvasIndex: 1,
			obj: createButton("Remove hidden layer"),
			initCalls: [
				// Call for setting mouse press to getting random sample from dataset for stage
				{fnName: "mousePressed", args: [
					(() => {
						// Remove hidden layers & rebuild neural network
						nnStructure.hiddenLayers = [];
						onChangeNeuralNetwork();
					})
				]},
			],
			canvasRelativePosition: [0.14, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		},
		// Reset network button
		{
			subCanvasIndex: 1,
			obj: createButton("Reset network"),
			initCalls: [
				// Call for setting mouse press to getting random sample from dataset for stage
				{fnName: "mousePressed", args: [
					(() => {
						// Reset configs & rebuild neural network
						nnStructure.hiddenLayers = [...Array(getRandomInt(1, 4)).keys()].map(layer => (createDenseLayerConfig()));
						onChangeNeuralNetwork();
					})
				]},
			],
			canvasRelativePosition: [0.25, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		}
	];

	// Call init calls of GUI components
	guiComponents.forEach(gc => {
		gc.initCalls.forEach((ic) => {
			// Call one of the init calls of GUI object
			gc.obj[ic.fnName](...ic.args);
		});
	});
};

// Updates GUI components of main canvas & sub canvases
updateGUI = () => {
	guiComponents.forEach(gc => {
		// Update position
		gc.obj.position(
			((windowWidth * subCanvas.leftTabWidthRatio) + (getSubCanvasWidthWithIndex(gc.subCanvasIndex) * gc.canvasRelativePosition[0])),
			(getSubCanvasHeightWithIndex(gc.subCanvasIndex) * gc.canvasRelativePosition[1]),
		);
		// Update size
		gc.obj.size(
			(getSubCanvasWidthWithIndex(gc.subCanvasIndex) * gc.canvasRelativeSize[0]),
			(getSubCanvasHeightWithIndex(gc.subCanvasIndex) * gc.canvasRelativeSize[1]),
		);

		// Hide the object
		gc.obj.style("display", "none");

		// Show the object if conditions are met
		if((gc.subCanvasIndex == -1) || (gc.subCanvasIndex == subCanvas.nextIdx)){
		// if((gc.subCanvasIndex == -1) || ((!subCanvas.inTransition) && (gc.subCanvasIndex == subCanvas.currentIdx))){
			gc.obj.show();
		}
	});
};

//// Sketch
// Setup
setup = () => {
	// Create main canvas
	createCanvas(windowWidth, windowHeight);

	colorMode(RGB);
	textAlign(CENTER, CENTER);
	textFont(MAIN_FONT);

	// Create the sub-canvases
	createSubCanvas();

	// Initialize&Update GUI components of subcanvases
	initializeGUI();
	updateGUI();

	// Initialize dataset
	buildDataset(csvURLs[0]);

	// Build neural network
	buildNeuralNetwork();
};

// Main loop
draw = () => {
	rectMode(CORNER);
	angleMode(DEGREES);

	// Clear backgrounds
	background(1, 0, 2, 255);
	Object.values(subCanvas.c).forEach(sc => {
		sc.obj.background(1, 0, 2, 255);
	});

	// Update GUI components of subcanvases
	updateGUI();

	// Draw the whole network on the given subcanvas
	nn.draw(
		subCanvas.c[1].obj,
		data.stageSample,
		// Additional vArgs
		{
			mouseX: subCanvas.xToSubcanvasPosX(mouseX),
			mouseY: subCanvas.yToSubcanvasPosY(mouseY)
		}
	);

	// Update subcanvas related things
	updateSubCanvas();

	// Starting X position for all subcanvases
	let startX = (windowWidth * subCanvas.leftTabWidthRatio);

	// Get the current&next subcanvas objects
	let currentCanvas = subCanvas.c[subCanvas.currentIdx].obj;
	let nextCanvas = subCanvas.c[subCanvas.nextIdx].obj;
	
	// Draw the current&next sub-canvas if transition is happening
	if(subCanvas.inTransition){
		// Calculate subcanvas' starting y positions
		let currentY = (
			(subCanvas.transition.direction * subCanvas.transition.xAnim * windowHeight)
		);
		let nextY = (
			(-1 * subCanvas.transition.direction * windowHeight) +
			(subCanvas.transition.direction * subCanvas.transition.xAnim * windowHeight)
		);

		image(
			currentCanvas,
			// Position
			startX, currentY,
			// Size
			currentCanvas.width, currentCanvas.height
		);
		image(
			nextCanvas,
			// Position
			startX, nextY,
			// Size
			nextCanvas.width, nextCanvas.height
		);
	}
	// Draw only the current sub-canvas if transition isn't happening
	else{
		image(
			currentCanvas,
			// Position
			startX, 0,
			// Size
			currentCanvas.width, currentCanvas.height
		);
	}

	// Draw subcanvas' tabs to the left
	let eachTabW = (windowWidth * subCanvas.leftTabWidthRatio);
	let eachTabH = (windowHeight / subCanvas.c.length);

	let curScIdx = (subCanvas.nextIdx);
	subCanvas.c.forEach((sc, scIdx) => {
		stroke(255);
		strokeWeight((scIdx == curScIdx) ? 3 : 0);

		// Line or rect
		line(eachTabW, (eachTabH * (scIdx)), eachTabW, (eachTabH * (scIdx+1)));
		// noFill(); rect(0, (eachTabH*scIdx), eachTabW, eachTabH);

		// Write tab title
		stroke((scIdx == curScIdx) ? 255 : 64);
		fill((scIdx == curScIdx) ? 255 : 64);
		strokeWeight(1);
		textSize(32);
		// Use translate&rotate for drawing titles sideways
		push();
		translate(eachTabW/2, ((eachTabH*scIdx) + eachTabH/2));
		rotate(-90);
		text(sc.title, 0, 0);
		pop();
	})
};

// User-Events
// Resizes canvas' size when window is resized
windowResized = () => {
	resizeCanvas(windowWidth, windowHeight);
	// Recreate subcanvas objects
	createSubCanvas();
};

// Change the current subcanvas (tab)
mouseWheel = (event) => {
	// Return if already in transition process
	if(subCanvas.inTransition) return;

	// Increment&decrement index of subcanvas
	if(event.deltaY > 0) subCanvas.nextIdx = subCanvas.currentIdx + 1;
	if(event.deltaY < 0) subCanvas.nextIdx = subCanvas.currentIdx - 1;

	// Limit index number
	subCanvas.nextIdx = Math.min(
		subCanvas.c.length-1,
		Math.max(0, subCanvas.nextIdx)
	);

	// Start transition
	if(subCanvas.nextIdx != subCanvas.currentIdx){
		subCanvas.transition.x = 0;
		subCanvas.transition.xAnim = 0;
		subCanvas.transition.direction = (subCanvas.nextIdx > subCanvas.currentIdx) ? -1 : 1;
		subCanvas.inTransition = true;
	}
};
