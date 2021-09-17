// Dataset related variables and functions.

let data = {
	dataset: null,
	columns: {},
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
	},

	isCompiled: false,
	isLoading: false,
};

const csvURLs = {
	"Classification": "datasets/binary_classification_data.csv",
	"Regression": "datasets/regression_data.csv",
	"Boston Housing Regression": "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv"
};

// Resets everything about data
resetDataset = () => {
	data = {
		dataset: null,
		columns: {},
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
		},
	
		isCompiled: false,
		isLoading: false
	};
};

// Gets one sample and puts it to stage (side of the network)
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

// Compiles dataset, sets networks input/output unit counts
compileDataset = () => {
	// Set neural network input/output layers' neuron count
	nnStructure.inputLayerConfig.args.inputShape = [data.structure.n_features];
	nnStructure.outputLayerConfig.args.units = data.structure.n_targets;
	// Reinitialize neural network
	initializeNeuralNetwork();

	// Get input and target tensors of data
	data.X = tf.tensor(
		// Get all feature values in a nested-list
		data.dataset.map((row) => {
			return Object.entries(row).map(([k, v]) => {
				return (!data.columns[k].isTarget) ? v : null;
			}).filter(v => v !== null);
		}),
		// Shape
		[
			data.structure.n_samples,
			data.structure.n_features
		]
	);
	data.y = tf.tensor(
		// Get all target values in a nested-list
		data.dataset.map((row) => {
			return Object.entries(row).map(([k, v]) => {
				return (data.columns[k].isTarget) ? v : null;
			}).filter(v => v !== null);
		}),
		// Shape
		[
			data.structure.n_samples,
			data.structure.n_targets
		]
	);

	// Set dataset as compiled
	data.isCompiled = true;
	console.log("Dataset compiled");
};

// Loads&initializes dataset with given URL
loadDataset = async (url) => {
	if(url === null || url === undefined) url = "";
	if(url.length == 0 || !url.endsWith(".csv")) return false;

	// Set as loading
	data.isLoading = true;

	// Build CSVDataset & get full array
	let csvDataset = null;
	try{
		csvDataset = tf.data.csv(url);
	}catch{
		data.isLoading = false;
		return false;
	}
	if(!csvDataset){
		data.isLoading = false;
		return false;
	}

	let csvDatasetArray = await csvDataset.toArray();

	// Reset everything
	resetDataset();
	resetNeuralNetwork();

	// Set everything initially
	// Set builded dataset as main
	data.dataset = csvDatasetArray;

	// Get data columns
	data.columns = {};
	csvDataset.fullColumnNames.forEach(
		(colName, colIndex) => {
			data.columns[colName] = {
				isTarget: (colIndex == (csvDataset.fullColumnNames.length-1))
			}
		}
	);

	// Set data structure values
	data.structure.n_samples = csvDatasetArray.length;

	// Taking last column as target, others are X's (I SAID INITIALLY!)
	data.structure.n_features = (Object.keys(csvDatasetArray[0]).length-1);
	data.structure.n_targets = 1;

	console.log("Dataset loaded", data.structure);
	return true;
};

// Draws the dataset on the given canvas
drawDataset = (canvas, vArgs) => {
	// Calculate necessary values for drawing
	let tableW = (canvas.width * vArgs.scaleX);
	let tableH = (canvas.height * vArgs.scaleY);

	// Limit shown sample count
	let show_n_samples = max(min(data.structure.n_samples, 15), 10);
	let eachCellH = (tableH / show_n_samples);
	let eachCellW = (tableW / Object.keys(data.columns).length);

	let startTableX = (canvas.width * (1-vArgs.scaleX) / 2);
	let startTableY = (canvas.height * (1-vArgs.scaleY) / 2);
	let startCellX = startTableX + (eachCellW/2);
	let startCellY = startTableY + (eachCellH/2);

	// Table border
	canvas.push();
	canvas.strokeWeight(2);
	canvas.rect(
		startTableX+(tableW/2),
		startTableY+(tableH/2),
		tableW,
		tableH
	);
	canvas.pop();

	// Draw headers
	let rowIndex = 0;
	Object.entries(data.columns).forEach(([colName, colObj], colIndex) => {
		let centerX = (startCellX + (colIndex * eachCellW));
		let centerY = (startCellY + (rowIndex * eachCellH));

		// Header cell rect
		canvas.push();
		canvas.rect(centerX, centerY, eachCellW, eachCellH);
		canvas.pop();

		// Header text
		canvas.push();
		canvas.fill(255);
		canvas.textSize(
			calculateTextSize(
				colName.slice(0, 6),
				(eachCellW, eachCellH)
			)
		);
		canvas.text(colName.slice(0, 6), centerX, centerY);
		canvas.pop();
	});
};
