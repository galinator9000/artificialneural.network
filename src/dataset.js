// Dataset related variables and functions.

var data = {};

// Various (configurable) visual arguments
var datasetVArgs = {
	scaleX: 0.95,
	scaleY: 0.85,
	translateX: 0.0125,
	translateY: 0.05,
	stepPerScroll: 3,
	n_showSample: 16,
};

// All CSV URLs
const csvURLs = {
	// "Iris dataset": ,
	// "MNIST dataset": ,

	//// Custom datasets
	
	// Generated with sklearn's make_classification/make_regression methods
	"Multiple linear regression": "datasets/multiple_linear_regression.csv",
	"Simple linear regression": "datasets/simple_linear_regression.csv",
	"Binary classification": "datasets/binary_classification.csv",
	// "Multiclass classification": "datasets/multiclass_classification.csv",

	// "XOR": "datasets/xor.csv",
	// "OR": "datasets/or.csv",
	// "AND": "datasets/and.csv",
};

// Calculates necessary values for drawing the dataset
calculateDatasetVArgs = (canvas) => {
	let tableW = (canvas.width * datasetVArgs.scaleX);
	let tableH = (canvas.height * datasetVArgs.scaleY);

	let eachCellW = (tableW / Object.keys(data.columns).length);
	// +1 for minding the header cells
	let eachCellH = (tableH / (datasetVArgs.n_showSample + 1));
	let startTableX = (canvas.width * (1-datasetVArgs.scaleX) / 2) + (canvas.width * datasetVArgs.translateX);
	let startTableY = (canvas.height * (1-datasetVArgs.scaleY) / 2) + (canvas.height * datasetVArgs.translateY);
	let startCellX = startTableX + (eachCellW/2);
	let startCellY = startTableY + (eachCellH/2);

	return [tableW, tableH, eachCellW, eachCellH, startTableX, startTableY, startCellX, startCellY];
};

// Resets dynamic dataset GUI components
resetDatasetGUI = () => {
	// Remove all GUI components that configures the dataset
	getGUIComponentIDs().filter(gcId => gcId.startsWith("dataset_cfg")).map(gcId => {removeGUIComponentWithID(gcId)});

	// Add new GUI components if dataset is loaded (loadDataset fn)
	if(data.dataset === null) return;

	//// Calculate necessary values for placing dynamic components
	let sc = subCanvas.c[DATASET_SUBCANVAS_INDEX].obj;
	
	let [
		tableW, tableH,
		eachCellW, eachCellH,
		startTableX, startTableY,
		startCellX, startCellY
	] = calculateDatasetVArgs(sc);

	// ... all goes into new GUI components' configs
	let datasetGUIComponentDefaults = {
		subCanvasIndex: DATASET_SUBCANVAS_INDEX,
		isDynamic: true,
		attributes: [
			// "Disabled" attribute if dataset is built
			{
				name: "disabled", value: "",
				condition: () => ((data.isLoading) || (data.isCompiled))
			}
		],
	};

	// Add headers as dynamic buttons
	Object.entries(data.columns).forEach(([colName, colObj], colIndex) => {
		let centerX = (startCellX + (colIndex * eachCellW));
		let centerY = startCellY;
		let componentId = "dataset_cfg_column_header_"+colIndex.toString();
		addGUIComponent({
			...datasetGUIComponentDefaults,
			id: componentId,
			obj: createButton(`${colName}${data.columns[colName].isTarget ? " (target)" : ""}`),
			initCalls: [
				{fnName: "addClass", args: ["button-bottom-border"]},
				{fnName: "mousePressed", args: [
					(() => {
						// Disabled (temporarily?)
						// data.columns[colName].isTarget = !(data.columns[colName].isTarget);
						// resetDatasetGUI();
					})
				]},
			],
			canvasRelativePosition: [(centerX/sc.width), (centerY/sc.height)],
			canvasRelativeSize: [(eachCellW*0.90/sc.width), (eachCellH*0.90/sc.height)]
		});
	});
};

// Scrolls through dataset
scrollDataset = (x, y, delta) => {
	if(Math.abs(delta) === 0) return;

	// Scroll if dataset is loaded (loadDataset fn)
	if(data.dataset === null) return;

	let [
		tableW, tableH,
		eachCellW, eachCellH,
		startTableX, startTableY,
		startCellX, startCellY
	] = calculateDatasetVArgs(subCanvas.c[DATASET_SUBCANVAS_INDEX].obj);

	// Check if mouse wheel event in dataset boundaries
	if(
		(x >= startTableX)
		&& (y >= startTableY)
		&& (x <= (startTableX + tableW))
		&& (y <= (startTableY + tableH))
	){
		// Get mouse wheel delta's direction as 1 or -1
		let direction = delta / Math.abs(delta);

		// Step scroll value & limit
		data.currentScrollIdx = Math.min(
			Math.max(
				data.currentScrollIdx + (direction * datasetVArgs.stepPerScroll),
				0
			),
			(
				data.structure.n_samples - Math.min(
					datasetVArgs.n_showSample, data.structure.n_samples
				)
			)
		);

		return true;
	}
	return false;
};

// (re)Sets everything about data
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

		currentScrollIdx: 0,
		isCompiled: false,
		isLoading: false
	};
	resetDatasetGUI();
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
	// Setting the feature&target counts, reading from columns object
	let new_n_features = arrSum(Object.values(data.columns).map(col => col.isTarget ? 0 : 1));
	let new_n_targets = arrSum(Object.values(data.columns).map(col => col.isTarget ? 1 : 0));

	if((new_n_features) <= 0 || (new_n_targets <= 0)){
		alert("No feature or target column provided");
		return false;
	}

	data.structure.n_features = new_n_features;
	data.structure.n_targets = new_n_targets;

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

	// Set neural network input/output layers' neuron count
	nnStructure.inputLayerConfig.args.inputShape = [data.structure.n_features];
	nnStructure.outputLayerConfig.args.units = data.structure.n_targets;
	// (Re)build neural network
	buildNeuralNetwork();

	// Set dataset as compiled
	data.isCompiled = true;
	console.log("Dataset compiled");
	resetDatasetGUI();
	return true;
};

// Loads&initializes dataset with given URL
loadDataset = async (url) => {
	if(url === null || url === undefined) url = "";
	if(url.length == 0) return false;

	// Set as loading
	data.isLoading = true;

	// Build CSVDataset & get full array
	let csvDataset = null;
	try{
		csvDataset = tf.data.csv(url);
	}catch(err){
		alert(`An error occured while loading the dataset :(`);
		data.isLoading = false;
		return false;
	}
	if(!csvDataset){
		alert(`An error occured while loading the dataset :(`);
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
				// Last column is target, initially.
				isTarget: ((csvDataset.fullColumnNames.length-1) === colIndex)
			}
		}
	);

	// Set data structure values
	data.structure.n_samples = csvDatasetArray.length;

	// Taking last column as target, others are X's (I SAID INITIALLY!)
	data.structure.n_features = (Object.values(data.columns).length)-1;
	data.structure.n_targets = (1);

	console.log("Dataset loaded", data.structure);
	resetDatasetGUI();
	return true;
};

// Draws the dataset on the given canvas
drawDataset = (canvas) => {
	// Draw the dataset if dataset is loaded (loadDataset fn)
	if(data.dataset === null) return;

	let [
		tableW, tableH,
		eachCellW, eachCellH,
		startTableX, startTableY,
		startCellX, startCellY
	] = calculateDatasetVArgs(canvas);

	// Draw rows
	data.dataset.slice(
		// Slice the dataset for getting the samples which will be drawn
		(data.currentScrollIdx),
		(data.currentScrollIdx + datasetVArgs.n_showSample)
	).forEach((rowObj, rowIdx) => {
		let centerY = (startCellY + ((rowIdx+1) * eachCellH));

		// Row number
		canvas.push();
		canvas.fill(255);
		canvas.textSize(
			calculateTextSize(
				"   ",
				(canvas.width * (1-datasetVArgs.scaleX) / 2)
			)
		);
		canvas.text(
			(data.currentScrollIdx+rowIdx+1).toString(),
			(canvas.width * (1-datasetVArgs.scaleX) / 2),
			centerY
		);
		canvas.pop();

		// Draw every column's value
		Object.entries(rowObj).forEach(([colName, colValue], colIdx) => {
			let centerX = (startCellX + (colIdx * eachCellW));
			
			// Value text
			canvas.push();
			canvas.fill(255);
			canvas.textSize(
				calculateTextSize(
					"     ",
					(eachCellH*2.5)
				)
			);
			canvas.text(colValue.toFixed(3), centerX, centerY);
			canvas.pop();
		});
	});
};
