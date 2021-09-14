// Dataset related variables and functions.

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

const csvURLs = [
	"datasets/binary_classification_data.csv",
	"datasets/regression_data.csv",
	"https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv"
];

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

// Draws the dataset on the given canvas
drawDataset = (canvas, vArgs) => {

};
