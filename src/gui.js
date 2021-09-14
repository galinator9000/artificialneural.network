// GUI related variables and functions.

let guiComponents = [];

// Initializes GUI components of main canvas & sub canvases
initializeGUI = () => {
	guiComponents = [
		//// Main components
		{
			subCanvasIndex: -1,
			obj: createImg(
				"assets/join-globalaihub.png"
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
