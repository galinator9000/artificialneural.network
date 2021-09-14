// GUI related variables and functions.

let guiComponents = [];

// Initializes GUI components of main canvas & sub canvases
initializeGUI = () => {
	guiComponents = [
		//// Main GUI components

		// GlobalAIHub. REGISTER THERE! QUICK!
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

		//// NN GUI components

		// Get sample button
		{
			id: "get_sample_button",
			subCanvasIndex: 1,
			obj: createButton("Get sample"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (!subCanvas.c[getGUIComponentWithID("get_sample_button").subCanvasIndex].isActive())}
			],
			initCalls: [
				// Get random sample from dataset for stage
				{fnName: "mousePressed", args: [getStageSampleFromDataset]},
			],
			canvasRelativePosition: [0.03, 0.02],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Predict button
		{
			id: "predict_button",
			subCanvasIndex: 1,
			obj: createButton("Predict"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (!subCanvas.c[getGUIComponentWithID("predict_button").subCanvasIndex].isActive())}
			],
			initCalls: [
				{fnName: "mousePressed", args: [
					// Predict current stage sample
					(() => nn.predict(data.stageSample.input))
				]},
			],
			canvasRelativePosition: [0.14, 0.02],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Add hidden layer button
		{
			id: "add_hidden_layer_button",
			subCanvasIndex: 1,
			obj: createButton("Add hidden layer"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (!subCanvas.c[getGUIComponentWithID("add_hidden_layer_button").subCanvasIndex].isActive())}
			],
			initCalls: [
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
			id: "remove_hidden_layer_button",
			subCanvasIndex: 1,
			obj: createButton("Remove hidden layer"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (!subCanvas.c[getGUIComponentWithID("remove_hidden_layer_button").subCanvasIndex].isActive())}
			],
			initCalls: [
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
			id: "reset_network_button",
			subCanvasIndex: 1,
			obj: createButton("Reset network"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (!subCanvas.c[getGUIComponentWithID("reset_network_button").subCanvasIndex].isActive())}
			],
			initCalls: [
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
		},

		//// Dataset GUI components

		// Dataset source text
		{
			subCanvasIndex: 0,
			obj: createButton("Source"),
			initCalls: [
				// Behave as ghost button
				{fnName: "addClass", args: ["textButton"]},
			],
			canvasRelativePosition: [0.02, 0.02],
			canvasRelativeSize: [0.05, 0.06]
		},

		// Dataset source select / raw CSV URL provider
		{
			id: "dataset_url_select",
			subCanvasIndex: 0,
			obj: createSelect(),
			initCalls: [
				// Add option
				{fnName: "option", args: ["Enter CSV URL"]},

				// All options
				...(csvURLs.map(url => ({fnName: "option", args: [url]}))),
				// First CSV URL is selected
				{fnName: "selected", args: [csvURLs[0]]},

				// onChange event
				{fnName: "changed", args: [
					(event) => {
						// Get selected value
						let selectComponent = getGUIComponentWithID("dataset_url_select").obj;
						let selectedValue = selectComponent.value();

						// Load given URL
						if(selectedValue === "Enter CSV URL"){
							let selectedValue = window.prompt("Enter CSV URL");
							if(loadDataset(selectedValue)){
								// Add as option & make it selected
								selectComponent.option(selectedValue);
								selectComponent.selected(selectedValue);
							}
						}
						// Load dataset
						else{
							loadDataset(selectedValue);
						}
					}
				]},
			],
			canvasRelativePosition: [0.07, 0.02],
			canvasRelativeSize: [0.25, 0.06]
		},

		// Dataset compile button
		{
			id: "compile_dataset_button",
			subCanvasIndex: 0,
			obj: createButton("Compile dataset"),
			initCalls: [
				{fnName: "mousePressed", args: [
					(() => {
						// Compile dataset!
						compileDataset();
					})
				]},
			],
			canvasRelativePosition: [0.33, 0.02],
			canvasRelativeSize: [0.10, 0.06]
		},
	];

	// Call init calls of GUI components
	guiComponents.forEach(gc => {
		gc.initCalls.forEach((ic) => {
			// Call one of the init calls of GUI object
			gc.obj[ic.fnName](...ic.args);
		});
	});
};

getGUIComponentWithID = (id) => {
	return guiComponents.filter(gc => (gc.id && gc.id == id))[0];
};

// Updates GUI components of main canvas & sub canvases
updateGUI = () => {
	guiComponents.forEach(gc => {
		// Call updates of the GUI component
		((gc && gc.updateCalls) ? gc.updateCalls : []).forEach((uc) => {
			// Call from object if function is property of the GUI component
			gc.obj[uc.fnName](...uc.args);
		});

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

		// Process attributes
		((gc && gc.attributes) ? gc.attributes : []).forEach(attr => {
			if(attr.condition()){
				gc.obj.attribute(attr.name, attr.value);
			}else{
				gc.obj.removeAttribute(attr.name);
			}
		});
	});
};
