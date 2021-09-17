// GUI related variables and functions.

let guiComponents = [];
let cursors = [];

// Initializes GUI components of main canvas & sub canvases
initializeGUI = () => {
	// Set GUI cursors and conditions of them
	cursors = [
		{name: "pointer", condition: () => (mouseX < (windowWidth * subCanvas.leftTabWidthRatio))}
	];

	guiComponents = [
		//// Main GUI components

		// GlobalAIHub. REGISTER THERE! QUICK!
		{
			id: "join_globalaihub",
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

		// Add hidden layer button
		{
			id: "add_hidden_layer_button",
			subCanvasIndex: 1,
			obj: createButton("Add hidden layer"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => ((data.isLoading || !data.isCompiled || (nn && nn.isCompiled)))}
			],
			initCalls: [
				{fnName: "mousePressed", args: [
					(() => {
						// Add randomly built hidden layer to structure of the network
						nnStructure.hiddenLayersConfig.push(createDenseLayerConfig());
						// Reinitialize the network
						initializeNeuralNetwork();
					})
				]},
			],
			showCond: () => ((nn && !nn.isCompiled)),
			canvasRelativePosition: [0.29, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Compile network button
		{
			id: "compile_network_button",
			subCanvasIndex: 1,
			obj: createButton("Compile network!"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => ((data.isLoading || !data.isCompiled || (nn && nn.isCompiled)))}
			],
			initCalls: [
				{fnName: "mousePressed", args: [
					(() => {
						// Compile neural network
						compileNeuralNetwork();
						// Get random sample on stage
						getStageSampleFromDataset();
					})
				]},
			],
			showCond: () => ((nn && !nn.isCompiled)),
			canvasRelativePosition: [0.40, 0.92],
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
						// Reset & reinitialize the network
						resetNeuralNetwork();
						initializeNeuralNetwork();
					})
				]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.275, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Get sample button
		{
			id: "get_sample_button",
			subCanvasIndex: 1,
			obj: createButton("Get sample"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[getGUIComponentWithID("get_sample_button").subCanvasIndex].isActive()) || (nn && !nn.isCompiled)
				)}
			],
			initCalls: [
				// Get random sample from dataset for stage
				{fnName: "mousePressed", args: [getStageSampleFromDataset]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.385, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Predict button
		{
			id: "predict_button",
			subCanvasIndex: 1,
			obj: createButton("Predict"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[getGUIComponentWithID("predict_button").subCanvasIndex].isActive()) || (nn && !nn.isCompiled)
				)}
			],
			initCalls: [
				{fnName: "mousePressed", args: [
					// Predict current stage sample
					(() => nn.predict(data.stageSample.input))
				]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.495, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Fit button
		{
			id: "fit_button",
			subCanvasIndex: 1,
			obj: createButton("Train on dataset!"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[getGUIComponentWithID("fit_button").subCanvasIndex].isActive()) || (nn && !nn.isCompiled)
				)}
			],
			initCalls: [
				{fnName: "mousePressed", args: [
					// Fit the model on dataset
					(() => nn.fit(data.X, data.y, {epochs: 100, batchSize: data.structure.n_samples}))
				]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.605, 0.92],
			canvasRelativeSize: [0.10, 0.06]
		},

		//// Dataset GUI components

		// Dataset source text
		{
			id: "dataset_source_text",
			subCanvasIndex: 0,
			obj: createButton("Source"),
			initCalls: [
				// Behave as ghost button
				{fnName: "addClass", args: ["textButton"]},
			],
			canvasRelativePosition: [0.05, 0.02],
			canvasRelativeSize: [0.05, 0.06]
		},

		// Dataset source select / raw CSV URL provider
		{
			id: "dataset_url_select",
			subCanvasIndex: 0,
			obj: createSelect(),
			initCalls: [
				// Enter CSV URL option
				{fnName: "option", args: ["Enter CSV URL"]},

				// All constant options
				...(Object.entries(csvURLs).map(([key, value]) => ({fnName: "option", args: [key, value]}))),
				// First CSV URL is selected
				{fnName: "selected", args: [Object.values(csvURLs)[0]]},

				// onChange event
				{fnName: "changed", args: [
					(event) => {
						// Get selected value
						let selectComponent = getGUIComponentWithID("dataset_url_select").obj;
						let selectedURL = selectComponent.value();

						// Load given URL
						if(selectedURL === "Enter CSV URL"){
							selectedURL = window.prompt("Enter CSV URL");

							// Attempt to load given URL, if successful, add as an option
							loadDataset(selectedURL).then((success) => {
								if(success){
									// Add as option & make it selected
									selectComponent.option(selectedURL, selectedURL);
									selectComponent.selected(selectedURL);
								}
							});
						}
						// Load dataset
						else{
							loadDataset(selectedURL);
						}
					}
				]},
			],
			canvasRelativePosition: [0.10, 0.02],
			canvasRelativeSize: [0.25, 0.06]
		},

		// Dataset compile button
		{
			id: "compile_dataset_button",
			subCanvasIndex: 0,
			obj: createButton("Compile dataset!"),
			attributes: [
				// "Disabled" attribute for compile button (if data is compiled, disable it)
				{
					name: "disabled", value: "",
					condition: () => (
						(data.isLoading) || (data.isCompiled)
					)
				}
			],
			initCalls: [
				{fnName: "mousePressed", args: [
					(() => {
						// Compile dataset!
						compileDataset();
						// Switch to NN subcanvas
						switchSubCanvas(1);
					})
				]},
			],
			canvasRelativePosition: [0.36, 0.02],
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

// Removes GUI component with given ID
removeGUIComponentWithID = (id) => {
	// Get GUI component itself and it's index on the list
	let [gc, gcIdx] = guiComponents.map(
		(gc, gcIdx) => ([gc, gcIdx])
	).filter(
		([gc, gcIdx]) => (gc.id == id)
	)[0];

	// Remove HTML element, pop it from the list
	gc.obj.remove();
	guiComponents.splice(gcIdx, 1);
};

// Adds given GUI component & initializes it
addGUIComponent = (guiComponentObj) => {
	guiComponents.push(guiComponentObj);
	
	// Call init calls of added GUI object
	guiComponentObj.initCalls.forEach((ic) => {
		guiComponentObj.obj[ic.fnName](...ic.args);
	});
};

// Updates GUI components of main canvas & sub canvases
updateGUI = () => {
	// Set cursor according to mouse position
	cursor("");
	for(let c = 0; c<cursors.length; c++){
		if(cursors[c].condition()){
			cursor(cursors[c].name);
		}
	}

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
		if(
			(
				(gc.subCanvasIndex == -1)
				|| (gc.subCanvasIndex == subCanvas.nextIdx)
			) && ((gc.showCond === undefined) || (gc && gc.showCond && gc.showCond()))
		){
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
