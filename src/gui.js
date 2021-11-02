// GUI related variables and functions.

var guiComponents = [];
var cursors = [];

var buttonEvents = {
	// Resets & rebuilds the network
	resetNetwork: () => {
		resetNeuralNetwork();
		buildNeuralNetwork();

		//// Reset values
		// Propagation
		nn.vArgs.propagation.x = 0.0;
		nn.vArgs.propagation.xAnim = 0.0;
		nn.vArgs.propagation.xTarget = 0.0;
		nn.vArgs.predicted = false;
		nn.vArgs.backpropagated = false;
		// Auto train
		nn.vArgs.autoTrain.isEnabled = false;
		nn.vArgs.autoTrain.inProgress = false;
	},

	// Gets random sample from dataset for stage
	getSample: () => {
		getStageSampleFromDataset();

		// Reset values
		nn.vArgs.predicted = false;
		nn.vArgs.backpropagated = false;

		// Reset gradient values
		nn.resetGradients();
	},

	// Feed forwards with given matrix
	feedForward: (X) => {
		return nn.feedForward(X);
	},

	// Backpropagates with given input&target matrix
	backpropagate: (X, y) => {
		return nn.backpropagate(X, y);
	},

	// Applies gradients to weights, resets gradient values of Weight objects
	applyGradient: (X, y) => {
		if(nn.applyGradients(X, y)){
			nn.resetGradients();
			return true;
		}
		return false;
	},

	// Fits nn with given dataset matrices
	fit: (X, y) => {
		return nn.fit(
			X, y,
			{epochs: 10, batchSize: 32}
		);
	},
}

//// Define auto training events
buttonEvents.disableAutoTraining = () => {
	if(nn.vArgs.autoTrain.inProgress) return;
	nn.vArgs.autoTrain.isEnabled = false;
	nn.vArgs.autoTrain.clicked = false;
};
buttonEvents.enableAutoTraining = () => {
	if(nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.isEnabled) return;
	nn.vArgs.autoTrain.isEnabled = true;

	// Set sleep time
	let sleepMs;
	if(nn.vArgs.animatePropagation) sleepMs = 500;
	else sleepMs = 150;

	// Define as async and run immediately
	(async () => {
		do{
			nn.vArgs.autoTrain.inProgress = true;

			// Feed forward
			while(
				!buttonEvents.feedForward(data.stageSample.input)
				|| !nn.vArgs.autoTrain.isEnabled
			){
				await sleep(sleepMs);
			}
			if(nn.vArgs.autoTrain.isEnabled) await sleep(sleepMs*4);

			// Backpropagate
			while(
				!buttonEvents.backpropagate(data.stageSample.input, data.stageSample.target)
				|| !nn.vArgs.autoTrain.isEnabled
			){
				await sleep(sleepMs);
			}
			if(nn.vArgs.autoTrain.isEnabled) await sleep(sleepMs*4);

			// Apply gradient
			while(
				!buttonEvents.applyGradient(data.stageSample.input, data.stageSample.target)
				|| !nn.vArgs.autoTrain.isEnabled
			){
				await sleep(sleepMs);
			}

			// Done.
			nn.vArgs.autoTrain.inProgress = false;
			if(nn.vArgs.autoTrain.isEnabled) await sleep(1000);
		}while(nn.vArgs.autoTrain.isEnabled);
	})();

	nn.vArgs.autoTrain.clicked = false;
};

// Initializes GUI components of main canvas & sub canvases
initializeGUI = () => {
	// Set GUI cursors and conditions of them
	cursors = [
		{name: "pointer", condition: () => (mouseX < subCanvas.subcanvasStartX)}
	];

	guiComponents = [
		//// Dataset GUI components
		
		// Dataset source text
		{
			id: "dataset_source_text",
			subCanvasIndex: DATASET_SUBCANVAS_INDEX,
			obj: createButton("Problem type"),
			initCalls: [
				// Behave as ghost button
				{fnName: "addClass", args: ["text-button"]},
				{fnName: "style", args: ["z-index", "1"]},
			],
			canvasRelativePosition: [0.05625, 0.0625],
			canvasRelativeSize: [0.0875, 0.06]
		},

		// Dataset source select / raw CSV URL provider
		{
			id: "dataset_source_select",
			subCanvasIndex: DATASET_SUBCANVAS_INDEX,
			obj: createSelect(),
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},

				// Add initial option for text, disable and make it selected
				{fnName: "option", args: ["Select problem type"]},
				{fnName: "disable", args: ["Select problem type"]},
				{fnName: "selected", args: ["Select problem type"]},
				
				// Pick file option
				// {fnName: "option", args: ["Pick file..."]},

				// "Enter CSV URL" option
				// {fnName: "option", args: ["Enter CSV URL..."]},

				// All constant dataset source options
				...(Object.entries(csvURLs).map(([key, value]) => ({fnName: "option", args: [key, value]}))),
				// First problem type selected
				{fnName: "selected", args: [Object.values(csvURLs)[0]]},

				// onChange event
				{fnName: "changed", args: [
					(event) => {
						// Get selected value
						let selectComponent = getGUIComponentWithID("dataset_source_select").obj;
						let selectedValue = selectComponent.value();

						// Load given URL
						if(selectedValue === "Enter CSV URL..."){
							selectedValue = window.prompt(selectedValue);

							// Attempt to load given URL, if successful, add as an option
							loadDataset(selectedValue).then((success) => {
								if(success){
									// Add as option & make it selected
									selectComponent.option(selectedValue, selectedValue);
									selectComponent.selected(selectedValue);
								}
							});
						}
						// Pick file from local
						// else if(selectedValue === "Pick file..."){
						// 	document.getElementById("dataset-file-input").click();
						// }
						// Load dataset
						else{
							loadDataset(selectedValue);
						}
					}
				]},
			],
			canvasRelativePosition: [0.225, 0.0625],
			canvasRelativeSize: [0.25, 0.06]
		},

		// Dataset compile button
		{
			id: "dataset_compile_button",
			subCanvasIndex: DATASET_SUBCANVAS_INDEX,
			obj: createButton("Compile dataset!"),
			attributes: [
				// "Disabled" attribute for compile button (if data is compiled/loading, disable it)
				{
					name: "disabled", value: "",
					condition: () => (
						(data.isLoading) || (data.isCompiled)
					)
				}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [
					(() => {
						// Compile dataset, and switch to NN subcanvas
						if(compileDataset()) switchSubCanvas(NN_SUBCANVAS_INDEX);
					})
				]},
			],
			canvasRelativePosition: [0.41, 0.0625],
			canvasRelativeSize: [0.10, 0.06]
		},
		
		//// NN GUI components

		// Compile network button
		{
			id: "nn_compile_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Compile network!"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => ((data.isLoading || !data.isCompiled || (nn && nn.isCompiled)))}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
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
			canvasRelativePosition: [0.46, 0.0625],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Reset network button
		{
			id: "nn_reset_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Reset network"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[NN_SUBCANVAS_INDEX].isActive())
					|| (nn && !nn.isCompiled)
					|| (nn && nn.vArgs.propagation.inProgress)
					|| (nn && (nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.isEnabled))
				)}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [buttonEvents.resetNetwork]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.22, 0.9375],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Get sample button
		{
			id: "nn_get_sample_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Get sample"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[NN_SUBCANVAS_INDEX].isActive())
					|| (nn && !nn.isCompiled)
					|| (nn && nn.vArgs.propagation.inProgress)
					|| (nn && (nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.isEnabled))
				)}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [buttonEvents.getSample]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.33, 0.9375],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Feed forward button
		{
			id: "nn_feed_forward_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Predict"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[NN_SUBCANVAS_INDEX].isActive())
					|| (nn && !nn.isCompiled)
					|| (nn && nn.vArgs.propagation.inProgress)
					|| (nn && (nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.isEnabled))
				)}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [() => {
					buttonEvents.feedForward(data.stageSample.input);
				}]},
			],
			showCond: () => ((nn && nn.isCompiled && (
				// Training stage
				(!nn.vArgs.predicted && !nn.vArgs.backpropagated)
			))),
			canvasRelativePosition: [0.44, 0.9375],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Backpropagate button
		{
			id: "nn_backpropagate_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Backpropagate"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[NN_SUBCANVAS_INDEX].isActive())
					|| (nn && !nn.isCompiled)
					|| (nn && nn.vArgs.propagation.inProgress)
					|| (nn && (nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.isEnabled))
				)}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [() => {
					buttonEvents.backpropagate(data.stageSample.input, data.stageSample.target);
				}]},
			],
			showCond: () => ((nn && nn.isCompiled && (
				// Training stage
				(nn.vArgs.predicted && !nn.vArgs.backpropagated)
			))),
			canvasRelativePosition: [0.44, 0.9375],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Apply gradients button
		{
			id: "nn_apply_gradients_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Apply gradients"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[NN_SUBCANVAS_INDEX].isActive())
					|| (nn && !nn.isCompiled)
					|| (nn && nn.vArgs.propagation.inProgress)
					|| (nn && (nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.isEnabled))
				)}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [() => {
					buttonEvents.applyGradient(data.stageSample.input, data.stageSample.target);
				}]},
			],
			showCond: () => ((nn && nn.isCompiled && (
				// Training stage
				(nn.vArgs.predicted && nn.vArgs.backpropagated)
			))),
			canvasRelativePosition: [0.44, 0.9375],
			canvasRelativeSize: [0.10, 0.06]
		},

		// Train with sample button
		{
			id: "nn_train_wsample_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Train with sample!"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[NN_SUBCANVAS_INDEX].isActive())
					|| (nn && !nn.isCompiled)
					|| (nn && nn.vArgs.propagation.inProgress)
					|| (nn && (nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.clicked))
				)}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [
					() => {
						nn.vArgs.autoTrain.clicked = true;

						// Activate/Deactivate training loop
						if(nn && nn.vArgs.autoTrain.isEnabled){
							buttonEvents.disableAutoTraining();
							getGUIComponentWithID("nn_train_wsample_button").obj.elt.innerHTML = "Train with sample!";
						}else{
							buttonEvents.enableAutoTraining();
							getGUIComponentWithID("nn_train_wsample_button").obj.elt.innerHTML = "Stop training";
						}
					}
				]},
			],
			showCond: () => ((nn && nn.isCompiled) && !nn.vArgs.autoTrain.clicked),
			canvasRelativePosition: [0.57, 0.9375],
			canvasRelativeSize: [0.14, 0.06]
		},

		// Sample group title
		{
			id: "nn_sample_group_title",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("with Sample"),
			attributes: [{name: "disabled", value: "", condition: () => true}],
			initCalls: [
				{fnName: "addClass", args: ["button-bottom-border"]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.46, 0.8825],
			canvasRelativeSize: [0.35, 0.03]
		},

		// Fit dataset button
		{
			id: "nn_fit_dataset_button",
			subCanvasIndex: NN_SUBCANVAS_INDEX,
			obj: createButton("Fit dataset!"),
			attributes: [
				// "Disabled" attribute for button
				{name: "disabled", value: "", condition: () => (
					(!subCanvas.c[NN_SUBCANVAS_INDEX].isActive())
					|| (nn && !nn.isCompiled)
					|| (nn && nn.vArgs.propagation.inProgress)
					|| (nn && (nn.vArgs.autoTrain.inProgress || nn.vArgs.autoTrain.isEnabled))
				)}
			],
			initCalls: [
				{fnName: "style", args: ["z-index", "1"]},
				{fnName: "mousePressed", args: [
					() => {
						// Fit dataset
						buttonEvents.fit(data.X, data.y);
					}
				]},
			],
			showCond: () => ((nn && nn.isCompiled)),
			canvasRelativePosition: [0.70, 0.9375],
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

getGUIComponentIDs = () => {
	return guiComponents.map(gc => gc.id);
};

getGUIComponentsIdIndexPair = () => {
	let id_index_pair = {};
	guiComponents.forEach((gc, gcIdx) => {
		id_index_pair[gc.id] = gcIdx;
	});
	return id_index_pair;
};

// Removes GUI component with given ID
removeGUIComponentWithID = (removeId) => {
	let removeComponentIDs = guiComponents.map(gc => gc.id).filter(gcId => gcId == removeId);

	removeComponentIDs.forEach(gcId => {
		// Get current index of the GUI component
		let gcIdx = getGUIComponentsIdIndexPair()[gcId];

		// Remove HTML element, pop it from the component list
		guiComponents[gcIdx].obj.remove();
		guiComponents.splice(gcIdx, 1);
	});
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

		//// Update position & size of GUI component

		// Get canvas relative position & sizes
		let relativePositionX = gc.canvasRelativePosition[0];
		let relativePositionY = gc.canvasRelativePosition[1];
		let relativeSizeX = gc.canvasRelativeSize[0];
		let relativeSizeY = gc.canvasRelativeSize[1];

		// Calculate absolute position & size relative to absolute canvas
		let subCanvasWidth = getSubCanvasWidthWithIndex(gc.subCanvasIndex);
		let subCanvasHeight = getSubCanvasHeightWithIndex(gc.subCanvasIndex);

		let posX; let posY; let sizeX; let sizeY;

		// Calculate the transformed position & size of the component relative to main canvas (dynamic component)
		if((gc && gc.isDynamic) && (gc.subCanvasIndex !== -1)){
			let scPosVec = subCanvas.c[gc.subCanvasIndex].subCanvasPos_to_absoluteCanvasPos(
				(subCanvasWidth * relativePositionX),
				(subCanvasHeight * relativePositionY)
			);
			posX = scPosVec.x;
			posY = scPosVec.y;
			sizeX = (subCanvasWidth * relativeSizeX) * subCanvas.transform.scale.x;
			sizeY = (subCanvasHeight * relativeSizeY) * subCanvas.transform.scale.y;
		}
		// Calculate absolute position & size relative to main canvas (static component)
		else{
			sizeX = (subCanvasWidth * relativeSizeX);
			sizeY = (subCanvasHeight * relativeSizeY);
			posX = (subCanvas.subcanvasStartX + (subCanvasWidth * relativePositionX));
			posY = ((subCanvasHeight * relativePositionY));
		}

		// Center -> top-left corner position
		posX -= (sizeX/2);
		posY -= (sizeY/2);
		
		// Set position & size of the GUI component
		gc.obj.position(posX, posY);
		gc.obj.size(sizeX, sizeY);

		// Check the component if it's in the boundaries of the main canvas
		let inBoundaries = false;
		if(
			(posX > subCanvas.subcanvasStartX)
			&& (posY > 0)
			&& ((posX + sizeX) < windowWidth)
			&& ((posY + sizeY) < windowHeight)
		) inBoundaries = true;

		// Hide the object
		gc.obj.style("display", "none");

		// Show the object if conditions are met
		if(
			(
				(gc.subCanvasIndex == -1)
				|| (gc.subCanvasIndex == subCanvas.nextIdx)
			)
			&& ((gc.showCond === undefined) || (gc && gc.showCond && gc.showCond()))
			&& (inBoundaries)
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
