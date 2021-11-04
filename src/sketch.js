// Constants
var MAIN_FONT;
var BG_COLOR;
preload = () => {
	MAIN_FONT = loadFont("assets/Inconsolata-Medium.ttf");
};

// Setup
setup = () => {
	BG_COLOR = color(1, 0, 2, 255);

	// Create main canvas
	createCanvas(windowWidth, windowHeight);

	// Apply canvas settings which will not be changed
	colorMode(RGB);
	angleMode(DEGREES);
	textFont(MAIN_FONT);
	textAlign(CENTER, CENTER);
	rectMode(CENTER, CENTER);

	// Default values unless otherwise specified while drawing
	noFill();
	stroke(255);
	strokeWeight(1);

	// Create the sub-canvases
	createSubCanvas();

	// Initialize & Update once GUI components
	initializeGUI();
	updateGUI();

	// Assign data structure initially
	resetDataset();

	// Init dummy network on main page
	initializeDummyNeuralNetwork();

	loadDataset(
		Object.values(csvURLs)[0]
	);
};

// Main loop
draw = () => {
	// Clear backgrounds
	background(BG_COLOR);

	// Update GUI components
	updateGUI();

	// Process dummy nn
	if(dummynn){
		// Get the canvas
		let dummynnCanvas = subCanvas.c[MAIN_SUBCANVAS_INDEX];

		// Update the dummy network
		dummynn.update(dummynnCanvas.obj);

		// Check if neural network should be drawn to it's subcanvas
		if(shouldSubCanvasBeDrawn(MAIN_SUBCANVAS_INDEX)){
			// Clear background
			dummynnCanvas.obj.background(BG_COLOR);

			dummynnCanvas.obj.push();

			// Apply transformations to the nn canvas
			applyTransformationsToSubCanvas(dummynnCanvas.obj);

			// Draw network
			dummynn.initDraw(dummynnCanvas.obj);
			dummynn.draw(dummynnCanvas.obj);
			
			dummynnCanvas.obj.pop();
		}
	}

	// Process nn if built
	if(nn){
		// Get the canvas
		let nnCanvas = subCanvas.c[NN_SUBCANVAS_INDEX];

		// Update network
		nn.update(nnCanvas.obj);

		// Check if neural network should be drawn to it's subcanvas
		if(shouldSubCanvasBeDrawn(NN_SUBCANVAS_INDEX)){
			// Clear background
			nnCanvas.obj.background(BG_COLOR);

			nnCanvas.obj.push();

			// Apply transformations to the nn canvas
			applyTransformationsToSubCanvas(nnCanvas.obj);

			// Draw network
			nn.initDraw(nnCanvas.obj);
			nn.draw(nnCanvas.obj, data.stageSample);
			
			nnCanvas.obj.pop();
		}
	}

	// Check if dataset should be drawn to it's subcanvas
	let datasetCanvas = subCanvas.c[DATASET_SUBCANVAS_INDEX];
	if(shouldSubCanvasBeDrawn(DATASET_SUBCANVAS_INDEX)){
		// Clear background
		datasetCanvas.obj.background(BG_COLOR);
		datasetCanvas.obj.push();

		// Apply transformations to the dataset canvas
		applyTransformationsToSubCanvas(datasetCanvas.obj);

		// Draw dataset on given subcanvas
		drawDataset(datasetCanvas.obj);

		datasetCanvas.obj.pop();
	}

	// Update subcanvas related things
	updateSubCanvas();

	//// Draw subcanvases over main one
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

		// Draw subcanvases on to main canvas
		push();
		image(
			currentCanvas,
			// Position
			subCanvas.subcanvasStartX, currentY,
			// Size
			currentCanvas.width, currentCanvas.height
		);
		image(
			nextCanvas,
			// Position
			subCanvas.subcanvasStartX, nextY,
			// Size
			nextCanvas.width, nextCanvas.height
		);
		pop();
	}
	// Draw only the current sub-canvas if transition isn't happening
	else{
		push();
		image(
			currentCanvas,
			// Position
			subCanvas.subcanvasStartX, 0,
			// Size
			currentCanvas.width, currentCanvas.height
		);
		pop();
	}

	//// Draw subcanvas tabs to the left of the screen
	let eachTabW = subCanvas.subcanvasStartX;
	let eachTabH = (windowHeight / subCanvas.c.length);
	let curScIdx = (subCanvas.nextIdx);

	// Calculate tab titles' text size
	let tabTextSize = calculateTextsSize(
		subCanvas.c.map(sc => sc.title),
		(eachTabH*0.90)
	);

	// Draw each tab title
	subCanvas.c.forEach((sc, scIdx) => {
		// Active tab underline
		push();
		strokeWeight((scIdx == curScIdx) ? 3 : 0);
		line(eachTabW, (eachTabH * (scIdx)), eachTabW, (eachTabH * (scIdx+1)));
		pop();

		// Write tab title (use translate&rotate for drawing titles sideways)
		push();
		fill(subCanvas.c[scIdx].isActive() ? 192 : 64);
		stroke(subCanvas.c[scIdx].isActive() ? 192 : 64);
		translate(eachTabW/2, ((eachTabH*scIdx) + eachTabH/2));
		rotate(-90);
		textSize(tabTextSize);
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

// Processes mouse wheel events
mouseWheel = (event) => {
	// Reject event during transition (should return true!)
	if(subCanvas.inTransition) return true;

	// Get current subcanvas
	let sc = subCanvas.c[subCanvas.currentIdx];
	let eventProcessed = false;

	// Check if event occured at subcanvas region and event exists.
	if((event.x > subCanvas.subcanvasStartX) && sc.eventHandlers && sc.eventHandlers.mouseWheel){
		// Main canvas position to subcanvas position
		let scPosVec = subCanvas.c[subCanvas.currentIdx].absolutePos_to_SubCanvasPos(event.x, event.y);

		// Call the event handler
		eventProcessed = sc.eventHandlers.mouseWheel(
			scPosVec.x,
			scPosVec.y,
			event.deltaY
		);
	}

	//// If subcanvases didn't process the event, main functionality handles it
	if(!eventProcessed){
		// Change the current tab
		if(event.deltaY > 0) switchSubCanvas(subCanvas.currentIdx+1);
		if(event.deltaY < 0) switchSubCanvas(subCanvas.currentIdx-1);
	}

	return true;
};

// Processes mouse click events
mouseClicked = (event) => {
	// Reject event during transition (should return true!)
	if(subCanvas.inTransition) return true;
	// Avoid unwanted side effect
	if((event.x === 0) || (event.y === 0)) return true;

	// Get current subcanvas
	let sc = subCanvas.c[subCanvas.currentIdx];
	let eventProcessed = false;

	// Check if event occured at subcanvas region and event exists.
	if((event.x > subCanvas.subcanvasStartX) && sc.eventHandlers && sc.eventHandlers.mouseClicked){
		// Main canvas position to subcanvas position
		let scPosVec = subCanvas.c[subCanvas.currentIdx].absolutePos_to_SubCanvasPos(event.x, event.y);

		// Call the event handler
		eventProcessed = sc.eventHandlers.mouseClicked(
			scPosVec.x,
			scPosVec.y
		);
	}

	//// If subcanvases didn't process the event, main functionality handles it
	if(!eventProcessed){
		// Check if clicked on the tab section
		if(event.x < subCanvas.subcanvasStartX){
			// Switch clicked tab
			let eachTabH = (windowHeight / subCanvas.c.length);
			let clickedSubCanvasIndex = Math.floor(event.y / eachTabH);
			switchSubCanvas(clickedSubCanvasIndex);
		}
	}

	return true;
};

// Processes mouse drag events
mouseDragged = (event) => {
	// Reject event during transition (should return true!)
	if(subCanvas.inTransition) return true;

	// Get current subcanvas
	let sc = subCanvas.c[subCanvas.currentIdx];
	let eventProcessed = false;

	// Check if event occured at subcanvas region and event exists.
	if((event.x > subCanvas.subcanvasStartX) && sc.eventHandlers && sc.eventHandlers.mouseDragged){
		// Call the event handler
		eventProcessed = sc.eventHandlers.mouseDragged(event.movementX, event.movementY);
	}

	//// If subcanvases didn't process the event, main functionality handles it
	if(!eventProcessed){}

	return true;
};
