// Constants
let MAIN_FONT;
let BG_COLOR;
preload = () => {
	MAIN_FONT = loadFont("assets/Inconsolata-SemiBold.ttf");
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

	// Load dataset
	loadDataset(Object.values(csvURLs)[0]).then(() => {
		compileDataset();
	});
};

// Main loop
draw = () => {
	// Clear backgrounds
	background(BG_COLOR);
	Object.values(subCanvas.c).forEach(sc => {
		sc.obj.background(BG_COLOR);
	});

	// Update GUI components
	updateGUI();

	// Process nn if initialized
	if(nn){
		// Get the canvas
		let nnCanvas = subCanvas.c[1];

		// Update network
		nn.update(nnCanvas.obj);

		// Check if neural network should be drawn to it's subcanvas
		if(nnCanvas.shouldDraw()){
			nnCanvas.obj.push();

			// Apply transformations to the nn canvas
			applyTransformationsToSubCanvas(nnCanvas.obj);

			// Draw network
			nn.draw(nnCanvas.obj, data.stageSample);
			
			nnCanvas.obj.pop();
		}
	}

	// Check if dataset should be drawn to it's subcanvas
	let datasetCanvas = subCanvas.c[0];
	if(datasetCanvas.shouldDraw()){
		datasetCanvas.obj.push();

		// Apply transformations to the dataset canvas
		applyTransformationsToSubCanvas(datasetCanvas.obj);

		// Draw dataset on given subcanvas
		drawDataset(
			datasetCanvas.obj,
			// Additional vArgs
			{scaleX: 0.90, scaleY: 0.85, translateX: 0.00, translateY: 0.05}
		);

		datasetCanvas.obj.pop();
	}

	// Update subcanvas related things
	updateSubCanvas();

	//// Draw subcanvases over main one
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

		// Draw subcanvases on to main canvas
		push();
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
		pop();
	}
	// Draw only the current sub-canvas if transition isn't happening
	else{
		push();
		image(
			currentCanvas,
			// Position
			startX, 0,
			// Size
			currentCanvas.width, currentCanvas.height
		);
		pop();
	}

	//// Draw subcanvas tabs to the left of the screen
	let eachTabW = (windowWidth * subCanvas.leftTabWidthRatio);
	let eachTabH = (windowHeight / subCanvas.c.length);
	let curScIdx = (subCanvas.nextIdx);

	// Calculate tab titles' text size & apply it
	textSize(calculateTextsSize(subCanvas.c.map(sc => sc.title), eachTabH));

	// Draw each tab title
	subCanvas.c.forEach((sc, scIdx) => {
		// Active tab underline
		push();
		strokeWeight((scIdx == curScIdx) ? 3 : 0);
		line(eachTabW, (eachTabH * (scIdx)), eachTabW, (eachTabH * (scIdx+1)));
		pop();

		// Write tab title (use translate&rotate for drawing titles sideways)
		push();
		fill((scIdx == curScIdx) ? 255 : 64);
		stroke((scIdx == curScIdx) ? 255 : 64);
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

// Processes mouse wheel events
mouseWheel = (event) => {
	// Reject event during transition (should return true!)
	if(subCanvas.inTransition) return true;

	// Get current subcanvas
	let sc = subCanvas.c[subCanvas.currentIdx];
	let eventProcessed = false;

	// Check if event occured at subcanvas region and event exists.
	if((event.x > subCanvas.subcanvasStartX) && sc.eventHandlers && sc.eventHandlers.mouseWheel){
		// Call the event handler
		eventProcessed = sc.eventHandlers.mouseWheel(
			sc.xToSubCanvasPosX(event.x),
			sc.yToSubCanvasPosY(event.y),
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

// Processes mouse press events
mousePressed = (event) => {
	// Reject event if it didn't occur on the main canvas (should return true!)
	if(event.path[0].className !== "p5Canvas") return true;
	// Reject event during transition (should return true!)
	if(subCanvas.inTransition) return true;

	// Get current subcanvas
	let sc = subCanvas.c[subCanvas.currentIdx];
	let eventProcessed = false;

	// Check if event occured at subcanvas region and event exists.
	if((event.x > subCanvas.subcanvasStartX) && sc.eventHandlers && sc.eventHandlers.mousePressed){
		// Call the event handler
		eventProcessed = sc.eventHandlers.mousePressed(
			sc.xToSubCanvasPosX(event.x),
			sc.yToSubCanvasPosY(event.y)
		);
	}

	//// If subcanvases didn't process the event, main functionality handles it
	if(!eventProcessed){
		// Switch clicked tab
		let eachTabH = (windowHeight / subCanvas.c.length);
		let clickedSubCanvasIndex = Math.floor(event.y / eachTabH);
		switchSubCanvas(clickedSubCanvasIndex);
	}

	return true;
};
