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

	colorMode(RGB);
	textAlign(CENTER, CENTER);
	textFont(MAIN_FONT);

	// Create the sub-canvases
	createSubCanvas();

	// Initialize & Update once GUI components
	initializeGUI();
	updateGUI();

	// Load dataset
	loadDataset(Object.values(csvURLs)[0]).then(() => {compileDataset()});
};

// Main loop
draw = () => {
	rectMode(CORNER);
	angleMode(DEGREES);

	// Clear backgrounds
	background(BG_COLOR);
	Object.values(subCanvas.c).forEach(sc => {
		sc.obj.background(BG_COLOR);
	});

	// Update GUI components
	updateGUI();

	// Process nn if initialized
	if(nn){
		// Update network
		nn.update(
			subCanvas.c[1].obj,
			// Additional vArgs
			{
				mouseX: subCanvas.xToSubcanvasPosX(mouseX),
				mouseY: subCanvas.yToSubcanvasPosY(mouseY)
			}
		);

		// Check if neural network should be drawn to it's subcanvas
		if(subCanvas.c[1].shouldDraw()){
			// Draw network
			nn.draw(
				subCanvas.c[1].obj,
				data.stageSample
			);
		}
	}

	// Check if dataset should be drawn to it's subcanvas
	if(subCanvas.c[0].shouldDraw()){
		// Draw dataset on given subcanvas
		drawDataset(
			subCanvas.c[0].obj,
			// Additional vArgs
			{
				scaleX: 0.90, scaleY: 0.75,
				mouseX: subCanvas.xToSubcanvasPosX(mouseX),
				mouseY: subCanvas.yToSubcanvasPosY(mouseY)
			}
		);
	}

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

	// Calculate tab titles' text size & apply it
	textSize(calculateTextsSize(subCanvas.c.map(sc => sc.title), eachTabH));

	// Draw each tab title
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

// Processes mouse wheel events
mouseWheel = (event) => {
	// Change the current subcanvas (tab)
	if(event.deltaY > 0) switchSubcanvas(subCanvas.currentIdx+1);
	if(event.deltaY < 0) switchSubcanvas(subCanvas.currentIdx-1);
};

// Processes mouse press events
mousePressed = (event) => {
	// Reject event if it didn't occur on the main canvas
	if(event.path[0].className !== "p5Canvas") return true;
	// Reject event during transition
	if(subCanvas.inTransition) return true;

	let subcanvasStartX = (windowWidth * subCanvas.leftTabWidthRatio);
	let eachTabH = (windowHeight / subCanvas.c.length);

	// Check if it's on the main canvas
	if(event.x <= subcanvasStartX){
		let clickedSubcanvasIndex = Math.floor(event.y / eachTabH);

		// Switch tab
		switchSubcanvas(clickedSubcanvasIndex);
	}
	// Pass event to subcanvas
	else{
		// Get current subcanvas' events
		let events = subCanvas.c[subCanvas.currentIdx].events;

		// Check if event exists, call it
		if(events && events.mousePressed){
			events.mousePressed(
				subCanvas.xToSubcanvasPosX(event.x),
				subCanvas.yToSubcanvasPosY(event.y)
			);
		}
	}

	return true;
};
