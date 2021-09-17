// SubCanvas system related variables and functions.

let subCanvas = {
	// SubCanvas objects 
	c: [
		{
			title: "Dataset",
			obj: null,
			isActive: () => true,
			// Event handlers should return true/false if they were able to process the event or not
			eventHandlers: {
				mousePressed: (x, y) => {
					console.log("Dataset mousePressed", x, y);
					return false;
				},
				mouseWheel: (x, y, delta) => {
					if(delta < 0) zoomSubCanvas(1);
					if(delta > 0) zoomSubCanvas(-1);
					return true;
				},
			},
		},
		{
			title: "Neural Network",
			obj: null,
			isActive: () => (
				// NN GUI components are ready when data is ready
				(!data.isLoading) && data.isCompiled
			),
			// Event handlers should return true/false if they were able to process the event or not
			eventHandlers: {
				mousePressed: (x, y) => {
					nn.mousePressed(x, y);
					return true;
				},
				mouseWheel: (x, y, delta) => {
					if(delta < 0) zoomSubCanvas(1);
					if(delta > 0) zoomSubCanvas(-1);
					return true;
				},
			},
		},
		{
			title: "Stats",
			obj: null,
			isActive: () => (
				// Statistics GUI components are ready when nn&data is ready
				(!data.isLoading) && data.isCompiled && (nn && nn.isCompiled)
			),
		},
	],

	currentIdx: 0,
	nextIdx: 0,

	// Transforms of current canvas
	transform: {
		scale: {x: 1, y: 1, targetX: 1, targetY: 1},
		translate: {x: 0, y: 0, targetX: 0, targetY: 0}
	},
	
	// Constants
	zoomFactor: 0.25,
	zoomMin: 0.5,
	zoomMax: 3,
	movementSpeed: 0.25,

	// Animation of subcanvas transitions
	inTransition: false,
	transition: {
		x: 0.0,
		xAnim: 0.0,
		direction: 1,
		step: 0.05,
		animFn: AnimationUtils.easeOutExpo
	},
	leftTabWidthRatio: 0.05,
};

getSubCanvasWidthWithIndex = (cIdx) => {
	if(cIdx == -1) return windowWidth;
	return subCanvas.c[cIdx].obj.width;
};
getSubCanvasHeightWithIndex = (cIdx) => {
	if(cIdx == -1) return windowHeight;
	return subCanvas.c[cIdx].obj.height;
};

// Updates subcanvas related things
updateSubCanvas = () => {
	// Update subcanvas transition value
	if(subCanvas.inTransition){
		// Step x
		subCanvas.transition.x += subCanvas.transition.step;

		// Limit between 0 and 1
		subCanvas.transition.x = Math.min(1, Math.max(0, subCanvas.transition.x));

		// Calculate animation value
		subCanvas.transition.xAnim = subCanvas.transition.animFn(subCanvas.transition.x);

		// Stop & reset transition when reached to the 1
		if(subCanvas.transition.x >= 1.0){
			subCanvas.transition.x = 0.0;
			subCanvas.transition.xAnim = 0.0;
			subCanvas.transition.direction = 1;
			subCanvas.inTransition = false;

			// Set current to the transitioned one
			subCanvas.currentIdx = subCanvas.nextIdx;
		}
	}

	// Smoothly go towards the targets in transform values
	subCanvas.transform.translate.x += ((subCanvas.transform.translate.targetX - subCanvas.transform.translate.x) * subCanvas.movementSpeed);
	subCanvas.transform.translate.y += ((subCanvas.transform.translate.targetY - subCanvas.transform.translate.y) * subCanvas.movementSpeed);
	subCanvas.transform.scale.x += ((subCanvas.transform.scale.targetX - subCanvas.transform.scale.x) * subCanvas.movementSpeed);
	subCanvas.transform.scale.y += ((subCanvas.transform.scale.targetY - subCanvas.transform.scale.y) * subCanvas.movementSpeed);
};

// Creates sub canvas objects
createSubCanvas = () => {
	Object.entries(subCanvas.c).forEach(([k, v], cIdx) => {
		// Remove canvas first, if exists
		if(subCanvas.c[cIdx].obj) subCanvas.c[cIdx].obj.remove();

		// Create a fresh one
		subCanvas.c[cIdx].obj = createGraphics(
			(windowWidth * (1 - subCanvas.leftTabWidthRatio)),
			windowHeight
		);
		let sc = subCanvas.c[cIdx];

		// Mark the value of starting X position of subcanvases
		subCanvas.subcanvasStartX = (windowWidth * subCanvas.leftTabWidthRatio);

		// Apply canvas settings which will not be changed
		sc.obj.colorMode(RGB);
		sc.obj.angleMode(DEGREES);
		sc.obj.textFont(MAIN_FONT);
		sc.obj.textAlign(CENTER, CENTER);
		sc.obj.rectMode(CENTER, CENTER);

		// Specify default values while drawing (unless otherwise specified)
		sc.obj.noFill();
		sc.obj.stroke(255);
		sc.obj.strokeWeight(1);

		// Assign utility functions of the subcanvas for later use
		// Position converter functions (with applying transformations in reverse)
		sc.xToSubCanvasPosX = (x) => {
			// Calculate tab
			x = (x - (windowWidth * subCanvas.leftTabWidthRatio));

			// Apply transformations in reverse
			// Translate
			x -= subCanvas.transform.translate.x;
			// Scale
			x -= (sc.obj.width/2);
			x /= subCanvas.transform.scale.x;
			x += (sc.obj.width/2);

			return x;
		};
		sc.yToSubCanvasPosY = (y) => {
			// Apply transformations in reverse
			// Translate
			y -= subCanvas.transform.translate.y;
			// Scale
			y -= (sc.obj.height/2);
			y /= subCanvas.transform.scale.y;
			y += (sc.obj.height/2);
			
			return y;
		};
	});

	// Set shouldDraw functions of subcanvases
	subCanvas.c[0].shouldDraw = () => ([subCanvas.currentIdx, subCanvas.nextIdx].includes(0));
	subCanvas.c[1].shouldDraw = () => ([subCanvas.currentIdx, subCanvas.nextIdx].includes(1));
	subCanvas.c[2].shouldDraw = () => ([subCanvas.currentIdx, subCanvas.nextIdx].includes(2));
};

// Switches (transition) to given subcanvas idx
switchSubCanvas = (switchIdx) => {
	// Return if already in transition process
	if(subCanvas.inTransition) return;
	
	// Limit index number
	subCanvas.nextIdx = Math.min(
		subCanvas.c.length-1,
		Math.max(0, switchIdx)
	);

	// Apply transition to given subcanvas
	if(subCanvas.nextIdx != subCanvas.currentIdx){
		// Reset transform
		subCanvas.transform = {
			scale: {x: 1, y: 1, targetX: 1, targetY: 1},
			translate: {x: 0, y: 0, targetX: 0, targetY: 0}
		};

		// Start transition
		subCanvas.transition.x = 0;
		subCanvas.transition.xAnim = 0;
		subCanvas.transition.direction = (subCanvas.nextIdx > subCanvas.currentIdx) ? -1 : 1;
		subCanvas.inTransition = true;
	}
};

// Applies transformations to given canvas
applyTransformationsToSubCanvas = (canvas) => {
	// Apply scaling
	// Translate to center for scaling & then back
	canvas.translate(canvas.width/2, canvas.height/2);
	canvas.scale(...Object.values(subCanvas.transform.scale));
	canvas.translate(-(canvas.width/2), -(canvas.height/2));
	
	// Apply translation
	canvas.translate(...Object.values(subCanvas.transform.translate));
};

// Sets current scale values on transformation
zoomSubCanvas = (z) => {
	// New values
	let newScaleX = (subCanvas.transform.scale.x + (subCanvas.zoomFactor * z));
	let newScaleY = (subCanvas.transform.scale.y + (subCanvas.zoomFactor * z));

	// Limit new scaling values, set as target
	subCanvas.transform.scale.targetX = Math.min(Math.max(newScaleX, subCanvas.zoomMin), subCanvas.zoomMax);
	subCanvas.transform.scale.targetY = Math.min(Math.max(newScaleY, subCanvas.zoomMin), subCanvas.zoomMax);
};
