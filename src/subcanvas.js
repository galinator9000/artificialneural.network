// Subcanvas system related variables and functions.

let subCanvas = {
	// Subcanvas objects 
	c: [
		{
			title: "Dataset",
			obj: null,
			isActive: () => true,
		},
		{
			title: "Neural Network",
			obj: null,
			isActive: () => (
				// NN GUI components are ready when data is ready
				(!data.isLoading) && data.isCompiled
			),
			eventHandlers: {
				mousePressed: (x, y) => {nn.mousePressed(x, y)}
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
	transform: {scale: {x: 1, y: 1}, translate: {x: 0, y: 0}},

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
		sc.xToSubcanvasPosX = (x) => {
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
		sc.yToSubcanvasPosY = (y) => {
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
switchSubcanvas = (switchIdx) => {
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
		subCanvas.transform = {scale: {x: 1, y: 1}, translate: {x: 0, y: 0}};

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
