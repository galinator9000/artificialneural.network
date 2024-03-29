// SubCanvas system related variables and functions.

const HOME_SUBCANVAS_INDEX = 0;
const DATASET_SUBCANVAS_INDEX = 1;
const NN_SUBCANVAS_INDEX = 2;
const HOW_TO_SUBCANVAS_INDEX = 3;

const INITIAL_SUBCANVAS_INDEX = HOME_SUBCANVAS_INDEX;

var subCanvas = {
	// SubCanvas objects 
	c: [
		{
			title: "Home",
			obj: null,
			isActive: () => true,
			isDisplayedOnMenu: true,
			eventHandlers: {
				mouseWheel: (x, y, delta) => {
					return true;
				},
				mouseDragged: (mx, my) => {
					return true;
				},
			},
		},
		{
			title: "Dataset",
			obj: null,
			isActive: () => true,
			isDisplayedOnMenu: true,
			// Event handlers should return true/false if they were able to process the event or not
			eventHandlers: {
				mouseWheel: (x, y, delta) => {
					// Run scrolling
					if(scrollDataset(x, y, delta)) return true;
					// or zooming functionality
					// else return zoomSubCanvas(x, y, delta);
					return true;
				},
				mouseDragged: (mx, my) => {
					// return dragSubCanvas(mx, my);
					return true;
				},
			},
		},
		{
			title: "Neural Network",
			obj: null,
			isActive: () => (
				// NN GUI components are ready when data is ready
				(!data.isLoading) && data.isCompiled && (nn !== undefined)
			),
			isDisplayedOnMenu: true,
			// Event handlers should return true/false if they were able to process the event or not
			eventHandlers: {
				mouseClicked: (x, y) => {
					nn.mouseClicked(x, y);
					return true;
				},
				mouseWheel: (x, y, delta) => {
					return zoomSubCanvas(x, y, delta);
				},
				mouseDragged: (mx, my) => {
					return dragSubCanvas(mx, my);
				},
			},
		},
		{
			title: "How To",
			obj: null,
			isActive: () => true,
			isDisplayedOnMenu: false,
			eventHandlers: {
				mouseWheel: (x, y, delta) => {
					return zoomSubCanvas(x, y, delta);
				},
				mouseDragged: (mx, my) => {
					return dragSubCanvas(mx, my);
				},
			},
		},
	],

	// Keeps the value of the current subcanvas' index (& next one if in transition)
	currentIdx: INITIAL_SUBCANVAS_INDEX, nextIdx: INITIAL_SUBCANVAS_INDEX,

	// Transforms of current canvas
	transform: {
		scale: {x: 1, y: 1, targetXY: 1},
		translate: {x: 0, y: 0, targetX: 0, targetY: 0}
	},
	
	//// Constants

	// Transform target value reach speed
	targetValueSpeed: 0.25,

	// Zoom consts
	zoomFactor: 0.25,
	zoomMin: 0.33,
	zoomMax: 10,

	// Drag consts
	dragFactor: 2,
	dragLimitRatio: 0.80,

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
shouldSubCanvasBeDrawn = (cIdx) => ([subCanvas.currentIdx, subCanvas.nextIdx].includes(cIdx));

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

	// Update each transform value according to their target value
	// Translation values
	subCanvas.transform.translate.x += ((subCanvas.transform.translate.targetX - subCanvas.transform.translate.x) * subCanvas.targetValueSpeed);
	subCanvas.transform.translate.y += ((subCanvas.transform.translate.targetY - subCanvas.transform.translate.y) * subCanvas.targetValueSpeed);

	// Scaling happens on both axis for preserving ratio
	subCanvas.transform.scale.x += ((subCanvas.transform.scale.targetXY - subCanvas.transform.scale.x) * subCanvas.targetValueSpeed);
	subCanvas.transform.scale.y += ((subCanvas.transform.scale.targetXY - subCanvas.transform.scale.y) * subCanvas.targetValueSpeed);
};

// Creates sub canvas objects
createSubCanvas = () => {
	[...Array(subCanvas.c.length).keys()].forEach((scIndex) => {
		// Remove canvas first, if exists
		if(subCanvas.c[scIndex].obj) subCanvas.c[scIndex].obj.remove();

		// Create a fresh one
		subCanvas.c[scIndex].obj = createGraphics(
			(windowWidth * (1 - subCanvas.leftTabWidthRatio)),
			windowHeight
		);
		let sc = subCanvas.c[scIndex];

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

		// Position converter functions
		sc.absolutePos_to_SubCanvasPos = (x, y) => {
			let vec = createVector(x, y);

			// Mind the tab offset
			vec.x -= subCanvas.subcanvasStartX;

			// Apply transformations, in reverse

			// Scale
			vec.add(-(sc.obj.width/2), -(sc.obj.height/2));
			vec.mult(createVector(1/subCanvas.transform.scale.x, 1/subCanvas.transform.scale.y));
			vec.add((sc.obj.width/2), (sc.obj.height/2));

			// Translate
			vec.add(-subCanvas.transform.translate.x, -subCanvas.transform.translate.y);

			return vec;
		};
		sc.subCanvasPos_to_absoluteCanvasPos = (x, y) => {
			let vec = createVector(x, y);

			// Translate
			vec.add(subCanvas.transform.translate.x, subCanvas.transform.translate.y);

			// Scale
			vec.add(-(sc.obj.width/2), -(sc.obj.height/2));
			vec.mult(createVector(subCanvas.transform.scale.x, subCanvas.transform.scale.y));
			vec.add((sc.obj.width/2), (sc.obj.height/2));

			// Mind the tab offset
			vec.x += subCanvas.subcanvasStartX;
			
			return vec;
		};
	});
};

// Switches (transition) to given subcanvas idx
switchSubCanvas = (switchIdx) => {
	// C'mon bruh
	if(switchIdx < 0 || switchIdx >= subCanvas.c.length) return;
	// Return if already in transition process
	if(subCanvas.inTransition) return;
	// Return if the canvas isn't active
	if(!subCanvas.c[switchIdx].isActive()) return;
	
	// Limit index number and set the next id for transition
	subCanvas.nextIdx = Math.min(
		subCanvas.c.length-1,
		Math.max(0, switchIdx)
	);

	// Apply transition to given subcanvas
	if(subCanvas.nextIdx != subCanvas.currentIdx){
		resetSubCanvasTransforms();

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

// Sets current scaling values on transformation
zoomSubCanvas = (x, y, delta) => {
	if(Math.abs(delta) === 0) return;

	// Get mouse wheel delta's direction as 1 or -1, reverse it for zooming direction
	let direction = -(delta / Math.abs(delta));

	// New value
	let newScale = (subCanvas.transform.scale.x + (
		(subCanvas.zoomFactor * direction) * subCanvas.transform.scale.targetXY
	));

	if(isNaN(newScale) || isNaN(direction)) return;

	// Limit new scaling value, set as target for both XY
	subCanvas.transform.scale.targetXY = Math.min(Math.max(newScale, subCanvas.zoomMin), subCanvas.zoomMax);

	return true;
};

// Sets current translation values on transformation
dragSubCanvas = (mx, my) => {
	if(
		Math.abs(mx) === 0 && Math.abs(my) === 0
	) return;

	// New XY values
	let newTranslateX = (subCanvas.transform.translate.x + (
		// Scaling value affects the dragging factors inversely
		(mx * subCanvas.dragFactor) / subCanvas.transform.scale.targetXY
	));
	let newTranslateY = (subCanvas.transform.translate.y + (
		(my * subCanvas.dragFactor) / subCanvas.transform.scale.targetXY
	));

	// Set new translation values as target
	subCanvas.transform.translate.targetX = newTranslateX;
	subCanvas.transform.translate.targetY = newTranslateY;

	// Limit new translation values with calculating boundaries
	// let minX = -(subCanvas.c[subCanvas.currentIdx].obj.width * subCanvas.dragLimitRatio);
	// let maxX = (subCanvas.c[subCanvas.currentIdx].obj.width * subCanvas.dragLimitRatio);
	// let minY = -(subCanvas.c[subCanvas.currentIdx].obj.height * subCanvas.dragLimitRatio);
	// let maxY = (subCanvas.c[subCanvas.currentIdx].obj.height * subCanvas.dragLimitRatio);
	// subCanvas.transform.translate.targetX = Math.min(Math.max(subCanvas.transform.translate.targetX, minX), maxX);
	// subCanvas.transform.translate.targetY = Math.min(Math.max(subCanvas.transform.translate.targetY, minY), maxY);

	return true;
};

resetSubCanvasTransforms = () => {
	// Reset transforms
	subCanvas.transform = {
		scale: {x: 1, y: 1, targetXY: 1},
		translate: {x: 0, y: 0, targetX: 0, targetY: 0}
	};
};
