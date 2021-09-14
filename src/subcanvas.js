// Subcanvas system related variables and functions.

let subCanvas = {
	// Subcanvas objects 
	c: [
		{
			title: "Dataset",
			obj: null,
		},
		{
			title: "Neural Network",
			obj: null,
		},
		{
			title: "Stats",
			obj: null,
		},
	],

	currentIdx: 1,
	nextIdx: 1,

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
// Position converter functions
subCanvas.xToSubcanvasPosX = (x) => (x - (windowWidth * subCanvas.leftTabWidthRatio));
subCanvas.yToSubcanvasPosY = (y) => (y);

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
		subCanvas.c[cIdx].obj = createGraphics(
			(windowWidth * (1 - subCanvas.leftTabWidthRatio)),
			windowHeight
		);
	});
};
