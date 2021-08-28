let WIDTH = 1024;
let HEIGHT = 768;

function setup(){
	// Create p5 canvas
	createCanvas(WIDTH, HEIGHT);

	// Test tensorflow.js
	const a = tf.tensor([[1, 2], [3, 4]]);
	console.log("Shape: ", a.shape);
	a.print();
}

function draw(){
	// Background
	background(1, 0, 2, 255);
}
