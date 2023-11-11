// Draws the how to page on the given canvas
drawHowToCanvas = (canvas) => {
	// "What even is this" title
	canvas.push();
	canvas.translate(
		canvas.width/2,
		canvas.height*0.05
	);
	canvas.fill(255);
	canvas.textSize(34);
	canvas.text("What even is this?", 0, 0);
	canvas.pop();

	// Draw the whole paragraph right here
	canvas.push();
	canvas.translate(
		canvas.width/2,
		canvas.height*0.47
	);
	canvas.textAlign(CENTER);
	drawParagraph(
		canvas,
		howToBody
	);
	canvas.pop();
};

drawParagraph = (canvas, text) => {
	canvas.fill(255);
	canvas.strokeWeight(0);
	canvas.textStyle(NORMAL);
	canvas.textSize(18);
	canvas.text(text, 0, 0);
}