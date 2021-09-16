// Linear interpolation
const lerp = (value, target, amount) => {
	return (value + ((target - value) * amount));
};

const arrSum = (arr) => arr.reduce((a, b) => (a + b), 0);
const arrBoolAny = (arr) => arr.reduce((a, b) => (a || b), false);
const arrBoolAll = (arr) => arr.reduce((a, b) => (a && b), true);

// Calculates text size with given text
const calculateTextSize = (text, areaW, areaH) => {
	let s = Math.floor(
		areaW / text.length
	);

	return s;
};

// Calculates text size with given text list
const calculateTextsSize = (texts, areaW) => {
	return Math.floor(
		// Area / Mean of character count
		areaW / (arrSum(texts.map(t => t.length)) / texts.length)
	)
};

// Random int
const getRandomInt = (min, max) => {
	min = Math.ceil(min);
	max = Math.floor(max);
	return Math.floor(Math.random() * (max - min) + min);
}

// Opens link in new tab
const openURLInNewTab = (url) => {
	window.open(url, "_blank").focus();
};

// Animation related functions
class AnimationUtils{
	// Exponential
	static easeOutExpo = (x) => {
		if(x == 1.0){
			return 1.0;
		}else{
			return (1.0 - Math.pow(2, -10.0 * x));
		}
	}
	static easeInExpo = (x) => {
		if(x == 0.0){
			return 0.0;
		}else{
			return (Math.pow(2, (10.0 * x) - 10.0));
		}
	}

	// Cubic
	static easeInCubic = (x) => {
		return (x*x*x);
	}
	static easeOutCubic = (x) => {
		return (1.0 - Math.pow(1-x, 3));
	}

	// Quad
	static easeInQuad = (x) => {
		return (x*x);
	}
	static easeOutQuad = (x) => {
		return (1.0 - ((1.0 - x) * (1.0 - x)));
	}

	static linear = (x) => {
		return x;
	}
};
