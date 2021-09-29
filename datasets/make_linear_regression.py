from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(
	n_samples=128,
	n_features=1,
	noise=16,
	random_state=32
)
if len(y.shape) <= 1: y = np.expand_dims(y, axis=-1)

with open("one_variable_linear_regression.csv", "w", encoding="utf-8") as outF:
	# Write headers
	outF.write(",".join([
		str(i) for i in list(range((X.shape[1] + y.shape[1])))
	]) + "\n")

	# Let's do it in a stupid way.
	finalStr = "\n".join([",".join([str(val) for i, val in enumerate(sample)]) for sample in np.concatenate([X, y], axis=-1)])
	outF.write(finalStr)
