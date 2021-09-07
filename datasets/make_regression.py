from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(
	n_samples=128,
	n_features=10,
	noise=16,
	random_state=32
)

with open("regression_data.csv", "w", encoding="utf-8") as outF:
	# Let's do it in a stupid way.
	finalStr = "\n".join([",".join([str(val) for i, val in enumerate(sample)]) for sample in np.concatenate([X, np.expand_dims(y, axis=-1)], axis=-1)])
	outF.write(finalStr)
