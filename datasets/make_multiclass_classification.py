from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
	n_samples=128,
	n_features=10,
	n_informative=5,
	n_redundant=0,
	n_classes=5,
	random_state=32
)
if len(y.shape) <= 1: y = np.expand_dims(y, axis=-1)

with open("multiclass_classification.csv", "w", encoding="utf-8") as outF:
	# Write headers
	outF.write(",".join([
		str(i) for i in list(range((X.shape[1] + y.shape[1])))
	]) + "\n")

	# Let's do it in a stupid way.
	finalStr = "\n".join([",".join([(str(val) if i != (len(sample)-1) else str(int(val))) for i, val in enumerate(sample)]) for sample in np.concatenate([X, y], axis=-1)])
	outF.write(finalStr)
