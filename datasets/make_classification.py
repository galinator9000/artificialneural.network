from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(
	n_samples=128,
	n_features=5,
	n_classes=2,
	random_state=32
)

with open("binary_classification_data.csv", "w", encoding="utf-8") as outF:
	# Let's do it in a stupid way.
	finalStr = "\n".join([",".join([(str(val) if i != (len(sample)-1) else str(int(val))) for i, val in enumerate(sample)]) for sample in np.concatenate([X, np.expand_dims(y, axis=-1)], axis=-1)])
	outF.write(finalStr)
