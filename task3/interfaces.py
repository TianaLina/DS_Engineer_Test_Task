from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class DigitClassificationInterface(ABC):
    """Common interface for all MNIST digit classification models."""

    @abstractmethod
    def preprocess(self, image: np.ndarray):
        """Convert a (28, 28, 1) image into the input format of the model."""

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the model."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Take a (28, 28, 1) image and return a digit 0-9."""


class CNNModel(DigitClassificationInterface):
    """Wraps an already trained CNN (e.g. a Keras model).

    Expected input: tensor of shape (28, 28, 1); the model gets a batch
    of shape (1, 28, 28, 1) and must return class probabilities via
    `model.predict(batch)`.
    """

    def __init__(self, model=None):
        self.model = model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image.reshape(1, 28, 28, 1)  # add batch dimension

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is not implemented for CNNModel.")

    def predict(self, image: np.ndarray) -> int:
        if self.model is None:
            raise RuntimeError("CNNModel has no trained model attached.")
        batch = self.preprocess(image)
        probs = np.asarray(self.model.predict(batch))
        return int(np.argmax(probs, axis=-1).ravel()[0])


class RandomForest(DigitClassificationInterface):
    """Random Forest on a flat vector of length 784."""

    def __init__(self, model: RandomForestClassifier = None):
        self.model = model if model is not None else RandomForestClassifier()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image.reshape(1, 784)  # sklearn expects 2D: (n_samples, 784)

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is not implemented for RandomForest.")

    def predict(self, image: np.ndarray) -> int:
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise RuntimeError("RandomForest model is not trained/fitted.")
        data = self.preprocess(image)
        return int(self.model.predict(data)[0])


class RandomModel(DigitClassificationInterface):
    """Returns a random digit. Input: 10x10 center crop."""

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        return image[9:19, 9:19, :].reshape(10, 10)

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is not implemented for RandomModel.")

    def predict(self, image: np.ndarray) -> int:
        _ = self.preprocess(image)
        return int(np.random.randint(0, 10))


class DigitClassifier:
    """Single entry point: same input/output regardless of the algorithm.

    Usage:
        DigitClassifier("rand").predict(img)
        DigitClassifier("rf", model=fitted_rf).predict(img)
        DigitClassifier("cnn", model=trained_cnn).predict(img)
    """

    ALGORITHMS = {
        "cnn": CNNModel,
        "rf": RandomForest,
        "rand": RandomModel,
    }

    def __init__(self, algorithm: str, model=None):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm!r}. "
                f"Choose from {list(self.ALGORITHMS)}."
            )
        cls = self.ALGORITHMS[algorithm]
        # RandomModel needs no underlying model; others accept an optional one
        self.model: DigitClassificationInterface = (
            cls() if algorithm == "rand" else cls(model)
        )

    def predict(self, image: np.ndarray) -> int:
        image = np.asarray(image)
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected image shape (28, 28, 1), got {image.shape}")
        return int(self.model.predict(image))

    def train(self, *args, **kwargs):
        raise NotImplementedError("Training is not implemented in DigitClassifier.")


if __name__ == "__main__":
    img = np.random.randint(0, 256, size=(28, 28, 1))
    print("rand:", DigitClassifier("rand").predict(img))

    # Quick demo for rf with a dummy fit (replace with real MNIST training)
    rf = RandomForestClassifier(n_estimators=5).fit(
        np.random.rand(50, 784), np.random.randint(0, 10, 50)
    )
    print("rf:", DigitClassifier("rf", model=rf).predict(img))
