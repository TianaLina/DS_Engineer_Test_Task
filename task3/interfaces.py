from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class DigitClassificationInterface(ABC):
    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def train(self, image):
        pass

    @abstractmethod
    def predict(self, image) -> int:
        pass
    



class RandomForest(DigitClassificationInterface):

    # def __init__(self):
    #     self.model = RandomForestClassifier()

    def preprocess(self, image):
        image = np.asarray(image)
        flattened = image.reshape(784)

        return flattened

    def predict(self, image:np.ndarray) -> int:
        data = self.preprocess(image)

        return "Here will be the prediction"

    def train(self, image):
        pass

class RandomModel(DigitClassificationInterface):

    def preprocess(self, image):
        image = np.asarray(image)

        # Center crop: (28, 28, 1) -> (10, 10, 1)
        cropped = image[9:19, 9:19, :]
        
        return cropped

    def predict(self, image: np.ndarray) -> int:

        input = self.preprocess(image)
        return int(np.random.randint(0, 10))


# class DigitClassifier:

#     def __init__(self, algorithm: str, models: dict):
#         algorithms = {
#             "cnn": CNNModel,
#             "rf": RandomForest,
#             "rand": RandomModel,
#         }

#         if algorithm not in algorithms:
#             raise ValueError(
#                 f"Unknown algorithm: {algorithm}. "
#                 f"Choose from {list(algorithms)}."
#             )

#         if algorithm == "rand":
#             self.model = RandomModel()
#         else:
#             if algorithm not in models:
#                 raise ValueError(
#                     f"A model instance must be provided for '{algorithm}'."
#                 )
#             self.model = algorithms[algorithm](models[algorithm])

#     def predict(self, image: np.ndarray) -> int:
#         image = np.asarray(image)

#         if image.shape != (28, 28, 1):
#             raise ValueError(
#                 f"Expected image shape (28, 28, 1), got {image.shape}"
#             )

#         return int(self.model.predict(image))

#     def train(self, *args, **kwargs):
#         raise NotImplementedError(
#             "Training is not implemented in DigitClassifier."
#         )

model_forest = RandomForest()
model_forest.preprocess()